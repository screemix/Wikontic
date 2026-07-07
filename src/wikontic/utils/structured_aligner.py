from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
import torch
from dotenv import load_dotenv, find_dotenv
from wikontic.db.factory import ensure_storage_backend
from wikontic.db.interfaces import VectorQuery
from wikontic.utils.embedding_model import (
    SUPPORTED_MODELS,
    load_embedding_model,
    get_embedding as _get_embedding,
    get_embeddings as _get_embeddings,
)
from wikontic.utils.language_config import default_embedding_model_for_language

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
_ = load_dotenv(find_dotenv())


@dataclass
class PropertyConstraints:
    subject_properties: Set[str]
    object_properties: Set[str]


class EntityAlias(BaseModel):
    _id: int
    label: str
    entity_type: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]


class Aligner:
    def __init__(self, ontology_db, triplets_db, device=None, embedding_model: str = None, language: str = None):
        if embedding_model is None:
            embedding_model = default_embedding_model_for_language(language)
        if embedding_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown embedding model {embedding_model!r}. "
                f"Supported values: {sorted(SUPPORTED_MODELS)}"
            )
        self.ontology_db = ensure_storage_backend(ontology_db)
        self.triplets_db = ensure_storage_backend(triplets_db)

        self.entity_type_collection_name = "entity_types"
        self.entity_type_aliases_collection_name = "entity_type_aliases"
        self.property_collection_name = "properties"
        self.property_aliases_collection_name = "property_aliases"

        self.entity_type_vector_index_name = "entity_type_aliases"
        self.property_vector_index_name = "property_aliases"

        self.entity_aliases_collection_name = "entity_aliases"
        self.triplets_collection_name = "triplets"
        self.filtered_triplets_collection_name = "filtered_triplets"
        self.ontology_filtered_triplets_collection_name = "ontology_filtered_triplets"
        self.initial_triplets_collection_name = "initial_triplets"
        self.entities_vector_index_name = "entity_aliases"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model_name = embedding_model
        self.tokenizer, self.model, self.device = load_embedding_model(
            embedding_model, device=str(device)
        )

    def get_embedding(self, text):
        return _get_embedding(
            text, self.tokenizer, self.model, self.device, self.embedding_model_name
        )

    def get_embeddings(self, texts):
        return _get_embeddings(
            texts, self.tokenizer, self.model, self.device, self.embedding_model_name
        )

    def _get_unique_similar_entity_types(
        self, target_entity_type: str, k: int = 5, max_attempts: int = 10, embedding: Optional[List[float]] = None
    ) -> List[str]:
        # retrieve k most similar entity types to the given triplet
        # using the entity type index
        # return the wikidata ids of the most similar entity types

        query_k = k * 2
        attempt = 0
        unique_ranked_entities: List[str] = []
        query_embedding = embedding if embedding is not None else self.get_embedding(target_entity_type)
        if query_embedding is None:
            return []

        # as we search among aliases, there can be duplicated original entitites
        # and as we want K unique entities in result, we querying the index until we get exactly K unique entities
        while len(unique_ranked_entities) < k and attempt < max_attempts:
            result = self.ontology_db.vector_search(
                VectorQuery(
                    collection_name=self.entity_type_aliases_collection_name,
                    index_name=self.entity_type_vector_index_name,
                    query_vector=query_embedding,
                    vector_field="alias_text_embedding",
                    limit=query_k,
                    projection={"_id": 0, "entity_type_id": 1},
                )
            )
            for res in result:
                if res.get("entity_type_id") not in unique_ranked_entities:
                    unique_ranked_entities.append(res["entity_type_id"])
                if len(unique_ranked_entities) == k:
                    break
            query_k *= 2
            attempt += 1

        return unique_ranked_entities

    def retrieve_similar_entity_types(
        self,
        triplet: Dict[str, str],
        k: int = 10,
        subject_type_embedding: Optional[List[float]] = None,
        object_type_embedding: Optional[List[float]] = None,
    ) -> Tuple[List[str], List[str]]:

        similar_subject_types = self._get_unique_similar_entity_types(
            target_entity_type=triplet["subject_type"], k=k, embedding=subject_type_embedding
        )
        if "object_type" in triplet:

            similar_object_types = self._get_unique_similar_entity_types(
                target_entity_type=triplet["object_type"], k=k, embedding=object_type_embedding
            )
        else:
            similar_object_types = []
        return similar_subject_types, similar_object_types

    def _get_valid_property_ids_by_entity_type(
        self, entity_type: str, is_object: bool = True
    ) -> Tuple[Set[str], Set[str]]:
        """
        Get direct and inverse properties for an entity type.

        Args:
            entity_type: The entity type to look up
            is_object: Whether this is an object type in triplet (True) or a subject type (False)
        """

        collection = self.ontology_db.get_collection(self.entity_type_collection_name)

        # Get extended types including supertypes
        extended_types = [entity_type, "ANY"]
        hirerarchy = collection.find_one(
            {"entity_type_id": entity_type}, {"parent_type_ids": 1, "_id": 0}
        )
        extended_types.extend(hirerarchy["parent_type_ids"])

        matched_types = collection.find(
            {"entity_type_id": {"$in": extended_types}},
            {"valid_subject_property_ids": 1, "valid_object_property_ids": 1, "_id": 0},
        )
        subject_props_set: Set[str] = set()
        object_props_set: Set[str] = set()
        for item in matched_types:
            subject_props_set.update(item.get("valid_subject_property_ids") or [])
            object_props_set.update(item.get("valid_object_property_ids") or [])
        subject_props = list(subject_props_set)
        object_props = list(object_props_set)

        if is_object:
            direct_props = set(object_props)
            inverse_props = set(subject_props)
        else:
            direct_props = set(subject_props)
            inverse_props = set(object_props)

        return direct_props, inverse_props

    def _get_ranked_properties(
        self,
        prop_2_direction: Dict[str, List[str]],
        target_property: str,
        k: int,
        embedding: Optional[List[float]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Rank properties based on similarity to target relation.
        """
        query_embedding = embedding if embedding is not None else self.get_embedding(target_property)
        if query_embedding is None:
            return []
        props = list(prop_2_direction.keys())

        query_k = k * 2
        max_attempts = 5
        attempt = 0
        unique_ranked_properties: List[str] = []

        while len(unique_ranked_properties) < k and attempt < max_attempts:

            similar_properties = self.ontology_db.vector_search(
                VectorQuery(
                    collection_name=self.property_aliases_collection_name,
                    index_name=self.property_vector_index_name,
                    query_vector=query_embedding,
                    vector_field="alias_text_embedding",
                    limit=query_k,
                    filters={"relation_id": {"$in": props}},
                    projection={"_id": 0, "relation_id": 1},
                )
            )

            for prop in similar_properties:
                if prop["relation_id"] not in unique_ranked_properties:
                    unique_ranked_properties.append(prop["relation_id"])
                if len(unique_ranked_properties) == k:
                    break

            query_k *= 2
            attempt += 1

        # taking into account directions of properties
        unique_ranked_properties_with_direction = []
        for prop_id in unique_ranked_properties:
            for direction in prop_2_direction[prop_id]:
                unique_ranked_properties_with_direction.append((prop_id, direction))
        return unique_ranked_properties_with_direction

    def retrieve_properties_for_entity_type(
        self,
        target_relation: str,  # relation from triplet
        object_types: List[str],
        subject_types: List[str],
        k: int = 10,
        embedding: Optional[List[float]] = None,
    ) -> List[Tuple[str, str]]:  # List of tuples (<property_id>, <property_direction>)
        """
        Retrieve and rank properties that match given entity types and relation.

        Args:
            target_relation: The relation to search for
            object_types: List of valid object types
            subject_types: List of valid subject types
            k: Number of results to return
            embedding: Optional precomputed embedding for target_relation, to avoid recomputing it

        Returns:
            List of tuples (<property_id>, <property_direction>)
        """
        # Initialize property constraints
        direct_props = PropertyConstraints(set(), set())
        inverse_props = PropertyConstraints(set(), set())

        # Collect object type properties
        for obj_type in object_types:
            obj_direct, obj_inverse = self._get_valid_property_ids_by_entity_type(
                obj_type, is_object=True
            )
            direct_props.object_properties.update(obj_direct)
            inverse_props.subject_properties.update(obj_inverse)

        # Collect subject type properties
        for subj_type in subject_types:
            subj_direct, subj_inverse = self._get_valid_property_ids_by_entity_type(
                subj_type, is_object=False
            )
            direct_props.subject_properties.update(subj_direct)
            inverse_props.object_properties.update(subj_inverse)

        # Find valid properties that satisfy both subject and object constraints
        valid_direct = direct_props.subject_properties & direct_props.object_properties
        valid_inverse = (
            inverse_props.subject_properties & inverse_props.object_properties
        )

        prop_id_2_direction = {prop_id: ["direct"] for prop_id in valid_direct}
        for prop_id in valid_inverse:
            if prop_id in prop_id_2_direction:
                prop_id_2_direction[prop_id].append("inverse")
            else:
                prop_id_2_direction[prop_id] = ["inverse"]

        return self._get_ranked_properties(prop_id_2_direction, target_relation, k, embedding=embedding)

    def retrieve_properties_labels_and_constraints(
        self, property_id_list: List[str]
    ) -> Dict[str, Dict[str, str]]:
        collection = self.ontology_db.get_collection(self.property_collection_name)
        result = collection.find(
            query={"property_id": {"$in": property_id_list}},
            projection={
                "_id": 0,
                "property_id": 1,
                "label": 1,
                "valid_subject_type_ids": 1,
                "valid_object_type_ids": 1,
            },
        )

        result_dict = {
            item["property_id"]: {
                "label": item["label"],
                "valid_subject_type_ids": item["valid_subject_type_ids"],
                "valid_object_type_ids": item["valid_object_type_ids"],
            }
            for item in result
        }

        return result_dict

    def retrieve_entity_type_labels(self, entity_type_ids: List[str]):
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        result = collection.find(
            {"entity_type_id": {"$in": entity_type_ids}},
            {"_id": 0, "entity_type_id": 1, "label": 1},
        )

        result_dict = {item["entity_type_id"]: item["label"] for item in result}

        return result_dict

    def retrieve_entity_type_hierarchy(self, entity_type: str) -> List[str]:
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        entity_id_parent_types = collection.find_one(
            {"label": entity_type},
            {"entity_type_id": 1, "parent_type_ids": 1, "label": 1, "_id": 0},
        )
        parent_type_id_labels = collection.find(
            {"entity_type_id": {"$in": entity_id_parent_types["parent_type_ids"]}},
            {"_id": 0, "label": 1, "entity_type_id": 1},
        )
        if entity_id_parent_types:
            extended_types = [entity_id_parent_types["entity_type_id"]] + [
                item["entity_type_id"] for item in parent_type_id_labels
            ]

        return extended_types

    def retrieve_entity_by_type(self, entity_name, entity_type, sample_id, k=10, embedding=None):

        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        entity_id_parent_types = collection.find_one(
            {"label": entity_type},
            {"entity_type_id": 1, "parent_type_ids": 1, "label": 1, "_id": 0},
        )
        extended_types = [
            entity_id_parent_types["entity_type_id"]
        ] + entity_id_parent_types["parent_type_ids"]
        extended_types = [
            elem["label"]
            for elem in collection.find(
                {"entity_type_id": {"$in": extended_types}},
                {"_id": 0, "label": 1, "entity_type_id": 1},
            )
        ]

        collection = self.triplets_db.get_collection(
            self.entity_aliases_collection_name
        )

        query_embedding = embedding if embedding is not None else self.get_embedding(entity_name)
        if query_embedding is None:
            return {}

        if not sample_id:
            filter_query = {
                "entity_type": {"$in": extended_types},
            }
        else:
            filter_query = {
                "entity_type": {"$in": extended_types},
                "sample_id": {"$eq": sample_id},
            }
        result = self.triplets_db.vector_search(
            VectorQuery(
                collection_name=self.entity_aliases_collection_name,
                index_name=self.entities_vector_index_name,
                query_vector=query_embedding,
                vector_field="alias_text_embedding",
                limit=k,
                filters=filter_query,
                projection={"_id": 0, "label": 1, "alias": 1},
            )
        )
        result_dict = {item["alias"]: item["label"] for item in result}

        return result_dict

    def add_entity(self, entity_name, alias, entity_type, sample_id, embedding=None):
        if not sample_id:
            sample_id = "all"
        if embedding is None:
            embedding = self.get_embedding(alias)

        self.triplets_db.upsert_many(
            collection_name=self.entity_aliases_collection_name,
            documents=[
                {
                    "label": entity_name,
                    "entity_type": entity_type,
                    "alias": alias,
                    "sample_id": sample_id,
                    "alias_text_embedding": embedding,
                }
            ],
            unique_fields=["label", "entity_type", "alias", "sample_id"],
        )

    def add_triplets(self, triplets_list, sample_id):
        if not sample_id:
            sample_id = "all"
        for triple in triplets_list:
            triple["sample_id"] = sample_id
        self.triplets_db.upsert_many(
            collection_name=self.triplets_collection_name,
            documents=triplets_list,
            unique_fields=[
                "subject",
                "relation",
                "object",
                "subject_type",
                "object_type",
                "sample_id",
            ],
        )

    def add_filtered_triplets(self, triplets_list, sample_id):
        if not sample_id:
            sample_id = "all"
        for triple in triplets_list:
            triple["sample_id"] = sample_id
        self.triplets_db.upsert_many(
            collection_name=self.filtered_triplets_collection_name,
            documents=triplets_list,
            unique_fields=[
                "subject",
                "relation",
                "object",
                "subject_type",
                "object_type",
                "sample_id",
            ],
        )

    def add_ontology_filtered_triplets(self, triplets_list, sample_id):
        if not sample_id:
            sample_id = "all"
        for triple in triplets_list:
            triple["sample_id"] = sample_id
        self.triplets_db.upsert_many(
            collection_name=self.ontology_filtered_triplets_collection_name,
            documents=triplets_list,
            unique_fields=[
                "subject",
                "relation",
                "object",
                "subject_type",
                "object_type",
                "sample_id",
            ],
        )

    def add_initial_triplets(self, triplets_list, sample_id):
        if not sample_id:
            sample_id = "all"
        for triple in triplets_list:
            triple["sample_id"] = sample_id
        self.triplets_db.upsert_many(
            collection_name=self.initial_triplets_collection_name,
            documents=triplets_list,
            unique_fields=[
                "subject",
                "relation",
                "object",
                "subject_type",
                "object_type",
                "sample_id",
            ],
        )

    def retrieve_similar_entity_names(
        self, entity_name: str, k: int = 10, sample_id: str = None, embedding: Optional[List[float]] = None
    ) -> List[Dict[str, str]]:
        embedded_query = embedding if embedding is not None else self.get_embedding(entity_name)
        if embedded_query is None:
            return []

        if sample_id:
            search_filter = {"sample_id": {"$eq": sample_id}}
        else:
            search_filter = {}

        try:
            result_list = self.triplets_db.vector_search(
                VectorQuery(
                    collection_name=self.entity_aliases_collection_name,
                    index_name=self.entities_vector_index_name,
                    query_vector=embedded_query,
                    vector_field="alias_text_embedding",
                    limit=k,
                    filters=search_filter,
                    projection={"_id": 0, "label": 1, "entity_type": 1},
                )
            )
            return [{"entity": item["label"]} for item in result_list]
        except Exception:
            return self._retrieve_entity_names_cosine_fallback(
                embedded_query, k, sample_id
            )

    def _retrieve_entity_names_cosine_fallback(
        self, query_embedding: List[float], k: int, sample_id: str = None
    ) -> List[Dict[str, str]]:
        """In-memory cosine-similarity fallback when $vectorSearch is unavailable."""
        import numpy as np

        mongo_query = {"sample_id": sample_id} if sample_id else {}
        collection = self.triplets_db.get_collection(self.entity_aliases_collection_name)
        docs = collection.find(mongo_query, {"label": 1, "alias_text_embedding": 1})

        q = np.array(query_embedding, dtype=float)
        q_norm = float(np.linalg.norm(q))

        scored: List[tuple] = []
        for doc in docs:
            emb = doc.get("alias_text_embedding")
            if not emb:
                continue
            v = np.array(emb, dtype=float)
            v_norm = float(np.linalg.norm(v))
            score = float(np.dot(q, v) / (q_norm * v_norm)) if q_norm and v_norm else 0.0
            scored.append((score, doc.get("label", "")))

        scored.sort(key=lambda x: x[0], reverse=True)

        seen: set = set()
        result: List[Dict[str, str]] = []
        for _, label in scored:
            if label and label not in seen:
                seen.add(label)
                result.append({"entity": label})
            if len(result) >= k:
                break
        return result
