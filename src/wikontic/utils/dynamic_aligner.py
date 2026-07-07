from typing import List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import torch
from wikontic.db.factory import ensure_storage_backend
from wikontic.db.interfaces import VectorQuery
from wikontic.utils.embedding_model import (
    SUPPORTED_MODELS,
    load_embedding_model,
    get_embedding as _get_embedding,
    get_embeddings as _get_embeddings,
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
_ = load_dotenv(find_dotenv())


class EntityAlias(BaseModel):
    _id: int
    label: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]


class PropertyAlias(BaseModel):
    _id: int
    label: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]


class Aligner:
    def __init__(self, triplets_db, device=None, embedding_model: str = "contriever"):
        if embedding_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown embedding model {embedding_model!r}. "
                f"Supported values: {sorted(SUPPORTED_MODELS)}"
            )
        self.db = ensure_storage_backend(triplets_db)

        self.entity_aliases_collection_name = "entity_aliases"
        self.property_aliases_collection_name = "property_aliases"

        self.property_vector_index_name = "property_aliases"
        self.entities_vector_index_name = "entity_aliases"

        self.initial_triplets_collection_name = "initial_triplets"
        self.triplets_collection_name = "triplets"
        self.filtered_triplets_collection_name = "filtered_triplets"

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

    def retrieve_similar_properties(
        self, target_relation: str, sample_id: str, k: int = 10, embedding: Optional[List[float]] = None
    ) -> List[str]:  # List of property labels
        """
        Retrieve and rank properties that match given relation.

        Args:
            target_relation: The relation to search for
            k: Number of results to return
            embedding: Optional precomputed embedding for target_relation, to avoid recomputing it

        Returns:
            List of property labels
        """

        query_embedding = embedding if embedding is not None else self.get_embedding(target_relation)
        if query_embedding is None:
            return []

        query_k = k * 2
        max_attempts = 5  #
        attempt = 0
        unique_ranked_properties: List[str] = []

        while len(unique_ranked_properties) < k and attempt < max_attempts:

            similar_properties = self.db.vector_search(
                VectorQuery(
                    collection_name=self.property_aliases_collection_name,
                    index_name=self.property_vector_index_name,
                    query_vector=query_embedding,
                    vector_field="alias_text_embedding",
                    limit=query_k if query_k < 150 else 150,
                    projection={"_id": 0, "label": 1},
                )
            )

            for prop in similar_properties:
                if prop.get("label") not in unique_ranked_properties:
                    unique_ranked_properties.append(prop["label"])
                if len(unique_ranked_properties) == k:
                    break

            query_k *= 2
            attempt += 1

        return unique_ranked_properties

    def retrieve_similar_entity_names(
        self, entity_name: str, sample_id: Optional[str] = None, k: int = 10, embedding: Optional[List[float]] = None
    ) -> List[str]:  # List of entity labels
        """
        Retrieve and rank entities that match given entity.

        Args:
            entity_name: The entity to search for
            k: Number of results to return
            embedding: Optional precomputed embedding for entity_name, to avoid recomputing it

        Returns:
            List of entity labels
        """

        query_embedding = embedding if embedding is not None else self.get_embedding(entity_name)
        if query_embedding is None:
            return []

        query_k = k * 2
        max_attempts = 5  #
        attempt = 0
        unique_ranked_entities: List[str] = []

        while len(unique_ranked_entities) < k and attempt < max_attempts:

            if sample_id is not None:
                query_filter = {"sample_id": {"$eq": sample_id}}
            else:
                query_filter = {}
            similar_entities = self.db.vector_search(
                VectorQuery(
                    collection_name=self.entity_aliases_collection_name,
                    index_name=self.entities_vector_index_name,
                    query_vector=query_embedding,
                    vector_field="alias_text_embedding",
                    limit=query_k if query_k < 150 else 150,
                    filters=query_filter,
                    projection={"_id": 0, "label": 1},
                )
            )

            for entity in similar_entities:
                if entity["label"] not in unique_ranked_entities:
                    unique_ranked_entities.append(entity["label"])
                if len(unique_ranked_entities) == k:
                    break

            query_k *= 2
            attempt += 1

        return unique_ranked_entities

    def add_entity(self, entity_name, alias, sample_id, embedding=None):
        if embedding is None:
            embedding = self.get_embedding(alias)
        self.db.upsert_many(
            collection_name=self.entity_aliases_collection_name,
            documents=[
                {
                    "label": entity_name,
                    "alias": alias,
                    "sample_id": sample_id,
                    "alias_text_embedding": embedding,
                }
            ],
            unique_fields=["label", "alias", "sample_id"],
        )

    def add_property(self, property_name, alias, sample_id, embedding=None):
        if embedding is None:
            embedding = self.get_embedding(alias)
        self.db.upsert_many(
            collection_name=self.property_aliases_collection_name,
            documents=[
                {
                    "label": property_name,
                    "alias": alias,
                    # "sample_id": sample_id,
                    "alias_text_embedding": embedding,
                }
            ],
            unique_fields=["label", "alias"],
        )

    def add_triplets(self, triplets_list, sample_id):
        for triple in triplets_list:
            triple["sample_id"] = sample_id
        self.db.upsert_many(
            collection_name=self.triplets_collection_name,
            documents=triplets_list,
            unique_fields=["subject", "relation", "object", "sample_id"],
        )

    def add_filtered_triplets(self, triplets_list, sample_id):
        for triple in triplets_list:
            triple["sample_id"] = sample_id
        self.db.upsert_many(
            collection_name=self.filtered_triplets_collection_name,
            documents=triplets_list,
            unique_fields=["subject", "relation", "object", "sample_id"],
        )

    def add_initial_triplets(self, triplets_list, sample_id):
        for triple in triplets_list:
            triple["sample_id"] = sample_id
        self.db.upsert_many(
            collection_name=self.initial_triplets_collection_name,
            documents=triplets_list,
            unique_fields=["subject", "relation", "object", "sample_id"],
        )
