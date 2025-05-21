import json
import faiss
from functools import lru_cache
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

@dataclass
class PropertyConstraints:
    subject_properties: Set[str]
    object_properties: Set[str]


class Aligner:
    def __init__(self, mappings_path='utils/ontology_mappings', device='cuda:2'):
        self.device = device
        self.dim = 768
        self.metric = faiss.METRIC_INNER_PRODUCT
        self.mappings_path = mappings_path
        self.type2entity = {}
        # Initialize models
        self._init_models()
        # Load all mappings
        self._load_mappings()
        # Load indices
        self._load_indices()
        self.entity_name_index = faiss.index_factory(self.dim, "IDMap,Flat", self.metric)
        self.entity_name2id = {}
        self.id2entity_name = {}
        

    def _init_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)

        
    def _load_mappings(self):
        # Load all JSON mapping files
        mapping_files = {     
            'obj2prop': 'obj_constraint2prop.json',
            'subj2prop': 'subj_constraint2prop.json',
            'prop2constraint': 'prop2constraints.json',
            'prop2label': 'prop2label.json',
            'enum_prop_ids': 'enum_prop_ids.json',
            'propid2enum': 'propid2enum.json',
            'entity_type2label': 'entity_type2label.json',
            'label2entity_type': 'label2entity.json',
            'entity2hierarchy': 'entity_hierarchy.json',
            'enum_entity_id': 'enum_entity_ids.json',
            "prop2constraints": "prop2constraints.json"

        }
        for attr_name, filename in mapping_files.items():
            with open(os.path.join(self.mappings_path, filename), 'r') as f:
                setattr(self, attr_name, json.load(f))
    

    def _load_indices(self):
        """Load FAISS indices"""
        self.entity_type_index = faiss.read_index(f"{self.mappings_path}/wikidata_ontology_entities.index")
        self.property_index = faiss.read_index(f"{self.mappings_path}/wikidata_relations.index")


    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    @lru_cache(maxsize=1000)
    def embed_batch(self, names: Tuple[str, ...]) -> np.ndarray:
        inputs = self.tokenizer(list(names), padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs.to(self.device))
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        return np.array(embeddings.detach().cpu().numpy())


    def calculate_string_similarities(self, candidates_labels: List[str], target: str):
        candidate_embeddings = self.embed_batch(tuple(candidates_labels))
        target_embedding = self.embed_batch((target,))
        similarities = cosine_similarity(candidate_embeddings, target_embedding).flatten()
        return similarities
    
    # ranking names in a subset, e.g. for properties we want to search only among 
    # ones that meet the condition of subject and object types constraints

    def search_within_subset(
            self,
            query: str, 
            subset_ids: List[int],
            faiss_id_index: faiss.IndexFlatL2,
            k: int = 10
        ) -> Tuple[np.ndarray, np.ndarray]:

        query = self.embed_batch((query,))
        subset_vectors = np.array([faiss_id_index.index.reconstruct(int(i)) for i in subset_ids])
    
        subset_index = faiss.IndexFlatL2(self.dim)
        subset_index.add(subset_vectors)
    
        distances, subset_indices = subset_index.search(query, k)
        retrieved_ids = np.array(subset_ids)[subset_indices]

        return distances[0], retrieved_ids
    

    def _get_unique_similar_entities(
        self,
        entity_type: str,
        k: int,
        max_attempts: int = 5
    ) -> List[str]:
        # retrieve k most similar entity types to the given triplet
        # using the entity type index
        # return the wikidata ids of the most similar entity types
        query_k = k
        attempt = 0
        unique_ranked_entities: List[str] = []

        # as we search among aliases, there can be duplicated original entitites
        # and as we want K unique entities in result, we querying the index until we get exactly K unique entities
        while len(unique_ranked_entities) < k and attempt < max_attempts:
            # Embed and search
            embedded_query = self.embed_batch((entity_type,))
            _, indices = self.entity_type_index.search(embedded_query, query_k)
            
            # Convert to entity IDs while preserving order and ensuring uniqueness
            for idx in indices[0]:
                entity_id = self.enum_entity_id[idx]
                if entity_id not in unique_ranked_entities:
                    unique_ranked_entities.append(entity_id)
                if len(unique_ranked_entities) == k:
                    break
            query_k *= 2
            attempt += 1

        return unique_ranked_entities


    def retrieve_similar_entity_types(
        self,
        triplet: Dict[str, str],
        k: int = 10
    ) -> Tuple[List[str], List[str]]:

        # Get similar types for both subject and object
        similar_subject_types = self._get_unique_similar_entities(
            entity_type=triplet['subject_type'],
            k=k
        )
        if 'object_type' in triplet:
            similar_object_types = self._get_unique_similar_entities(
                entity_type=triplet['object_type'],
                k=k
            )
        else: 
            similar_object_types = []
        return similar_subject_types, similar_object_types


    def _get_properties_by_type(
        self,
        entity_type: str,
        is_object: bool = True
    ) -> Tuple[Set[str], Set[str]]:
        """
        Get direct and inverse properties for an entity type.
        
        Args:
            entity_type: The entity type to look up
            is_object: Whether this is an object type in triplet (True) or a subject type (False)
        """
        # Get extended types including supertypes
        extended_types = [entity_type, '<ANY SUBJECT>', '<ANY OBJECT>']
        if entity_type in self.entity2hierarchy:
            extended_types.extend(self.entity2hierarchy[entity_type])

        direct_props, inverse_props = set(), set()
        for entity in extended_types:
            if is_object:
                if entity in self.obj2prop:
                    direct_props.update(self.obj2prop[entity])
                if entity in self.subj2prop:
                    inverse_props.update(self.subj2prop[entity])
            else:
                if entity in self.subj2prop:
                    direct_props.update(self.subj2prop[entity])
                if entity in self.obj2prop:
                    inverse_props.update(self.obj2prop[entity])

        return direct_props, inverse_props


    def _get_ranked_properties(
        self,
        props: List[Tuple[str, str, str]], # tuple (prop_id, label, direction)
        target_relation: str,
        k: int
    ) -> List[Tuple[str, str, str]]:
        """
        Rank properties based on similarity to target relation.
        """
        # Get property IDs for FAISS lookup among aliases
        subset_prop_ids = []
        for prop in props:
            prop_id = prop[0]  # Get property ID from the tuple (prop_id, label, direction)
            # propid2enum maps a property ID to a list of its aliases' indices in the FAISS index
            prop_indices = self.propid2enum[prop_id]
            subset_prop_ids.extend(prop_indices)
        
        # Convert to list of unique indices
        subset_prop_ids = list(set(subset_prop_ids))

        query_k = k
        max_attempts = 5  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            alias_distances, ids = self.search_within_subset(
                query=target_relation,
                subset_ids=list(subset_prop_ids),
                faiss_id_index=self.property_index,
                k=query_k
            )
            ids = [self.enum_prop_ids[i] for i in ids[0]]
            # Map IDs to original properties
            prop_matches = []
            for prop in props:
                if prop[0] in ids:
                    idx = ids.index(prop[0])
                    prop_matches.append((prop, alias_distances[idx]))
            
            # Sort by distance
            ranked_props = [p[0] for p in sorted(prop_matches, key=lambda x: x[1])]
            
            # if len(ranked_props) >= k:
            unique_ranked_props = set([p[1] for p in ranked_props]) # getting unique names
            if len(unique_ranked_props) >= k:
                return ranked_props[:len(set(unique_ranked_props))]

            # print(sorted(prop_matches, key=lambda x: x[1]))
                
            query_k *= 2
            attempts += 1
        
            
        unique_ranked_props = set([p[1] for p in ranked_props]) # getting unique names
        return ranked_props[:len(set(unique_ranked_props))] if ranked_props else []


    def retrieve_properties_for_entity_type(
        self,
        target_relation: str,
        object_types: List[str],
        subject_types: List[str],
        k: int = 10
    ) -> List[Tuple[str, str, str]]:
        """
        Retrieve and rank properties that match given entity types and relation.
        
        Args:
            target_relation: The relation to search for
            object_types: List of valid object types
            subject_types: List of valid subject types
            k: Number of results to return
            
        Returns:
            List of tuples (property_id, property_label, direction)
        """
        # Initialize property constraints
        direct_props = PropertyConstraints(set(), set())
        inverse_props = PropertyConstraints(set(), set())

        # Collect object type properties
        for obj_type in object_types:
            obj_direct, obj_inverse = self._get_properties_by_type(obj_type, is_object=True)
            direct_props.object_properties.update(obj_direct)
            inverse_props.subject_properties.update(obj_inverse)
            
        # Collect subject type properties
        for subj_type in subject_types:
            subj_direct, subj_inverse = self._get_properties_by_type(subj_type, is_object=False)
            direct_props.subject_properties.update(subj_direct)
            inverse_props.object_properties.update(subj_inverse)

        # Find valid properties that satisfy both subject and object constraints
        valid_direct = direct_props.subject_properties & direct_props.object_properties
        valid_inverse = inverse_props.subject_properties & inverse_props.object_properties

        
        props = [
            (prop, self.prop2label[prop], "direct") for prop in valid_direct
        ] + [
            (prop, self.prop2label[prop], "inverse") for prop in valid_inverse
        ]
        
        return self._get_ranked_properties(props, target_relation, k)

    
    def retrieve_entity_by_type(self, entity: str, entity_type: str, k: int = 5):
        extended_entity_types = [entity_type]
        if entity_type in self.entity2hierarchy:
            extended_entity_types.extend(self.entity2hierarchy[entity_type])

        entities_for_ranking = []

        for e_type in extended_entity_types:
            if e_type in self.type2entity:
                entities_for_ranking.extend(self.type2entity[e_type])

        if len(entities_for_ranking) > 0:        
            similarities = self.calculate_string_similarities(candidates_labels=entities_for_ranking, target=entity)
            ranked_entities = sorted(zip(entities_for_ranking, similarities), key=lambda x: x[1], reverse=True)
            ranked_entities = [ranked_entity[0] for ranked_entity in ranked_entities[:k]]
            return ranked_entities
        else:
            return []


    def add_entity(self, entity: str, entity_type: str):
        if entity_type not in self.type2entity:
            self.type2entity[entity_type] = []
        if entity not in self.type2entity[entity_type]:
            self.type2entity[entity_type].append(entity)
            embeddings = self.embed_batch((entity,))
            id_ = len(self.id2entity_name)
            self.entity_name_index.add_with_ids(embeddings, [id_])
            self.id2entity_name[id_] = (entity, entity_type)
            self.entity_name2id[entity] = id_


    def retrive_similar_entity_names(self, entity_name: str, k: int = 10) -> List[Dict[str, str]]:

        embedded_query = self.embed_batch((entity_name,))
        _, indices = self.entity_name_index.search(embedded_query, k)
        similar_entities = [self.id2entity_name[ind] for ind in indices[0]]
        similar_entities_with_types = [{'entity': e[0], 'entity_type': e[1]} for e in similar_entities]

        return similar_entities_with_types