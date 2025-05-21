from typing import List, Tuple, Set, Dict
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError

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
    def __init__(self, db):
        self.db = db
        self.entity_type_vector_index_name = 'entity_type_aliases'
        self.property_vector_index_name = 'property_aliases_ids'

        self.entity_type_collection_name = 'entity_types'
        self.entity_type_aliases_collection_name = 'entity_type_aliases'
        self.property_collection_name = 'properties'
        self.property_aliases_collection_name = 'property_aliases'

        self.entity_aliases_collection_name = 'entity_aliases'
        self.triplets_collection_name = 'triplets'

        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever').to("cuda")


    def get_embedding(self, text):

        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        if not text or not isinstance(text, str):
            return None

        try:
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**inputs.to('cuda'))
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            return embeddings.detach().cpu().tolist()[0]
        
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            return None
        
    
    def _get_unique_similar_entities(
        self,
        target_entity_type: str,
        k: int = 5,
        max_attempts: int = 10
    ) -> List[str]:
        # retrieve k most similar entity types to the given triplet
        # using the entity type index
        # return the wikidata ids of the most similar entity types
        
        query_k = k * 2
        attempt = 0
        unique_ranked_entities: List[str] = []
        query_embedding = self.get_embedding(target_entity_type)
        collection = self.db.get_collection(self.entity_type_aliases_collection_name)

        # as we search among aliases, there can be duplicated original entitites
        # and as we want K unique entities in result, we querying the index until we get exactly K unique entities
        while len(unique_ranked_entities) < k and attempt < max_attempts:
            search_pipeline = [{
                    "$vectorSearch": {
                    "index": self.entity_type_vector_index_name, #
                    "queryVector": query_embedding, 
                    "path": 'alias_text_embedding', 
                    "numCandidates": 150, 
                    "limit": query_k 
                    }
                }, 
                {
                    "$project": {
                        "_id": 0,
                        "entity_type_id": 1

                    }
                }
            ]
            result = collection.aggregate(search_pipeline)
            for res in result: 
                if res['entity_type_id'] not in unique_ranked_entities:
                    unique_ranked_entities.append(res['entity_type_id'])
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
            target_entity_type=triplet['subject_type'],
            k=k
        )
        if 'object_type' in triplet:
            similar_object_types = self._get_unique_similar_entities(
                target_entity_type=triplet['object_type'],
                k=k
            )
        else: 
            similar_object_types = []
        return similar_subject_types, similar_object_types
    

    def _get_valid_property_ids_by_entity_type(
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

        collection = self.db.get_collection(self.entity_type_collection_name)
        
        # Get extended types including supertypes
        extended_types = [entity_type, 'ANY']
        hirerarchy = collection.find_one({"entity_type_id": entity_type}, {"parent_type_ids": 1, "_id": 0})
        extended_types.extend(hirerarchy['parent_type_ids'])
        
        pipeline = [
            {"$match": {"entity_type_id": {"$in": extended_types}}},
            {
                "$group": {
                    "_id": None,
                    "subject_ids": {"$addToSet": {"$ifNull": ["$valid_subject_property_ids", []]}},
                    "object_ids": {"$addToSet": {"$ifNull": ["$valid_object_property_ids", []]}}
                }
            },
            {
                "$project": {
                    "subject_ids": {"$reduce": {
                        "input": "$subject_ids",
                        "initialValue": [],
                        "in": {"$setUnion": ["$$value", "$$this"]}
                    }},
                    "object_ids": {"$reduce": {
                        "input": "$object_ids",
                        "initialValue": [],
                        "in": {"$setUnion": ["$$value", "$$this"]}
                    }}
                }
            }
        ]
        result = collection.aggregate(pipeline)

        result_data = next(result, {})

        subject_props = result_data.get("subject_ids", [])
        object_props = result_data.get("object_ids", [])

        if is_object:
            direct_props = set(object_props)
            inverse_props = set(subject_props)
        else:
            direct_props = set(subject_props)
            inverse_props = set(object_props)

        return direct_props, inverse_props
    

    def _get_ranked_properties(
        self,
        prop_2_direction: Dict[str, List[str]], # mapping of property_ids to their direction that can be used in the specified context
        target_property: str,
        k: int
    ) -> List[Tuple[str, str]]: # List of tuples (<property_id>, <property_direction>)
        """
        Rank properties based on similarity to target relation.
        """
        collection = self.db.get_collection(self.property_aliases_collection_name)
        query_embedding = self.get_embedding(target_property)
        props = list(prop_2_direction.keys())

        query_k = k * 2
        max_attempts = 5  # Prevent infinite loops
        attempt = 0
        unique_ranked_properties: List[str] = []
        
        while len(unique_ranked_properties) < k and attempt < max_attempts:

            pipeline = [{
                "$vectorSearch": {
                    "index": "property_aliases_ids", 
                    "queryVector": query_embedding,  
                    "path": "alias_text_embedding", 
                    "numCandidates": 150,  
                    "limit": query_k,  
                    "filter": {"relation_id": {"$in": props}},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "relation_id": 1,
                    # "score": {"$meta": "vectorSearchScore"} 
                }
            }
            ]

            similar_properties = collection.aggregate(pipeline)

            for prop in similar_properties: 
                if prop['relation_id'] not in unique_ranked_properties:
                    unique_ranked_properties.append(prop['relation_id'])
                if len(unique_ranked_properties) == k:
                    break
            
            query_k *= 2
            attempt += 1
        
        # taking into account directions of properties 
        unique_ranked_properties_with_direction = []
        for prop_id in unique_ranked_properties:
            for direction in prop_2_direction[prop_id]:
                unique_ranked_properties_with_direction.append((prop_id, direction))
            
            if len(unique_ranked_properties_with_direction) >= k:
                break

        return unique_ranked_properties_with_direction


    def retrieve_properties_for_entity_type(
        self,
        target_relation: str,
        object_types: List[str],
        subject_types: List[str],
        k: int = 10
    ) -> List[Tuple[str, str]]: # List of tuples (<property_id>, <property_direction>)
        """
        Retrieve and rank properties that match given entity types and relation.
        
        Args:
            target_relation: The relation to search for
            object_types: List of valid object types
            subject_types: List of valid subject types
            k: Number of results to return
            
        Returns:
            List of direct property_ids
            List of inverse property_ids
        """
        # Initialize property constraints
        direct_props = PropertyConstraints(set(), set())
        inverse_props = PropertyConstraints(set(), set())

        # Collect object type properties
        for obj_type in object_types:
            obj_direct, obj_inverse = self._get_valid_property_ids_by_entity_type(obj_type, is_object=True)
            direct_props.object_properties.update(obj_direct)
            inverse_props.subject_properties.update(obj_inverse)
            
        # Collect subject type properties
        for subj_type in subject_types:
            subj_direct, subj_inverse = self._get_valid_property_ids_by_entity_type(subj_type, is_object=False)
            direct_props.subject_properties.update(subj_direct)
            inverse_props.object_properties.update(subj_inverse)

        # Find valid properties that satisfy both subject and object constraints
        valid_direct = direct_props.subject_properties & direct_props.object_properties
        valid_inverse = inverse_props.subject_properties & inverse_props.object_properties

        prop_id_2_direction = {prop_id: ["direct"] for prop_id in valid_direct}
        for prop_id in valid_inverse:
            if prop_id in prop_id_2_direction:
                prop_id_2_direction[prop_id].append("inverse")
            else: 
                prop_id_2_direction[prop_id] = ["inverse"]
        
        return self._get_ranked_properties(prop_id_2_direction, target_relation, k)


    def retrieve_properties_labels_and_constraints(self, property_id_list: List[str]) -> Dict[str, Dict[str, str]]:
        collection = self.db.get_collection(self.property_collection_name)

        
        pipeline = [
            {"$match": {"property_id": {"$in": property_id_list}}},
            {
                "$project": {
                    "_id": 0,
                    "property_id": 1,
                    "label": 1,
                    "valid_subject_type_ids": 1,
                    "valid_object_type_ids": 1
                }
            }
        ]
        result = collection.aggregate(pipeline)

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
        collection = self.db.get_collection(self.entity_type_collection_name)
        pipeline = [
            {"$match": {"entity_type_id": {"$in": entity_type_ids}}},
            {
                "$project": {
                    "_id": 0,
                    "entity_type_id": 1,
                    "label": 1,

                }
            }
        ]
        result = collection.aggregate(pipeline)

        result_dict = {
            item["entity_type_id"]: {"label": item["label"]}
            for item in result
        }

        return result_dict


    def retrieve_entity_type_hirerarchy(self, entity_type: str) -> List[str]:
        collection = self.db.get_collection(self.entity_type_collection_name)
        entity_id_parent_types = collection.find_one({"label": entity_type}, {"entity_type_id": 1, "parent_type_ids": 1, "_id": 0})
        if entity_id_parent_types:        
            extended_types = [entity_id_parent_types['entity_type_id']] + entity_id_parent_types['parent_type_ids']
        return extended_types


    def retrieve_entity_by_type(self, entity_name, entity_type, sample_id):
        # class EntityAliases(BaseModel):
        #     _id: int
        #     sample_id: str
        #     alias: str
        #     entity_type: str
        #     label_embedding: List[float]
        collection = self.db.get_collection(self.entity_type_collection_name)
        entity_id_parent_types = collection.find_one({"label": entity_type}, {"entity_type_id": 1, "parent_type_ids": 1, "_id": 0})        
        extended_types = [entity_id_parent_types['entity_type_id']] + entity_id_parent_types['parent_type_ids']
    
        collection = self.db.get_collection(self.entity_aliases_collection_name)

        query_embedding = self.get_embedding(entity_name)

        pipeline = [{
            "$vectorSearch": {
                    "index": "entity_aliases_entity_types", 
                    "queryVector": query_embedding,  
                    "path": "alias_text_embedding", 
                    "numCandidates": 150,  
                    "limit": 10,  
                    "filter": {"entity_type": {"$in": extended_types},
                               "sample_id": {"$eq": sample_id}
                               },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "label": 1,
                    "alias": 1
                }
            }
            ]

        result = collection.aggregate(pipeline)
        result_dict = {item['alias']: item['label'] for item in result}

        return result_dict

    def add_entity(self, entity_name, alias, entity_type, sample_id):
        collection = self.db.get_collection(self.entity_type_collection_name)
        entity_type_id = collection.find_one({"label": entity_type}, {"_id": 0, "entity_type_id": 1})['entity_type_id']

        collection = self.db.get_collection(self.entity_aliases_collection_name)

        collection.insert_one({
                "label": entity_name, 
                "entity_type": entity_type_id,
                "alias": alias, 
                "sample_id": sample_id,
                "alias_text_embedding": self.get_embedding(alias)
            })

    def add_triplets(self, triplets_list, sample_id):
        collection = self.db.get_collection(self.triplets_collection_name)
        for triple in triplets_list:
            triple['sample_id'] = sample_id
        collection.insert_many(triplets_list)    
