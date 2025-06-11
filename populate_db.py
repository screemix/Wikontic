from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import pymongo

from typing import List
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import time
import argparse
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model = AutoModel.from_pretrained('facebook/contriever').to("cuda")

class EntityType(BaseModel):
    _id: int
    entity_type_id: str
    label: str
    parent_type_ids: List[str]
    valid_subject_property_ids: List[str]
    valid_object_property_ids: List[str]

class Property(BaseModel):
    _id: int
    property_id: str
    label: str 
    valid_subject_type_ids: List[str]
    valid_object_type_ids: List[str]

class EntityTypeAlias(BaseModel):
    _id: int
    entity_type_id: str
    alias_label: str    
    alias_text_embedding: List[float]

class PropertyAlias(BaseModel):
    _id: int
    relation_id: str
    alias_label: str
    alias_text_embedding: List[float]


def get_embedding(text):
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    if not text or not isinstance(text, str):
        return None

    try:
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs.to('cuda'))
        embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        return embeddings.detach().cpu().tolist()[0]
    
    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
        return None


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    logger.info("Connection to MongoDB successful")
    return client


def populate_entity_types(ENTITY_2_LABEL, ENTITY_2_HIERARCHY, SUBJ_2_PROP_CONSTRAINTS, OBJ_2_PROP_CONSTRAINTS, db, collection_name='entity_types'):
    logger.info(f"Starting to populate {collection_name} collection")
    entity_metadata_list = []

    for i, entity in enumerate(ENTITY_2_LABEL.keys()):
        label = ENTITY_2_LABEL[entity]
        parents = ENTITY_2_HIERARCHY[entity]

        valid_subject_property_ids = SUBJ_2_PROP_CONSTRAINTS[entity] if entity in SUBJ_2_PROP_CONSTRAINTS else []
        valid_object_property_ids = OBJ_2_PROP_CONSTRAINTS[entity] if entity in OBJ_2_PROP_CONSTRAINTS else []
        
        entity_metadata_list.append({"_id": i, "entity_type_id": entity, 
                                    'label': label, "parent_type_ids": parents,
                                    "valid_subject_property_ids": valid_subject_property_ids,
                                    "valid_object_property_ids": valid_object_property_ids})


    entity_metadata_list.append({"_id": i+1, 
                                "entity_type_id": "ANY", 
                                "label": "ANY", 
                                "parent_type_ids": [],
                                "valid_subject_property_ids": SUBJ_2_PROP_CONSTRAINTS["<ANY SUBJECT>"],
                                "valid_object_property_ids": OBJ_2_PROP_CONSTRAINTS["<ANY OBJECT>"]})
        
    try:
        records = [EntityType(**record).model_dump() for record in entity_metadata_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_entity_type_aliases(ENTITY_2_LABEL, ENTITY_2_ALIASES, db, collection_name="entity_type_aliases"):
    logger.info(f"Starting to populate {collection_name} collection")
    entity_types_list = []
    id_count = 0

    for e, aliases in tqdm(ENTITY_2_ALIASES.items()):
        alias_embedding = get_embedding(ENTITY_2_LABEL[e])
        entity_types_list.append({"_id": id_count, "entity_type_id": e, "alias_label": ENTITY_2_LABEL[e], "alias_text_embedding": alias_embedding})
        id_count += 1

        for alias in aliases:
            alias_embedding = get_embedding(alias)
            entity_types_list.append({"_id": id_count, "entity_type_id": e, "alias_label": alias, "alias_text_embedding": alias_embedding})
            id_count += 1
    try:
        records = [EntityTypeAlias(**record).model_dump() for record in entity_types_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_properties(PROP_2_LABEL, PROP_2_CONSTRAINT, db, collection_name="properties"):
    logger.info(f"Starting to populate {collection_name} collection")
    property_list = []

    for i, prop_id in enumerate(PROP_2_LABEL.keys()):
        property_list.append({"_id": i, "property_id": prop_id, "label": PROP_2_LABEL[prop_id], 
                              "valid_subject_type_ids": PROP_2_CONSTRAINT[prop_id]['Subject type constraint'],
                              "valid_object_type_ids": PROP_2_CONSTRAINT[prop_id]['Value-type constraint']})

    try:
        records = [Property(**record).model_dump() for record in property_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_property_aliases(PROP_2_LABEL, PROP_2_ALIASES, db, collection_name="property_aliases"):
    logger.info(f"Starting to populate {collection_name} collection")
    relation_alias_id_pairs = []
    id_count = 0

    for r, aliases in tqdm(PROP_2_ALIASES.items()):
        alias_embedding = get_embedding(PROP_2_LABEL[r])
        relation_alias_id_pairs.append({"_id": id_count, "relation_id": r, "alias_label": PROP_2_LABEL[r], "alias_text_embedding": alias_embedding})
        id_count += 1

        for alias in aliases:
            alias_embedding = get_embedding(alias)
            relation_alias_id_pairs.append({"_id": id_count, "relation_id": r, "alias_label": alias, "alias_text_embedding": alias_embedding})
            id_count += 1
    try:
        records = [PropertyAlias(**record).model_dump() for record in relation_alias_id_pairs]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")
    
    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def create_index_for_entity_types(db, collection_name='entity_type_aliases', embedding_field_name="alias_text_embedding", index_name='entity_type_aliases'):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": { 
                "dynamic": True, 
                "fields": { 
                    embedding_field_name: { 
                        "dimensions": 768, 
                        "similarity": "cosine", 
                        "type": "knnVector",
                    }
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20) 
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")
        

def create_index_for_entities(db, collection_name='entity_aliases', embedding_field_name="alias_text_embedding", entity_type_id_field_name='entity_type', index_name='entities'):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": 768,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },
                    entity_type_id_field_name: {
                        "type": "token"
                    },
                    "sample_id": {
                        # "type": "number"
                        "type": "token"

                    }
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)  
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


def create_index_for_properties(db, collection_name='property_aliases', embedding_field_name='alias_text_embedding', prop_id_field_name='relation_id', index_name='property_aliases_ids'):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": 768,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },
                    prop_id_field_name: {
                        "type": "token"
                    },
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)  
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Populate MongoDB with Wikidata ontology data')
    
    parser.add_argument('--mappings_dir', type=str, default="utils/ontology_mappings/",
                        help='Directory containing ontology mapping files')
    parser.add_argument('--mongo_uri', type=str, default="mongodb://localhost:27018/?directConnection=true",
                        help='MongoDB connection URI')
    parser.add_argument('--database', type=str, default="wikidata_ontology",
                        help='MongoDB database name')
    
    # Collection names
    parser.add_argument('--entity_types_collection', type=str, default="entity_types",
                        help='Collection name for entity types')
    parser.add_argument('--entity_type_aliases_collection', type=str, default="entity_type_aliases",
                        help='Collection name for entity type aliases')
    parser.add_argument('--properties_collection', type=str, default="properties",
                        help='Collection name for properties')
    parser.add_argument('--property_aliases_collection', type=str, default="property_aliases",
                        help='Collection name for property aliases')
    parser.add_argument('--entity_collection', type=str, default="entity_aliases",
                        help='Collection name for entities')
    
    # Index names
    parser.add_argument('--entity_types_index', type=str, default="entity_type_aliases",
                        help='Index name for entity types')
    parser.add_argument('--property_aliases_index', type=str, default="property_aliases_ids",
                        help='Index name for property aliases')
    parser.add_argument('--entity_index', type=str, default="entities",
                        help='Index name for entities')
    
    
    args = parser.parse_args()

    logger.info("Starting database population process")
    logger.info(f"Using database: {args.database}")
    logger.info(f"Loading mapping files from: {args.mappings_dir}")

    # Load mapping files
    with open(args.mappings_dir+"subj_constraint2prop.json", 'r') as f:
        subj2prop_constraints = json.load(f)

    with open(args.mappings_dir+"obj_constraint2prop.json", 'r') as f:
        obj2prop_constraints = json.load(f)

    with open(args.mappings_dir+'entity_type2label.json', 'r') as f:
        ENTITY_2_LABEL = json.load(f)

    with open(args.mappings_dir+'entity_type2hierarchy.json', 'r') as f:
        ENTITY_2_HIERARCHY = json.load(f)

    with open(args.mappings_dir+'entity_type2aliases.json', 'r') as f:
        ENTITY_2_ALIASES = json.load(f)

    with open(args.mappings_dir+'prop2constraints.json', 'r') as f:
        PROP_2_CONSTRAINT = json.load(f)

    with open(args.mappings_dir+'prop2label.json', 'r') as f:
        PROP_2_LABEL = json.load(f)

    with open(args.mappings_dir+"prop2aliases.json", 'r') as f:
        PROP_2_ALIASES = json.load(f)

    logger.info("Successfully loaded all mapping files")

    # Connect to MongoDB
    mongo_client = get_mongo_client(args.mongo_uri)
    db = mongo_client.get_database(args.database)

    # Drop all existing collections
    logger.info("Dropping existing collections...")
    for collection_name in db.list_collection_names():
        logger.info(f"Dropping collection: {collection_name}")
        db.drop_collection(collection_name)
    logger.info("Successfully dropped all existing collections")

    # Populate collections
    populate_entity_types(ENTITY_2_LABEL, ENTITY_2_HIERARCHY, subj2prop_constraints, obj2prop_constraints, 
                         db, collection_name=args.entity_types_collection)
    
    populate_entity_type_aliases(ENTITY_2_LABEL, ENTITY_2_ALIASES, db, 
                               collection_name=args.entity_type_aliases_collection)
    
    populate_properties(PROP_2_LABEL, PROP_2_CONSTRAINT, db,
                       collection_name=args.properties_collection)
    
    populate_property_aliases(PROP_2_LABEL, PROP_2_ALIASES, db,
                            collection_name=args.property_aliases_collection)
    
    db.create_collection(args.entity_collection)
    
    # Create indexes
    create_index_for_entity_types(db, collection_name=args.entity_type_aliases_collection, 
                                index_name=args.entity_types_index)
    create_index_for_properties(db, collection_name=args.property_aliases_collection, 
                              index_name=args.property_aliases_index)
    create_index_for_entities(db, collection_name=args.entity_collection,
                              index_name=args.entity_index)
    
    logger.info("Database population process completed")