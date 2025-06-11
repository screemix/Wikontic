from pymongo.mongo_client import MongoClient
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    logger.info("Connection to MongoDB successful")
    return client

def create_indexes(db):
    logger.info("Creating indexes for entity_types collection...")
    db.entity_types.create_index([("entity_type_id", 1)])
    db.entity_types.create_index([("label", 1)])

    logger.info("Creating indexes for entity_type_aliases collection...")
    db.entity_type_aliases.create_index([("entity_type_id", 1)])
    db.entity_type_aliases.create_index([("alias_label", 1)])

    logger.info("Creating indexes for properties collection...")
    db.properties.create_index([("property_id", 1)])

    # logger.info("Creating indexes for property_aliases collection...")
    # db.property_aliases.create_index("relation_id")

    logger.info("Creating indexes for entity_aliases collection...")
    db.entity_aliases.create_index([("entity_type", 1), ("sample_id", 1)])
    db.entity_aliases.create_index([("label", 1)])
    
    db.create_collection("triplets")
    logger.info("Creating indexes for triplets collection...")
    db.triplets.create_index([("sample_id", 1)])

    logger.info("All indexes created successfully")

if __name__ == "__main__":
    mongo_uri = "mongodb://localhost:27018/?directConnection=true"
    mongo_client = get_mongo_client(mongo_uri)
    db = mongo_client.get_database("wikidata_ontology")

    create_indexes(db) 