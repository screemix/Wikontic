import argparse
import logging
from pymongo.mongo_client import MongoClient

from wikontic.db.bootstrap import ensure_collections
from wikontic.db.factory import create_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_triplets_database(
    backend: str = "mongodb",
    mongo_uri: str = "mongodb://localhost:27018/?directConnection=true",
    qdrant_url: str = ":memory:",
    qdrant_api_key: str = None,
    db_name: str = "triplets_db",
    entity_aliases_collection: str = "entity_aliases",
    property_aliases_collection: str = "property_aliases",
    triplets_collection: str = "triplets",
    initial_triplets_collection: str = "initial_triplets",
    filtered_triplets_collection: str = "filtered_triplets",
    entity_aliases_index: str = "entity_aliases",
    property_aliases_index: str = "property_aliases",
    embedding_dimensions: int = 768,
    drop_collections: bool = False,
    storage_backend=None,
):
    """
    Create collections and indexes for the dynamic triplets database.

    Args:
        mongo_uri: MongoDB connection URI
        db_name: Name of the database to create
        entity_aliases_collection: Collection name for entity aliases
        triplets_collection: Collection name for triplets
        initial_triplets_collection: Collection name for initial triplets
        filtered_triplets_collection: Collection name for filtered triplets
        entity_aliases_index: Index name for entities
        embedding_dimensions: Dimensions for embeddings (default: 768)
        drop_collections: Whether to drop existing collections before creating new ones

    Returns:
        Database object
    """
    mongo_db = None
    if storage_backend is not None:
        storage_backend = storage_backend
    elif backend == "mongodb":
        mongo_client = MongoClient(mongo_uri)
        mongo_db = mongo_client.get_database(db_name)
        logger.info("Connection to MongoDB successful")
        storage_backend = create_backend(
            backend_type=backend,
            mongo_db=mongo_db,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    else:
        storage_backend = create_backend(
            backend_type=backend,
            mongo_db=mongo_db,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    ensure_collections(
        storage_backend,
        [
            entity_aliases_collection,
            property_aliases_collection,
            triplets_collection,
            initial_triplets_collection,
            filtered_triplets_collection,
        ],
        drop_collections=drop_collections,
    )

    alias_index_fields = [["sample_id"], ["label"]]
    storage_backend.create_indexes(entity_aliases_collection, alias_index_fields)
    storage_backend.create_indexes(property_aliases_collection, alias_index_fields)
    for coll in (
        triplets_collection,
        initial_triplets_collection,
        filtered_triplets_collection,
    ):
        storage_backend.create_indexes(coll, [["sample_id"]])

    storage_backend.ensure_vector_index(
        collection_name=entity_aliases_collection,
        index_name=entity_aliases_index,
        vector_field="alias_text_embedding",
        num_dimensions=embedding_dimensions,
        token_fields=["sample_id"],
        recreate=drop_collections,
    )
    storage_backend.ensure_vector_index(
        collection_name=property_aliases_collection,
        index_name=property_aliases_index,
        vector_field="alias_text_embedding",
        num_dimensions=embedding_dimensions,
        recreate=drop_collections,
    )
    logger.info("Collections created successfully for backend=%s", backend)
    return storage_backend


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create collections and indexes for the dynamic triplets database"
    )
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27018/?directConnection=true",
    )
    parser.add_argument("--backend", type=str, default="mongodb")
    parser.add_argument("--qdrant_url", type=str, default=":memory:")
    parser.add_argument("--qdrant_api_key", type=str, default=None)
    parser.add_argument("--db_name", type=str, default="triplets_db")
    parser.add_argument(
        "--entity_aliases_collection",
        type=str,
        default="entity_aliases",
        help="Collection name for entity aliases",
    )
    parser.add_argument(
        "--property_aliases_collection",
        type=str,
        default="property_aliases",
        help="Collection name for property aliases",
    )
    parser.add_argument(
        "--triplets_collection",
        type=str,
        default="triplets",
        help="Collection name for triplets",
    )
    parser.add_argument(
        "--initial_triplets_collection",
        type=str,
        default="initial_triplets",
        help="Collection name for initial triplets",
    )
    parser.add_argument(
        "--filtered_triplets_collection",
        type=str,
        default="filtered_triplets",
        help="Collection name for filtered triplets",
    )
    parser.add_argument(
        "--entity_aliases_index",
        type=str,
        default="entity_aliases",
        help="MongoDB Atlas Vector Search index name for entity aliases",
    )
    parser.add_argument(
        "--property_aliases_index",
        type=str,
        default="property_aliases",
        help="MongoDB Atlas Vector Search index name for property aliases",
    )
    parser.add_argument(
        "--embedding_dimensions",
        type=int,
        default=768,
        help="Embedding dimensions for MongoDB Atlas Vector Search indexes",
    )
    parser.add_argument(
        "--drop_collections", type=bool, default=False, help="Drop existing collections"
    )

    args = parser.parse_args()
    create_triplets_database(
        backend=args.backend,
        mongo_uri=args.mongo_uri,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        db_name=args.db_name,
        entity_aliases_collection=args.entity_aliases_collection,
        property_aliases_collection=args.property_aliases_collection,
        triplets_collection=args.triplets_collection,
        initial_triplets_collection=args.initial_triplets_collection,
        filtered_triplets_collection=args.filtered_triplets_collection,
        entity_aliases_index=args.entity_aliases_index,
        property_aliases_index=args.property_aliases_index,
        embedding_dimensions=args.embedding_dimensions,
        drop_collections=args.drop_collections,
    )
