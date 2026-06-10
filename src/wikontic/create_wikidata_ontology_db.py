from typing import List
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
import json
import time
import argparse
import logging
import os
from pathlib import Path
import torch
from dotenv import load_dotenv, find_dotenv
from pymongo.mongo_client import MongoClient

from wikontic.db.bootstrap import ensure_collections
from wikontic.db.factory import create_backend
from wikontic.utils.contriever_model import load_contriever

_ = load_dotenv(find_dotenv())
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_tokenizer = None
_model = None
_contriever_device = device


def _ensure_contriever():
    global _tokenizer, _model, _contriever_device
    if _model is None:
        _tokenizer, _model, _contriever_device = load_contriever(device=str(device))
    return _tokenizer, _model, _contriever_device


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
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    if not text or not isinstance(text, str):
        return None

    try:
        tokenizer, model, embed_device = _ensure_contriever()
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(embed_device)
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
        return embeddings.detach().cpu().tolist()[0]

    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
        return None


def get_mongo_client(mongo_uri):
    return MongoClient(mongo_uri)


def populate_entity_types(
    ENTITY_2_LABEL,
    ENTITY_2_HIERARCHY,
    SUBJ_2_PROP_CONSTRAINTS,
    OBJ_2_PROP_CONSTRAINTS,
    db,
    collection_name="entity_types",
):
    logger.info(f"Starting to populate {collection_name} collection")
    entity_metadata_list = []

    for i, entity in enumerate(ENTITY_2_LABEL.keys()):
        label = ENTITY_2_LABEL[entity]
        parents = ENTITY_2_HIERARCHY[entity]

        valid_subject_property_ids = (
            SUBJ_2_PROP_CONSTRAINTS[entity] if entity in SUBJ_2_PROP_CONSTRAINTS else []
        )
        valid_object_property_ids = (
            OBJ_2_PROP_CONSTRAINTS[entity] if entity in OBJ_2_PROP_CONSTRAINTS else []
        )

        entity_metadata_list.append(
            {
                "_id": i,
                "entity_type_id": entity,
                "label": label,
                "parent_type_ids": parents,
                "valid_subject_property_ids": valid_subject_property_ids,
                "valid_object_property_ids": valid_object_property_ids,
            }
        )

    entity_metadata_list.append(
        {
            "_id": i + 1,
            "entity_type_id": "ANY",
            "label": "ANY",
            "parent_type_ids": [],
            "valid_subject_property_ids": SUBJ_2_PROP_CONSTRAINTS["<ANY SUBJECT>"],
            "valid_object_property_ids": OBJ_2_PROP_CONSTRAINTS["<ANY OBJECT>"],
        }
    )

    try:
        records = [EntityType(**record).model_dump() for record in entity_metadata_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_entity_type_aliases(
    ENTITY_2_LABEL, ENTITY_2_ALIASES, db, collection_name="entity_type_aliases"
):
    logger.info(f"Starting to populate {collection_name} collection")
    entity_types_list = []
    id_count = 0

    for e, aliases in tqdm(ENTITY_2_ALIASES.items()):
        alias_embedding = get_embedding(ENTITY_2_LABEL[e])
        entity_types_list.append(
            {
                "_id": id_count,
                "entity_type_id": e,
                "alias_label": ENTITY_2_LABEL[e],
                "alias_text_embedding": alias_embedding,
            }
        )
        id_count += 1

        for alias in aliases:
            alias_embedding = get_embedding(alias)
            entity_types_list.append(
                {
                    "_id": id_count,
                    "entity_type_id": e,
                    "alias_label": alias,
                    "alias_text_embedding": alias_embedding,
                }
            )
            id_count += 1
    try:
        records = [
            EntityTypeAlias(**record).model_dump() for record in entity_types_list
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_properties(
    PROP_2_LABEL, PROP_2_CONSTRAINT, db, collection_name="properties"
):
    logger.info(f"Starting to populate {collection_name} collection")
    property_list = []

    for i, prop_id in enumerate(PROP_2_LABEL.keys()):
        property_list.append(
            {
                "_id": i,
                "property_id": prop_id,
                "label": PROP_2_LABEL[prop_id],
                "valid_subject_type_ids": PROP_2_CONSTRAINT[prop_id][
                    "Subject type constraint"
                ],
                "valid_object_type_ids": PROP_2_CONSTRAINT[prop_id][
                    "Value-type constraint"
                ],
            }
        )

    try:
        records = [Property(**record).model_dump() for record in property_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_property_aliases(
    PROP_2_LABEL, PROP_2_ALIASES, db, collection_name="property_aliases"
):
    logger.info(f"Starting to populate {collection_name} collection")
    relation_alias_id_pairs = []
    id_count = 0

    for r, aliases in tqdm(PROP_2_ALIASES.items()):
        alias_embedding = get_embedding(PROP_2_LABEL[r])
        relation_alias_id_pairs.append(
            {
                "_id": id_count,
                "relation_id": r,
                "alias_label": PROP_2_LABEL[r],
                "alias_text_embedding": alias_embedding,
            }
        )
        id_count += 1

        for alias in aliases:
            alias_embedding = get_embedding(alias)
            relation_alias_id_pairs.append(
                {
                    "_id": id_count,
                    "relation_id": r,
                    "alias_label": alias,
                    "alias_text_embedding": alias_embedding,
                }
            )
            id_count += 1
    try:
        records = [
            PropertyAlias(**record).model_dump() for record in relation_alias_id_pairs
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")




def create_wikidata_ontology_database(
    backend: str = "mongodb",
    mongo_uri: str = "mongodb://localhost:27018/?directConnection=true",
    qdrant_url: str = ":memory:",
    qdrant_api_key: str = None,
    database: str = "wikidata_ontology",
    mappings_dir: str = None,
    entity_types_collection: str = "entity_types",
    entity_type_aliases_collection: str = "entity_type_aliases",
    properties_collection: str = "properties",
    property_aliases_collection: str = "property_aliases",
    entity_types_index: str = "entity_type_aliases",
    property_aliases_index: str = "property_aliases",
    embedding_dimensions: int = 768,
    drop_collections: bool = False,
):
    """
    Populate MongoDB with Wikidata ontology data.

    Args:
        mongo_uri: MongoDB connection URI
        database: MongoDB database name
        mappings_dir: Directory containing ontology mapping files. If None, uses default path.
        entity_types_collection: Collection name for entity types
        entity_type_aliases_collection: Collection name for entity type aliases
        properties_collection: Collection name for properties
        property_aliases_collection: Collection name for property aliases
        entity_types_index: Index name for entity types
        property_aliases_index: Index name for property aliases
        drop_collections: Whether to drop existing collections before creating new ones

    Returns:
        Database object
    """

    # Default mappings directory
    if mappings_dir is None:
        # Try to find the mappings directory relative to this file
        current_file = Path(__file__).parent
        mappings_dir = str(current_file /"src" / "wikontic" / "utils" / "ontology_mappings_en_en")
        if not os.path.exists(mappings_dir):
            # Fallback to relative path
            mappings_dir = "src/wikontic/utils/ontology_mappings_en_en"

    logger.info("Starting database population process")
    logger.info(f"Using database: {database}")
    logger.info(f"Loading mapping files from: {mappings_dir}")

    # Load mapping files
    with open(os.path.join(mappings_dir, "subj_constraint2prop.json"), "r") as f:
        subj2prop_constraints = json.load(f)

    with open(os.path.join(mappings_dir, "obj_constraint2prop.json"), "r") as f:
        obj2prop_constraints = json.load(f)

    with open(os.path.join(mappings_dir, "entity_type2label.json"), "r") as f:
        ENTITY_2_LABEL = json.load(f)

    with open(os.path.join(mappings_dir, "entity_type2hierarchy.json"), "r") as f:
        ENTITY_2_HIERARCHY = json.load(f)

    with open(os.path.join(mappings_dir, "entity_type2aliases.json"), "r") as f:
        ENTITY_2_ALIASES = json.load(f)

    with open(os.path.join(mappings_dir, "prop2constraints.json"), "r") as f:
        PROP_2_CONSTRAINT = json.load(f)

    with open(os.path.join(mappings_dir, "prop2label.json"), "r") as f:
        PROP_2_LABEL = json.load(f)

    with open(os.path.join(mappings_dir, "prop2aliases.json"), "r") as f:
        PROP_2_ALIASES = json.load(f)

    logger.info("Successfully loaded all mapping files")

    mongo_db = None
    if backend == "mongodb":
        mongo_client = get_mongo_client(mongo_uri)
        mongo_db = mongo_client.get_database(database)
    db = create_backend(
        backend_type=backend,
        mongo_db=mongo_db,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    ensure_collections(
        db,
        [
            entity_types_collection,
            entity_type_aliases_collection,
            properties_collection,
            property_aliases_collection,
        ],
        drop_collections=drop_collections,
    )

    # Populate collections
    populate_entity_types(
        ENTITY_2_LABEL,
        ENTITY_2_HIERARCHY,
        subj2prop_constraints,
        obj2prop_constraints,
        db,
        collection_name=entity_types_collection,
    )

    populate_entity_type_aliases(
        ENTITY_2_LABEL,
        ENTITY_2_ALIASES,
        db,
        collection_name=entity_type_aliases_collection,
    )

    populate_properties(
        PROP_2_LABEL, PROP_2_CONSTRAINT, db, collection_name=properties_collection
    )

    populate_property_aliases(
        PROP_2_LABEL,
        PROP_2_ALIASES,
        db,
        collection_name=property_aliases_collection,
    )

    db.create_indexes(entity_types_collection, [["entity_type_id"], ["label"]])
    db.create_indexes(entity_type_aliases_collection, [["entity_type_id"], ["alias_label"]])
    db.create_indexes(properties_collection, [["property_id"]])
    logger.info("Indexes created successfully")
    db.ensure_vector_index(
        collection_name=entity_type_aliases_collection,
        index_name=entity_types_index,
        vector_field="alias_text_embedding",
        num_dimensions=embedding_dimensions,
        token_fields=["entity_type_id"],
        recreate=drop_collections,
    )
    db.ensure_vector_index(
        collection_name=property_aliases_collection,
        index_name=property_aliases_index,
        vector_field="alias_text_embedding",
        num_dimensions=embedding_dimensions,
        token_fields=["relation_id"],
        recreate=drop_collections,
    )
    logger.info("Database population process completed")

    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate MongoDB with Wikidata ontology data"
    )

    parser.add_argument(
        "--mappings_dir",
        type=str,
        default="utils/ontology_mappings/",
        help="Directory containing ontology mapping files",
    )
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27018/?directConnection=true",
        help="MongoDB connection URI",
    )
    parser.add_argument("--backend", type=str, default="mongodb")
    parser.add_argument("--qdrant_url", type=str, default=":memory:")
    parser.add_argument("--qdrant_api_key", type=str, default=None)
    parser.add_argument(
        "--database",
        type=str,
        default="wikidata_ontology",
        help="MongoDB database name",
    )

    # Collection names
    parser.add_argument(
        "--entity_types_collection",
        type=str,
        default="entity_types",
        help="Collection name for entity types",
    )
    parser.add_argument(
        "--entity_type_aliases_collection",
        type=str,
        default="entity_type_aliases",
        help="Collection name for entity type aliases",
    )
    parser.add_argument(
        "--properties_collection",
        type=str,
        default="properties",
        help="Collection name for properties",
    )
    parser.add_argument(
        "--property_aliases_collection",
        type=str,
        default="property_aliases",
        help="Collection name for property aliases",
    )

    # Index names
    parser.add_argument(
        "--entity_types_index",
        type=str,
        default="entity_type_aliases",
        help="Index name for entity types",
    )
    parser.add_argument(
        "--property_aliases_index",
        type=str,
        default="property_aliases",
        help="Index name for property aliases",
    )

    args = parser.parse_args()
    create_wikidata_ontology_database(
        backend=args.backend,
        mongo_uri=args.mongo_uri,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        database=args.database,
        mappings_dir=args.mappings_dir,
        entity_types_collection=args.entity_types_collection,
        entity_type_aliases_collection=args.entity_type_aliases_collection,
        properties_collection=args.properties_collection,
        property_aliases_collection=args.property_aliases_collection,
        entity_types_index=args.entity_types_index,
        property_aliases_index=args.property_aliases_index,
    )
