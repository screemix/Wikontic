from typing import List, Optional
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
from wikontic.utils.embedding_model import (
    EMBEDDING_DIMS,
    load_embedding_model,
    get_embeddings as _embedding_model_get_embeddings,
)

from wikontic.logging_config import get_logger
from wikontic.utils.language_config import ontology_mappings_dir_for_language

_ = load_dotenv(find_dotenv())
# Configure logging
logger = get_logger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_tokenizer = None
_model = None
_embed_device = device
_embed_model_name: str = "contriever"


def _ensure_embedding_model(model_name: str = "contriever"):
    global _tokenizer, _model, _embed_device, _embed_model_name
    if _model is None or _embed_model_name != model_name:
        _tokenizer, _model, _embed_device = load_embedding_model(
            model_name, device=str(device)
        )
        _embed_model_name = model_name
    return _tokenizer, _model, _embed_device


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


EMBEDDING_BATCH_SIZE = 64


def _embed_chunk_with_oom_backoff(
    chunk: List[str], tokenizer, model, embed_device
) -> List[Optional[List[float]]]:
    """Embed one chunk, halving it and retrying on CUDA OOM instead of giving up.

    This GPU is shared with other processes, so free memory fluctuates — a
    chunk that fits one moment may not the next. Silently returning None for a
    whole chunk on failure would corrupt the ontology DB (missing vectors),
    so we only give up once we're down to a single, unembeddable text.
    """
    try:
        return _embedding_model_get_embeddings(
            chunk, tokenizer, model, embed_device, _embed_model_name
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(chunk) == 1:
            logger.error(f"Out of GPU memory embedding a single text: {chunk[0]!r}")
            raise
        mid = len(chunk) // 2
        return _embed_chunk_with_oom_backoff(
            chunk[:mid], tokenizer, model, embed_device
        ) + _embed_chunk_with_oom_backoff(chunk[mid:], tokenizer, model, embed_device)


def get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Embed many texts in one pass each, chunked so a single forward pass
    doesn't have to hold every alias in the ontology in memory at once."""
    tokenizer, model, embed_device = _ensure_embedding_model(_embed_model_name)
    embeddings: List[Optional[List[float]]] = []
    for start in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE)):
        chunk = texts[start : start + EMBEDDING_BATCH_SIZE]
        embeddings.extend(
            _embed_chunk_with_oom_backoff(chunk, tokenizer, model, embed_device)
        )
    return embeddings


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
        raise

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_entity_type_aliases(
    ENTITY_2_LABEL, ENTITY_2_ALIASES, db, collection_name="entity_type_aliases"
):
    logger.info(f"Starting to populate {collection_name} collection")

    entity_ids = []
    alias_labels = []
    for e, aliases in ENTITY_2_ALIASES.items():
        entity_ids.append(e)
        alias_labels.append(ENTITY_2_LABEL[e])
        for alias in aliases:
            entity_ids.append(e)
            alias_labels.append(alias)

    logger.info(f"Embedding {len(alias_labels)} entity type aliases in batches of {EMBEDDING_BATCH_SIZE}")
    embeddings = get_embeddings_batch(alias_labels)

    entity_types_list = [
        {
            "_id": id_count,
            "entity_type_id": entity_id,
            "alias_label": alias_label,
            "alias_text_embedding": embedding,
        }
        for id_count, (entity_id, alias_label, embedding) in enumerate(
            zip(entity_ids, alias_labels, embeddings)
        )
    ]
    try:
        records = [
            EntityTypeAlias(**record).model_dump() for record in entity_types_list
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")
        raise

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
        raise

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_property_aliases(
    PROP_2_LABEL, PROP_2_ALIASES, db, collection_name="property_aliases"
):
    logger.info(f"Starting to populate {collection_name} collection")

    relation_ids = []
    alias_labels = []
    for r, aliases in PROP_2_ALIASES.items():
        relation_ids.append(r)
        alias_labels.append(PROP_2_LABEL[r])
        for alias in aliases:
            relation_ids.append(r)
            alias_labels.append(alias)

    logger.info(f"Embedding {len(alias_labels)} property aliases in batches of {EMBEDDING_BATCH_SIZE}")
    embeddings = get_embeddings_batch(alias_labels)

    relation_alias_id_pairs = [
        {
            "_id": id_count,
            "relation_id": relation_id,
            "alias_label": alias_label,
            "alias_text_embedding": embedding,
        }
        for id_count, (relation_id, alias_label, embedding) in enumerate(
            zip(relation_ids, alias_labels, embeddings)
        )
    ]
    try:
        records = [
            PropertyAlias(**record).model_dump() for record in relation_alias_id_pairs
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")
        raise

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")




def _resolve_mappings_dir(
    mappings_dir: Optional[str],
    language: str,
    fallback_language: str,
) -> str:
    if mappings_dir is not None:
        path = Path(mappings_dir)
    else:
        path = ontology_mappings_dir_for_language(language, fallback_language)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Ontology mappings directory not found: {path}. "
            "Pass mappings_dir explicitly or generate mappings for the requested language."
        )
    return str(path)


def create_wikidata_ontology_database(
    backend: str = "mongodb",
    mongo_uri: str = "mongodb://localhost:27018/?directConnection=true",
    qdrant_url: str = ":memory:",
    qdrant_api_key: str = None,
    database: str = "wikidata_ontology",
    language: str = "en",
    fallback_language: str = "en",
    mappings_dir: str = None,
    entity_types_collection: str = "entity_types",
    entity_type_aliases_collection: str = "entity_type_aliases",
    properties_collection: str = "properties",
    property_aliases_collection: str = "property_aliases",
    entity_types_index: str = "entity_type_aliases",
    property_aliases_index: str = "property_aliases",
    embedding_dimensions: int = None,
    drop_collections: bool = False,
    storage_backend=None,
):
    """
    Populate MongoDB with Wikidata ontology data.

    Args:
        mongo_uri: MongoDB connection URI
        database: MongoDB database name
        language: Source language for ontology labels/aliases (`en` or `ru`).
        fallback_language: Fallback language suffix in mapping directory name (default `en`).
        mappings_dir: Directory containing ontology mapping files. If None, uses
            ontology_mappings_{language}_{fallback_language} from language_config.
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

    from wikontic.utils.language_config import default_embedding_model_for_language

    embedding_model = default_embedding_model_for_language(language)
    if embedding_dimensions is None:
        embedding_dimensions = EMBEDDING_DIMS[embedding_model]
    _ensure_embedding_model(embedding_model)
    logger.info("Language: %s → embedding model: %s (%d dims)", language, embedding_model, embedding_dimensions)

    mappings_dir = _resolve_mappings_dir(mappings_dir, language, fallback_language)

    logger.info("Starting database population process")
    logger.info(f"Using database: {database}")
    logger.info(
        "Loading mapping files (language=%s, fallback=%s) from: %s",
        language,
        fallback_language,
        mappings_dir,
    )

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
    if storage_backend is not None:
        db = storage_backend
    elif backend == "mongodb":
        mongo_client = get_mongo_client(mongo_uri)
        mongo_db = mongo_client.get_database(database)
        db = create_backend(
            backend_type=backend,
            mongo_db=mongo_db,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    else:
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
        "--language",
        type=str,
        default="en",
        help="Source language for ontology mappings (en or ru)",
    )
    parser.add_argument(
        "--fallback_language",
        type=str,
        default="en",
        help="Fallback language suffix in mapping directory name (default: en)",
    )
    parser.add_argument(
        "--mappings_dir",
        type=str,
        default=None,
        help="Override ontology mapping directory (default: ontology_mappings_{language}_{fallback})",
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
        language=args.language,
        fallback_language=args.fallback_language,
        mappings_dir=args.mappings_dir,
        entity_types_collection=args.entity_types_collection,
        entity_type_aliases_collection=args.entity_type_aliases_collection,
        properties_collection=args.properties_collection,
        property_aliases_collection=args.property_aliases_collection,
        entity_types_index=args.entity_types_index,
        property_aliases_index=args.property_aliases_index,
    )
