import json
import os
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv, find_dotenv
from pymongo.mongo_client import MongoClient
from tqdm import tqdm
import sys

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "analysis"))

from dump_kg import dump_kg_from_backend

from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.structured_aligner import Aligner as StructuredDBAligner
from wikontic.utils.inference_with_db import InferenceWithDB

from wikontic.utils.dynamic_aligner import Aligner as DynamicDBAligner
from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from wikontic.db.factory import create_backend

from wikontic.create_ontological_triplets_db import create_ontological_triplets_database
from wikontic.create_triplets_db import create_triplets_database
from wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database

import argparse

parser = argparse.ArgumentParser(
    description="Run KG construction with optional config file."
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to the config YAML file (overrides default/config from env var)",
)

logger = logging.getLogger("KGConstructionWithDB")
logger.setLevel(logging.ERROR)

_ = load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

CONFIG_DEFAULTS = {
    "mongo_uri": "mongodb://localhost:27018/?directConnection=true",
    "vector_db_backend": "mongodb",
    "qdrant_url": ":memory:",
    "qdrant_api_key": None,
    "ontology_db_name": "wikidata_ontology",
    "triplets_db_name": "triplets_db",
    "model_name": "gpt-4o-mini",
    "dataset_path": "datasets/musique_200_test_preprocessed.json",
    "preprocessing": "musique",
    "sample_start_index": 0,
    "num_samples": 50,
    "structured_inference": True,
    "dump_kg": False,
    "proxy_env_var": None,
    "base_url_env_var": "OPENROUTER_BASE_URL",
    "api_key_env_var": "KEY",
}


def load_config(config_path=None):
    if config_path:
        path = Path(config_path)
    elif os.environ.get("KG_CONSTRUCTION_CONFIG"):
        path = Path(os.environ["KG_CONSTRUCTION_CONFIG"])
    else:
        path = Path(__file__).resolve().with_name("musique_inference_with_db.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    merged = dict(CONFIG_DEFAULTS)
    for key in CONFIG_DEFAULTS:
        if key in raw:
            merged[key] = raw[key] if raw[key] is not None else CONFIG_DEFAULTS[key]

    return SimpleNamespace(**merged)


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client

def get_json_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        ds = json.load(f)
    return ds


def should_dump_kg(cfg) -> bool:
    """In-memory Qdrant is ephemeral — always persist KG to JSON. Other backends: opt-in."""
    if cfg.vector_db_backend == "qdrant" and cfg.qdrant_url == ":memory:":
        return True
    return bool(cfg.dump_kg)


ONTOLOGY_COLLECTIONS = {
    "entity_types",
    "entity_type_aliases",
    "properties",
    "property_aliases",
}
ONTO_TRIPLETS_COLLECTIONS = {
    "entity_aliases",
    "triplets",
    "initial_triplets",
    "filtered_triplets",
    "ontology_filtered_triplets",
}
TRIPLETS_COLLECTIONS = {
    "entity_aliases",
    "property_aliases",
    "triplets",
    "initial_triplets",
    "filtered_triplets",
}


def _collections_ready(backend, required):
    existing = set(backend.list_collection_names())
    return required.issubset(existing)


def _create_qdrant_backend(cfg):
    return create_backend(
        backend_type="qdrant",
        qdrant_url=cfg.qdrant_url,
        qdrant_api_key=cfg.qdrant_api_key,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)

    triplets_db_name = (
        cfg.triplets_db_name + "_" + cfg.model_name.replace("/", "_").replace(".", "_")
    )
    if cfg.structured_inference:
        triplets_db_name = triplets_db_name + "_onto"
    else:
        triplets_db_name = triplets_db_name + "_non_onto"

    mongo_client = get_mongo_client(cfg.mongo_uri) if cfg.vector_db_backend == "mongodb" else None
    qdrant_backend = (
        _create_qdrant_backend(cfg) if cfg.vector_db_backend == "qdrant" else None
    )

    if cfg.structured_inference:
        if cfg.vector_db_backend == "mongodb":
            ontology_db = create_backend(
                backend_type="mongodb",
                mongo_db=mongo_client.get_database(cfg.ontology_db_name),
            )
            if not ontology_db.list_collection_names():
                ontology_db = create_wikidata_ontology_database(
                    backend=cfg.vector_db_backend,
                    mongo_uri=cfg.mongo_uri,
                    qdrant_url=cfg.qdrant_url,
                    qdrant_api_key=cfg.qdrant_api_key,
                    database=cfg.ontology_db_name,
                )
                logger.info(
                    f"Wikidata ontology database created successfully: {cfg.ontology_db_name}"
                )
            triplets_db = create_backend(
                backend_type="mongodb",
                mongo_db=mongo_client.get_database(triplets_db_name),
            )
            if not triplets_db.list_collection_names():
                triplets_db = create_ontological_triplets_database(
                    backend=cfg.vector_db_backend,
                    mongo_uri=cfg.mongo_uri,
                    qdrant_url=cfg.qdrant_url,
                    qdrant_api_key=cfg.qdrant_api_key,
                    db_name=triplets_db_name,
                )
                logger.info(
                    f"Ontological triplets database created successfully: {triplets_db_name}"
                )
        else:
            ontology_db = qdrant_backend
            if not _collections_ready(ontology_db, ONTOLOGY_COLLECTIONS):
                ontology_db = create_wikidata_ontology_database(
                    backend=cfg.vector_db_backend,
                    mongo_uri=cfg.mongo_uri,
                    qdrant_url=cfg.qdrant_url,
                    qdrant_api_key=cfg.qdrant_api_key,
                    database=cfg.ontology_db_name,
                    storage_backend=qdrant_backend,
                )
                logger.info("Wikidata ontology collections created in Qdrant backend")
            triplets_db = ontology_db
            if not _collections_ready(triplets_db, ONTO_TRIPLETS_COLLECTIONS):
                triplets_db = create_ontological_triplets_database(
                    backend=cfg.vector_db_backend,
                    mongo_uri=cfg.mongo_uri,
                    qdrant_url=cfg.qdrant_url,
                    qdrant_api_key=cfg.qdrant_api_key,
                    db_name=triplets_db_name,
                    storage_backend=ontology_db,
                )
                logger.info(
                    f"Ontological triplets collections created successfully: {triplets_db_name}"
                )
    else:
        if cfg.vector_db_backend == "mongodb":
            triplets_db = create_backend(
                backend_type="mongodb",
                mongo_db=mongo_client.get_database(triplets_db_name),
            )
            if not triplets_db.list_collection_names():
                triplets_db = create_triplets_database(
                    backend=cfg.vector_db_backend,
                    mongo_uri=cfg.mongo_uri,
                    qdrant_url=cfg.qdrant_url,
                    qdrant_api_key=cfg.qdrant_api_key,
                    db_name=triplets_db_name,
                )
                logger.info(f"Triplets database created successfully: {triplets_db_name}")
        else:
            triplets_db = qdrant_backend
            if not _collections_ready(triplets_db, TRIPLETS_COLLECTIONS):
                triplets_db = create_triplets_database(
                    backend=cfg.vector_db_backend,
                    mongo_uri=cfg.mongo_uri,
                    qdrant_url=cfg.qdrant_url,
                    qdrant_api_key=cfg.qdrant_api_key,
                    db_name=triplets_db_name,
                    storage_backend=qdrant_backend,
                )
                logger.info(f"Triplets collections created successfully: {triplets_db_name}")
    api_key = os.getenv(cfg.api_key_env_var)
    # api_key = ""
    proxy_url = os.getenv(cfg.proxy_env_var) if cfg.proxy_env_var else None
    base_url = os.getenv(cfg.base_url_env_var)

    ds = get_json_dataset(cfg.dataset_path)

    extractor = LLMTripletExtractor(
        model=cfg.model_name, api_key=api_key, proxy=proxy_url, base_url=base_url
    )
    if cfg.structured_inference:
        logger.info("Structured inference enabled")
        aligner = StructuredDBAligner(ontology_db=ontology_db, triplets_db=triplets_db)
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )
    else:
        logger.info("Structured inference disabled, using dynamic inference")
        aligner = DynamicDBAligner(triplets_db=triplets_db)
        inference_with_db = InferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )   

    sampled_ids = list(ds.keys())[cfg.sample_start_index : cfg.sample_start_index + cfg.num_samples]

    for i, sample_id in tqdm(enumerate(sampled_ids), total=len(sampled_ids)):

        texts = ds[sample_id]

        for idx, text in tqdm(enumerate(texts), total=len(texts)):
            if (
                triplets_db.get_collection("triplets").count_documents(
                    {
                        "sample_id": sample_id,
                        "source_text_id": idx,
                    }
                )
                == 0
                and len(text.split()) != 0
            ):
                if cfg.structured_inference:
                    (
                        initial_triplets,
                        final_triplets,
                        filtered_triplets,
                        ontology_filtered_triplets,
                    ) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
                        text, sample_id=sample_id, source_text_id=idx
                    )
                else:
                    initial_triplets, final_triplets, filtered_triplets = (
                        inference_with_db.extract_triplets_and_add_to_db(
                            text, sample_id=sample_id, source_text_id=idx
                        )
                    )
        logger.info("CURRENT COST: %s", extractor.calculate_cost())
        logger.info("--------------------------------")

    if should_dump_kg(cfg):
        dump_path = dump_kg_from_backend(
            triplets_db,
            triplets_db_name,
            include_ontology_filtered=cfg.structured_inference,
        )
        logger.info("KG dump written to %s", dump_path)
