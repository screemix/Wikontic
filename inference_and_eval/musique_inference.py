import json
import os
import logging
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv, find_dotenv
from pymongo.mongo_client import MongoClient
from tqdm import tqdm
import sys

_analysis_dir = Path.cwd().resolve()
if _analysis_dir.name == "inference_and_eval":
    sys.path.insert(0, str(_analysis_dir.parent / "src"))

from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.structured_aligner import Aligner as StructuredDBAligner
from wikontic.utils.inference_with_db import InferenceWithDB
from wikontic.utils.dynamic_aligner import Aligner as DynamicDBAligner
from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB

from wikontic.create_ontological_triplets_db import create_ontological_triplets_database
from wikontic.create_triplets_db import create_triplets_database
from wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database

import argparse

parser = argparse.ArgumentParser(
    description="Run Musique inference with optional config file."
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to the config YAML file (overrides default/config from env var)",
)

logger = logging.getLogger("MusiqueInferenceWithDB")
logger.setLevel(logging.ERROR)

_ = load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

CONFIG_DEFAULTS = {
    "mongo_uri": "mongodb://localhost:27018/?directConnection=true",
    "ontology_db_name": "wikidata_ontology",
    "triplets_db_name": "triplets_db",
    "model_name": "gpt-4o-mini",
    "dataset_path": "datasets/musique_200_test.json",
    "num_samples": 50,
    "structured_inference": False,
    "proxy_env_var": None,
    "base_url_env_var": "OPENROUTER_BASE_URL",
    "api_key_env_var": "KEY",
}


def load_config(config_path=None):
    if config_path:
        path = Path(config_path)
    elif os.environ.get("MUSIQUE_INFERENCE_CONFIG"):
        path = Path(os.environ["MUSIQUE_INFERENCE_CONFIG"])
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


def get_jsonl_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        ds = [json.loads(line) for line in f]
    return ds


def convert_musique_dataset(ds):
    id2sample = {}
    for elem in ds:
        answerable = "Answerable" if elem["answerable"] else "Unanswerable"
        id2sample[elem["id"] + "_" + answerable] = elem
    return id2sample


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

    mongo_client = get_mongo_client(cfg.mongo_uri)
    if cfg.structured_inference:
        ontology_db = mongo_client.get_database(cfg.ontology_db_name)
        if ontology_db is None:
            create_wikidata_ontology_database(
                mongo_uri=cfg.mongo_uri, database=cfg.ontology_db_name
            )
            ontology_db = mongo_client.get_database(cfg.ontology_db_name)
            logger.info(
                f"Wikidata ontology database created successfully: {cfg.ontology_db_name}"
            )
        triplets_db = mongo_client.get_database(triplets_db_name)
        if triplets_db is None:
            create_ontological_triplets_database(
                mongo_uri=cfg.mongo_uri,
                db_name=triplets_db_name,
            )
            triplets_db = mongo_client.get_database(triplets_db_name)
            logger.info(
                f"Ontological triplets database created successfully: {triplets_db_name}"
            )
    else:
        triplets_db = mongo_client.get_database(triplets_db_name)
        if triplets_db is None:
            create_triplets_database(mongo_uri=cfg.mongo_uri, db_name=triplets_db_name)
            triplets_db = mongo_client.get_database(triplets_db_name)
            logger.info(f"Triplets database created successfully: {triplets_db_name}")
    api_key = os.getenv(cfg.api_key_env_var)
    proxy_url = cfg.proxy_env_var if cfg.proxy_env_var else None
    base_url = os.getenv(cfg.base_url_env_var)

    ds = (
        get_json_dataset(cfg.dataset_path)
        if cfg.dataset_path.endswith(".json")
        else get_jsonl_dataset(cfg.dataset_path)
    )
    if cfg.structured_inference:
        logger.info("Structured inference enabled")
        aligner = StructuredDBAligner(ontology_db=ontology_db, triplets_db=triplets_db)
    else:
        logger.info("Structured inference disabled, using dynamic inference")
        aligner = DynamicDBAligner(triplets_db=triplets_db)

    extractor = LLMTripletExtractor(
        model=cfg.model_name, api_key=api_key, proxy=proxy_url, base_url=base_url
    )

    if cfg.structured_inference:
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )
    else:
        inference_with_db = InferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )

    id2sample = convert_musique_dataset(ds)
    sampled_ids = list(id2sample.keys())[82 : cfg.num_samples]

    for i, sample_id in tqdm(enumerate(sampled_ids), total=len(sampled_ids)):

        sample = id2sample[sample_id]
        texts = [item["paragraph_text"] for item in sample["paragraphs"]]

        for idx, text in tqdm(enumerate(texts), total=len(texts)):

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
