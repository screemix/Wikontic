import datasets
import sys
from pathlib import Path

_analysis_dir = Path.cwd().resolve()
if _analysis_dir.name == "inference_and_eval":
    sys.path.insert(0, str(_analysis_dir.parent / "src"))

from wikontic.utils.structured_inference_with_db import (
    StructuredInferenceWithDB,
)
from wikontic.utils.structured_aligner import Aligner
from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.create_ontological_triplets_db import (
    create_ontological_triplets_database,
)
from dotenv import load_dotenv, find_dotenv
import os
import httpx
import argparse
from pymongo import MongoClient
from tqdm import tqdm
import logging
import time
import json

_ = load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Openai/Gpt-oss-120b")
    parser.add_argument("--num_samples", type=int, default=1000)
    return parser.parse_args()


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()  # Read entire file
        lines = content.splitlines(keepends=True)  # Keep original newlines

        i = 0
        while i < len(lines):
            # Collect all lines until we find a complete JSON object
            buffer = ""
            while i < len(lines):
                buffer += lines[i]
                try:
                    item = json.loads(buffer)
                    data.append(item)
                    i += 1
                    break
                except json.JSONDecodeError:
                    i += 1
            else:
                print(f"Warning: Incomplete JSON at end of file")
                break
    return data


def main():
    args = get_args()
    model_name = args.model_name
    api_key = os.getenv("AIRI_KEY_MAX")
    base_url = os.getenv("AIRI_BASE_URL")
    # proxy_url = os.getenv("PROXY_URL")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Number of samples: {args.num_samples}")
    dataset = load_jsonl("../datasets/edgar_preprocessed/year_1994.jsonl")

    # create_ontological_triplets_database(
    #     mongo_uri="mongodb://localhost:27018/?directConnection=true",
    #     db_name=f"edgar_1994_fixed_{model_name.replace('/', '_').replace('.','_')}_onto",
    #     drop_collections=False,
    # )

    mongo_client = get_mongo_client("mongodb://localhost:27018/?directConnection=true")
    triplets_db = mongo_client.get_database(
        f"edgar_1994_fixed_{model_name.replace('/', '_').replace('.', '_')}_onto"
    )
    ontology_db = mongo_client.get_database("wikidata_ontology")
    extractor = LLMTripletExtractor(
        model=model_name, api_key=api_key, base_url=base_url
    )

    aligner = Aligner(triplets_db=triplets_db, ontology_db=ontology_db)
    inferer = StructuredInferenceWithDB(extractor, aligner, triplets_db)

    time_per_sample = []
    for i in tqdm(range(30, args.num_samples)):
        time_start = time.time()
        ds_item = dataset[i]
        for section_key in ds_item.keys():
            if "section" in section_key:
                text = ds_item[section_key]["replaced"]
                if len(text.split()) != 0:
                    triplets = (
                        inferer.extract_triplets_with_ontology_filtering_and_add_to_db(
                            text=text,
                            sample_id=ds_item["filename"],
                            source_text_id=section_key,
                        )
                    )
        time_per_sample.append(time.time() - time_start)
        logger.info(f"Time per sample: {time.time() - time_start} seconds")

    logger.info(
        f"Average time per sample: {sum(time_per_sample) / len(time_per_sample)}"
    )


if __name__ == "__main__":
    main()
