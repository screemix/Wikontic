import json
from tqdm import tqdm
import argparse
import warnings
from pymongo.mongo_client import MongoClient
import logging
import os

logger = logging.getLogger("MusiqueInferenceWithDB")
logger.setLevel(logging.ERROR)

from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.structured_aligner import Aligner as StructuredDBAligner
from wikontic.utils.inference_with_db import (
    InferenceWithDB,
)
from wikontic.utils.dynamic_aligner import Aligner as DynamicDBAligner
from wikontic.utils.structured_inference_with_db import (
    StructuredInferenceWithDB,
)

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client


def get_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        ds = json.load(f)
    return ds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27018/?directConnection=true",
    )
    parser.add_argument("--ontology_db_name", type=str, default="wikidata_ontology")
    parser.add_argument("--triplets_db_name", type=str, default="triplets_db")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--dataset_path", type=str, default="datasets/musique_200_test.json"
    )
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument(
        "--structured_inference",
        action="store_true",
        help="Enable structured inference",
    )
    parser.add_argument(
        "--no_structured_inference",
        action="store_false",
        dest="structured_inference",
        help="Disable structured inference",
    )

    parser.set_defaults(structured_inference=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    mongo_client = get_mongo_client(args.mongo_uri)
    ontology_db = mongo_client.get_database(args.ontology_db_name)
    triplets_db = mongo_client.get_database(args.triplets_db_name)
    model_name = args.model_name
    dataset_path = args.dataset_path
    num_samples = args.num_samples

    api_key = os.getenv("OPENROUTER_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    proxy_url = os.getenv("PROXY_URL")

    ds = get_dataset(dataset_path)
    if args.structured_inference:
        logger.info("Structured inference enabled")
        aligner = StructuredDBAligner(ontology_db=ontology_db, triplets_db=triplets_db)
    else:
        logger.info("Structured inference disabled, using dynamic inference")
        aligner = DynamicDBAligner(triplets_db=triplets_db)

    extractor = LLMTripletExtractor(
        model=model_name, api_key=api_key, base_url=base_url, proxy=proxy_url
    )

    if args.structured_inference:
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )
    else:
        inference_with_db = InferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )

    ds = ds["data"]

    id2sample = {}
    for elem in ds:
        id2sample[elem["id"]] = elem

    sampled_ids = list(id2sample.keys())[:num_samples]

    for i, sample_id in tqdm(enumerate(sampled_ids), total=len(sampled_ids)):
        sample = id2sample[sample_id]
        texts = [item["paragraph_text"] for item in sample["paragraphs"]]

        for idx, text in tqdm(enumerate(texts), total=len(texts)):
            if args.structured_inference:
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

        logger.info("CURRENT COST: ", extractor.calculate_cost())
        logger.info("--------------------------------")
