from pymongo.mongo_client import MongoClient
import json
from tqdm import tqdm
import os

import argparse


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client


def dump_kg(db_name, include_initial_triplets, num_samples):

    sample_ids = list(
        mongo_client.get_database(db_name)
        .get_collection("initial_triplets")
        .distinct("sample_id")
    )
    if num_samples is not None:
        sample_ids = sample_ids[:num_samples]

    kg_dump = {}
    for sample_id in tqdm(sample_ids, total=len(sample_ids)):
        kg_dump[sample_id] = {}
        source_text_ids = list(
            mongo_client.get_database(db_name)
            .get_collection("initial_triplets")
            .find({"sample_id": sample_id})
            .distinct("source_text_id")
        )

        source_text_ids = list(set(source_text_ids))

        for source_text_id in source_text_ids:
            initial_triplets = (
                mongo_client.get_database(db_name)
                .get_collection("initial_triplets")
                .find(
                    {"sample_id": sample_id, "source_text_id": source_text_id},
                    {
                        "_id": 0,
                        "subject": 1,
                        "relation": 1,
                        "object": 1,
                        "subject_type": 1,
                        "object_type": 1,
                        "qualifiers": 1,
                    },
                )
            )
            
            triplets = (
                mongo_client.get_database(db_name)
                .get_collection("triplets")
                .find(
                    {"sample_id": sample_id, "source_text_id": source_text_id},
                    {
                        "_id": 0,
                        "subject": 1,
                        "relation": 1,
                        "object": 1,
                        "subject_type": 1,
                        "object_type": 1,
                        "qualifiers": 1,
                    },
                )
            )
            ontology_filtered_triplets = (
                mongo_client.get_database(db_name)
                .get_collection("ontology_filtered_triplets")
                .find(
                    {"sample_id": sample_id, "source_text_id": source_text_id},
                    {
                        "_id": 0,
                        "subject": 1,
                        "relation": 1,
                        "object": 1,
                        "subject_type": 1,
                        "object_type": 1,
                        "qualifiers": 1,
                    },
                )
            )

            filtered_triplets = (
                mongo_client.get_database(db_name)
                .get_collection("filtered_triplets")
                .find(
                    {"sample_id": sample_id, "source_text_id": source_text_id},
                    {
                        "_id": 0,
                        "subject": 1,
                        "relation": 1,
                        "object": 1,
                        "subject_type": 1,
                        "object_type": 1,
                        "qualifiers": 1,
                    },
                )
            )

            kg_dump[sample_id][source_text_id] = {
                "initial_triplets": list(initial_triplets),
                "triplets": list(triplets),
                "ontology_filtered_triplets": list(ontology_filtered_triplets),
                "filtered_triplets": list(filtered_triplets),
            }
            if include_initial_triplets:
                assert len(
                    kg_dump[sample_id][source_text_id]["initial_triplets"]
                ) == mongo_client.get_database(db_name).get_collection(
                    "initial_triplets"
                ).count_documents(
                    {"sample_id": sample_id, "source_text_id": source_text_id}
                )
            assert len(
                kg_dump[sample_id][source_text_id]["triplets"]
            ) == mongo_client.get_database(db_name).get_collection(
                "triplets"
            ).count_documents(
                {"sample_id": sample_id, "source_text_id": source_text_id}
            )
            assert len(
                kg_dump[sample_id][source_text_id]["ontology_filtered_triplets"]
            ) == mongo_client.get_database(db_name).get_collection(
                "ontology_filtered_triplets"
            ).count_documents(
                {"sample_id": sample_id, "source_text_id": source_text_id}
            )
            assert len(
                kg_dump[sample_id][source_text_id]["filtered_triplets"]
            ) == mongo_client.get_database(db_name).get_collection(
                "filtered_triplets"
            ).count_documents(
                {"sample_id": sample_id, "source_text_id": source_text_id}
            )

    if not os.path.exists("kg_dump"):
        os.makedirs("kg_dump")

    with open(f"kg_dump/kg_dump_{db_name}.json", "w") as f:
        json.dump(kg_dump, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default="triplets_db")
    parser.add_argument("--include_initial_triplets", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    db_name = args.db_name
    include_initial_triplets = args.include_initial_triplets
    num_samples = args.num_samples
    mongo_uri = "mongodb://localhost:27018/?directConnection=true"
    mongo_client = get_mongo_client(mongo_uri)
    dump_kg(db_name, include_initial_triplets, num_samples)
