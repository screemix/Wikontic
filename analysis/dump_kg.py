from pymongo.mongo_client import MongoClient
import json
from tqdm import tqdm
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import argparse

TRIPLET_EXPORT_FIELDS = (
    "subject",
    "relation",
    "object",
    "subject_type",
    "object_type",
    "qualifiers",
    "sample_id",
    "source_text_id",
)

KG_DUMP_COLLECTIONS: Tuple[Tuple[str, str], ...] = (
    ("initial_triplets", "initial_triplets"),
    ("triplets", "triplets"),
    ("ontology_filtered_triplets", "ontology_filtered_triplets"),
    ("filtered_triplets", "filtered_triplets"),
)


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client


def _export_triplet(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {key: doc[key] for key in TRIPLET_EXPORT_FIELDS if key in doc}


def _mongo_triplet_projection() -> Dict[str, int]:
    return {"_id": 0, **{field: 1 for field in TRIPLET_EXPORT_FIELDS}}


def _distinct_values(docs: Iterable[Dict[str, Any]], field: str) -> List[Any]:
    values = {doc[field] for doc in docs if field in doc and doc[field] is not None}
    return sorted(values, key=lambda value: (isinstance(value, str), value))


def dump_kg_from_backend(
    triplets_db,
    dump_name: str,
    *,
    output_dir: str = "kg_dump",
    include_ontology_filtered: Optional[bool] = None,
) -> str:
    """Dump KG triplets from a StorageBackend (e.g. Qdrant :memory:) to JSON."""
    if include_ontology_filtered is None:
        include_ontology_filtered = (
            "ontology_filtered_triplets" in triplets_db.list_collection_names()
        )

    collection_names = {name for name, _ in KG_DUMP_COLLECTIONS}
    if not include_ontology_filtered:
        collection_names.discard("ontology_filtered_triplets")

    initial_docs = triplets_db.match_documents("initial_triplets", {})
    sample_ids = _distinct_values(initial_docs, "sample_id")

    kg_dump: Dict[str, Dict[Any, Dict[str, List[Dict[str, Any]]]]] = {}
    for sample_id in tqdm(sample_ids, total=len(sample_ids)):
        kg_dump[sample_id] = {}
        sample_initial = triplets_db.match_documents(
            "initial_triplets", {"sample_id": sample_id}
        )
        source_text_ids = _distinct_values(sample_initial, "source_text_id")

        for source_text_id in source_text_ids:
            query = {"sample_id": sample_id, "source_text_id": source_text_id}
            entry: Dict[str, List[Dict[str, Any]]] = {}
            for dump_key, collection_name in KG_DUMP_COLLECTIONS:
                if collection_name not in collection_names:
                    entry[dump_key] = []
                    continue
                docs = triplets_db.match_documents(collection_name, query)
                entry[dump_key] = [_export_triplet(doc) for doc in docs]
                assert len(entry[dump_key]) == triplets_db.get_collection(
                    collection_name
                ).count_documents(query)
            kg_dump[sample_id][source_text_id] = entry

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"kg_dump_{dump_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg_dump, f, ensure_ascii=False)
    return output_path


def dump_kg(db_name):

    sample_ids = list(
        mongo_client.get_database(db_name)
        .get_collection("initial_triplets")
        .distinct("sample_id")
    )

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
            query = {"sample_id": sample_id, "source_text_id": source_text_id}
            projection = _mongo_triplet_projection()
            initial_triplets = (
                mongo_client.get_database(db_name)
                .get_collection("initial_triplets")
                .find(query, projection)
            )
            triplets = (
                mongo_client.get_database(db_name)
                .get_collection("triplets")
                .find(query, projection)
            )
            ontology_filtered_triplets = (
                mongo_client.get_database(db_name)
                .get_collection("ontology_filtered_triplets")
                .find(query, projection)
            )
            filtered_triplets = (
                mongo_client.get_database(db_name)
                .get_collection("filtered_triplets")
                .find(query, projection)
            )

            kg_dump[sample_id][source_text_id] = {
                "initial_triplets": list(initial_triplets),
                "triplets": list(triplets),
                "ontology_filtered_triplets": list(ontology_filtered_triplets),
                "filtered_triplets": list(filtered_triplets),
            }
            assert len(kg_dump[sample_id][source_text_id]["initial_triplets"]) == mongo_client.get_database(db_name).get_collection(
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

    with open(f"kg_dump/kg_dump_{db_name}.json", "w", encoding="utf-8") as f:
        json.dump(kg_dump, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default="triplets_db")
    args = parser.parse_args()
    db_name = args.db_name

    mongo_uri = "mongodb://localhost:27018/?directConnection=true"
    mongo_client = get_mongo_client(mongo_uri)
    dump_kg(db_name)
