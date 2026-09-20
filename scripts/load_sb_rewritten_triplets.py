"""Populate the sb_rewritten mock triplets DB from the JSON dump produced by
scripts/dump_sb_rewritten_triplets.py.

demo_app/streamlit_session.py (demo-mock branch) expects a MongoDB database
named `triplets_db_sb_rewritten_Openai_Gpt-oss-120b_onto` with `triplets` and
`initial_triplets` collections to exist, so it can serve pre-extracted
triplets instead of calling the LLM. Run this after MongoDB is up and before
launching the Streamlit app.

Usage:
    python scripts/load_sb_rewritten_triplets.py \
        --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
        --in demo_app/data/sb_rewritten_triplets.json
"""

import argparse
import json

from pymongo import MongoClient

SB_REWRITTEN_DB_NAME = "triplets_db_sb_rewritten_Openai_Gpt-oss-120b_onto"


def load(mongo_uri: str, db_name: str, in_path: str, drop_existing: bool) -> None:
    client = MongoClient(mongo_uri)
    db = client.get_database(db_name)

    with open(in_path, "r", encoding="utf-8") as f:
        dump_data = json.load(f)

    for collection_name, docs in dump_data.items():
        if drop_existing:
            db[collection_name].drop()
        if docs:
            db[collection_name].insert_many(docs)
        print(f"{collection_name}: inserted {len(docs)} documents into {db_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27018/?directConnection=true",
    )
    parser.add_argument("--db_name", type=str, default=SB_REWRITTEN_DB_NAME)
    parser.add_argument(
        "--in", dest="in_path", type=str, default="demo_app/data/sb_rewritten_triplets.json"
    )
    parser.add_argument(
        "--drop_existing",
        action="store_true",
        help="Drop each collection before inserting (default: append/insert only).",
    )
    args = parser.parse_args()
    load(args.mongo_uri, args.db_name, args.in_path, args.drop_existing)
