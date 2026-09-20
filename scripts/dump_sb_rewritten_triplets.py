"""Dump the sb_rewritten mock triplets DB used by demo_app/streamlit_session.py to JSON.

demo-mock's KG_Extraction page reads pre-extracted triplets for two fixed
passages ("ЮЛ - СК БУМ", source_text_id 1 and 3) plus a bulk seed of every
other sample from a local-only MongoDB database
(`triplets_db_sb_rewritten_Openai_Gpt-oss-120b_onto`) instead of calling the
LLM. Only the `triplets` and `initial_triplets` collections are ever read
from that database (see streamlit_session.py), so this dumps just those two.

Usage:
    python scripts/dump_sb_rewritten_triplets.py \
        --mongo_uri "mongodb://localhost:27018/?directConnection=true" \
        --out demo_app/data/sb_rewritten_triplets.json
"""

import argparse
import json

from pymongo import MongoClient

SB_REWRITTEN_DB_NAME = "triplets_db_sb_rewritten_Openai_Gpt-oss-120b_onto"
COLLECTIONS = ["triplets", "initial_triplets"]


def dump(mongo_uri: str, db_name: str, out_path: str) -> None:
    client = MongoClient(mongo_uri)
    db = client.get_database(db_name)

    dump_data = {}
    for collection_name in COLLECTIONS:
        docs = list(db[collection_name].find({}, {"_id": 0}))
        dump_data[collection_name] = docs
        print(f"{collection_name}: {len(docs)} documents")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27018/?directConnection=true",
    )
    parser.add_argument("--db_name", type=str, default=SB_REWRITTEN_DB_NAME)
    parser.add_argument(
        "--out", type=str, default="demo_app/data/sb_rewritten_triplets.json"
    )
    args = parser.parse_args()
    dump(args.mongo_uri, args.db_name, args.out)
