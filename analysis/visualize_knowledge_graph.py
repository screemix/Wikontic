from pyvis.network import Network
import argparse
from pymongo.mongo_client import MongoClient
import os
from tqdm import tqdm


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client


def visualize_knowledge_graph(
    triplets,
    output_file,
):
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )
    added_nodes = set()

    for s, r, o in tqdm(triplets):
        for node in [s, o]:
            if node not in added_nodes:
                if "ITEM" not in node:
                    net.add_node(node, label=node, color="#C7C8CC")
                    added_nodes.add(node)
        if "ITEM" not in s and "ITEM" not in o:
            net.add_edge(s, o, label=r, color="#000000")
    net.save_graph(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default="triplets_db")
    args = parser.parse_args()
    db_name = args.db_name

    mongo_uri = "mongodb://localhost:27018/?directConnection=true"
    mongo_client = get_mongo_client(mongo_uri)
    triplets = (
        mongo_client.get_database(db_name)
        .get_collection("triplets")
        .find({}, {"_id": 0, "subject": 1, "relation": 1, "object": 1})
    )
    triplets = [
        (elem["subject"], elem["relation"], elem["object"]) for elem in triplets
    ]
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    visualize_knowledge_graph(triplets, os.path.join("graphs", f"graph_{db_name}.html"))
