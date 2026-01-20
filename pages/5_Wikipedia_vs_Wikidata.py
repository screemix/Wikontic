# --- File: 0_KG_Extraction.py ---
import streamlit as st
from pyvis.network import Network

# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv

# from neo4j import GraphDatabase
from pymongo import MongoClient
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
import uuid
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("Wikipedia vs Wikidata")
logger.setLevel(logging.INFO)

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)

# --- Mongo Setup ---
_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
triplets_db = mongo_client.get_database("wiki_vs_wikidata")


def fetch_triplets(sample_id):
    collection = triplets_db.get_collection("triplets")
    query = {"sample_id": sample_id}
    results = collection.find(
        query, {"_id": 0, "subject": 1, "relation": 1, "object": 1}
    )
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


# --- Visualize ---
def visualize_knowledge_graph(
    triplets,
):
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )
    added_nodes = set()

    for s, r, o in triplets:
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(node, label=node, color="#C7C8CC")
                added_nodes.add(node)
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
        # graph_container.components.v1.html(f.read(), height=600, scrolling=True)
        # with expanded_kg_container:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)


# --- UI ---
with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">KG Viewer</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


wikipedia_subgraph = fetch_triplets(sample_id="wikipedia")
wikidata_subgraph = fetch_triplets(sample_id="wikidata")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Extracted KG from Wikipedia article")
    visualize_knowledge_graph(wikipedia_subgraph)

with col2:
    st.subheader("Wikidata KG")
    visualize_knowledge_graph(wikidata_subgraph)
