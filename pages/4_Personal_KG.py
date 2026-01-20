import streamlit as st
from pyvis.network import Network

# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from pymongo import MongoClient
import uuid
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("PersonalKG")
logger.setLevel(logging.INFO)


st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)

WIKIDATA_ONTOLOGY_DB_NAME = "wikidata_ontology"
TRIPLETS_DB_NAME = "demo"
# --- Mongo Setup ---
_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
api_key = os.getenv("KEY")
proxy_url = os.getenv("PROXY_URL")
ontology_db = mongo_client.get_database(WIKIDATA_ONTOLOGY_DB_NAME)
triplets_db = mongo_client.get_database(TRIPLETS_DB_NAME)


aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)


def fetch_related_triplets(entities):
    collection = triplets_db.get_collection("triplets")
    query = {
        "$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
        "sample_id": "personal_kg",
    }
    results = collection.find(
        query, {"_id": 0, "subject": 1, "relation": 1, "object": 1}
    )
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


# --- Visualize ---
def visualize_knowledge_graph(triplets, highlight_entities=None):
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )
    highlight_entities = highlight_entities or set()
    added_nodes = set()

    for s, r, o in triplets:
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(
                    node,
                    label=node,
                    color="#B2CD9C" if node in highlight_entities else "#C7C8CC",
                )
                added_nodes.add(node)
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:

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
        <h1 style="margin: 0;">Build your personal Knowledge Graph!</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox(
    "Choose a model for KG extraction:", model_options, index=0
)


# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.subheader("Input name and surname of the person you want to extract KG for")
input_text = st.text_area(
    "Enter name and surname:",
    value=st.session_state.input_text,
    placeholder="Enter name and surname of the person you want to extract KG for",
    height=68,
    key="name_surname",
)

trigger = st.button("Extract and Visualize KG for the person")

if trigger:
    if not input_text:
        st.warning(
            "Please enter name and surname of the person you want to extract KG for."
        )
    elif not selected_model:
        st.warning("Please select a model for KG extraction for the person.")
    else:
        extractor = LLMTripletExtractor(
            model=selected_model, api_key=api_key, proxy=proxy_url
        )
        response = extractor.client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search"}],
            input=f"Search recent and relevant info about {input_text} in the internet and return a paragraph that summarizes the info on the person. Return only the paragraph, no other text.",
        )
        personal_text = response.output_text

        logger.info(f"Personal text: {personal_text}")
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )
        (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        ) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
            text=personal_text, sample_id="personal_kg", source_text_id=None
        )
        logger.info(f"Initial triplets: {initial_triplets}")
        logger.info("-" * 100)
        logger.info(f"Refined triplets: {final_triplets}")
        logger.info("-" * 100)
        logger.info(f"filtered_triplets: {filtered_triplets}")
        logger.info("-" * 100)
        logger.info(f"ontology_filtered_triplets: {ontology_filtered_triplets}")
        logger.info("-" * 100)
        new_entities = {t["subject"] for t in final_triplets} | {
            t["object"] for t in final_triplets
        }
        subgraph = fetch_related_triplets(list(new_entities))
        st.success(
            f"✅ Extracted {len(final_triplets)} triplets and visualized {len(subgraph)} related ones."
        )

        st.subheader("Expanded KG Subgraph")
        visualize_knowledge_graph(subgraph, highlight_entities=new_entities)
