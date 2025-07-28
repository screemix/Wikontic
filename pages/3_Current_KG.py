# --- File: 0_KG_Extraction.py ---
import streamlit as st
from pyvis.network import Network
# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
# from neo4j import GraphDatabase
from utils.structured_inference_with_db import extract_triplets
from utils.structured_dynamic_index_utils_with_db import Aligner
from utils.openai_utils import LLMTripletExtractor
from pymongo import MongoClient
import uuid
import logging
import sys
import base64
from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger('KGExtraction')
logger.setLevel(logging.INFO)


# Ensure the same user_id across all pages
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id
logger.info(f"User ID: {user_id}")

st.set_page_config(
    page_title="Wikontic",
    page_icon="media/wikotic-wo-text.png",
    layout="wide"
)

# --- Mongo Setup ---
_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client.get_database("wikidata_ontology")

# --- Extractor Setup ---
# extractor = LLMTripletExtractor(model='gpt-4.1-mini')
# --- Aligner Setup ---
@st.cache_resource(show_spinner="Loading Contriever model...")
def load_contriever_model():
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', token=os.getenv("HF_KEY"))
    model = AutoModel.from_pretrained('facebook/contriever', token=os.getenv("HF_KEY"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        'facebook/contriever',
        token=os.getenv("HF_KEY"),
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_contriever_model()
aligner = Aligner(db, model, tokenizer, device)


def fetch_triplets():
    collection = db.get_collection('triplets')
    query = {
            "sample_id": user_id
        }
    results = collection.find(query, {"_id": 0, "subject": 1, "relation": 1, "object": 1})
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


# --- Visualize ---
def visualize_knowledge_graph(triplets, ):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    added_nodes = set()

    for s, r, o in triplets:
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(node, label=node,
                             color="#C7C8CC")
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


subgraph = fetch_triplets()
# st.session_state.kg = nx.DiGraph()
# for s, r, o in subgraph:
    # st.session_state.kg.add_edge(s, o, label=r, highlight=s in new_entities or o in new_entities)
st.success(f"✅ Retrieved {len(subgraph)} triplets.")
st.subheader("Current Knowledge Graph")
visualize_knowledge_graph(subgraph)

with st.expander("🗑 Drop Knowledge Graph", expanded=True):
    st.markdown("""⚠️ This action button will drop the knowledge graph built in current session.""")
    confirm = st.checkbox("Confirm Drop")
    drop_button = st.button("Drop")
    if confirm and drop_button:
        collection = db.get_collection('triplets')
        collection.delete_many({"sample_id": user_id})
        st.success("Knowledge Graph dropped.")
        logger.info(f"Knowledge Graph dropped for user {user_id}")
        st.stop()
