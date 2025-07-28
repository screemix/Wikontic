# --- File: 0_KG_Extraction.py ---
import streamlit as st
from pyvis.network import Network
# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
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


def fetch_related_triplets(entities):
    collection = db.get_collection('triplets')
    query = {"$or": [
                {"subject": {"$in": entities}},
                {"object": {"$in": entities}}
            ],
            "sample_id": user_id
        }
    results = collection.find(query, {"_id": 0, "subject": 1, "relation": 1, "object": 1})
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]



# --- Visualize ---
def visualize_knowledge_graph(triplets, highlight_entities=None):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    highlight_entities = highlight_entities or set()
    added_nodes = set()

    for s, r, o in triplets:
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(node, label=node,
                             color="#B2CD9C" if node in highlight_entities else "#C7C8CC")
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


def visualize_initial_knowledge_graph(initial_triplets):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)

    for t in initial_triplets:
        print(t)
        s, r, o = t['subject'], t['relation'], t['object']
        net.add_node(s, label=s, color="#B2CD9C")
        net.add_node(o, label=o, color="#B2CD9C")
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
        # graph_container.components.v1.html(f.read(), height=600, scrolling=True)
        # with initial_kg_container:
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
        <h1 style="margin: 0;">KG Extraction + Visualization</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox("Choose a model for KG extraction:", model_options, index=0)

input_text = st.text_area("Enter Text", placeholder="Paste your text here...")
trigger = st.button("Extract and Visualize")

if trigger:
    if not input_text:
        st.warning("Please enter a text to extract KG.")
    elif not selected_model:
        st.warning("Please select a model for KG extraction.")
    else:
        extractor = LLMTripletExtractor(model=selected_model)
        initial_triplets, refined_triplets, filtered_triplets = extract_triplets(input_text, sample_id=user_id, aligner=aligner, extractor=extractor)
        print("Initial triplets: ", initial_triplets)
        print()
        print("Refined triplets: ", refined_triplets)
        print()
        print("filtered_triplets: ", filtered_triplets)
        # insert_triplets_to_neo4j(refined_triplets)
        new_entities = {t["subject"] for t in refined_triplets} | {t["object"] for t in refined_triplets}
        subgraph = fetch_related_triplets(list(new_entities))
        # st.session_state.kg = nx.DiGraph()
        # for s, r, o in subgraph:
            # st.session_state.kg.add_edge(s, o, label=r, highlight=s in new_entities or o in new_entities)
        st.success(f"✅ Extracted {len(refined_triplets)} triplets and visualized {len(subgraph)} related ones.")
        

        col1, col2 = st.columns(2)
        
        with col1:
            # initial_kg_container = st.empty()
            # with initial_kg_container:
                st.subheader("Extracted Triplets")
                visualize_initial_knowledge_graph(initial_triplets['triplets'])

        with col2:
            # expanded_kg_container = st.empty()
            # with expanded_kg_container:
                st.subheader("Expanded KG Subgraph")
                visualize_knowledge_graph(subgraph, highlight_entities=new_entities)