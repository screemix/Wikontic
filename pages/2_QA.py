import streamlit as st
from pyvis.network import Network
import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
# from neo4j import GraphDatabase
from pymongo import MongoClient
from utils.structured_dynamic_index_utils_with_db import Aligner
from utils.openai_utils import LLMTripletExtractor
from utils.structured_inference_with_db import identify_relevant_entities, answer_question
import uuid
import logging
import sys
import base64
from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger('QA')
logger.setLevel(logging.ERROR)


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


# --- Visualize ---
def visualize_knowledge_graph(triplets, highlight_entities=None):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    highlight_entities = highlight_entities or set()
    added_nodes = set()

    for t in triplets:
        s, r, o =  t['subject'], t['relation'], t['object']
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
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)

def query_kg(question_text):
    identified_entities = identify_relevant_entities(question_text, extractor=extractor, aligner=aligner, sample_id=user_id)
    identified_entities_names = [e['entity'] for e in identified_entities]
    supporting_triplets, ans = answer_question(question_text, identified_entities, extractor=extractor, aligner=aligner, db=db, sample_id=user_id)
    return identified_entities_names, supporting_triplets, ans

with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Question Answering with KG</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox("Choose a model for QA:", model_options, index=0)
question = st.text_input("Ask a question about the Knowledge Graph")
trigger = st.button("Answer question")


if trigger:
    if not question:
        st.warning("Please enter a question.")
    elif not selected_model:
        st.warning("Please select a model.")
    else:
        extractor = LLMTripletExtractor(model=selected_model)
        
        st.markdown(f"#### Results for: *{question}*")
        identified_entities_names, supporting_triplets, ans = query_kg(question)

        st.session_state.kg = nx.DiGraph()
        for t in supporting_triplets:
            s, r, o = t['subject'], t['relation'], t['object']
            st.session_state.kg.add_edge(s, o, label=r, highlight=s in identified_entities_names or o in identified_entities_names)

        st.success(f"✅ Extracted {len(supporting_triplets)} supporting triplets.")

        st.subheader("Relevant Subgraph")
        st.markdown("""
        - 🟢 <span style='color:#B2CD9C'>**Highlighted Entity**</span> – relevant node from your query  
        - ⚪ <span style='color:#C7C8CC'>**Regular Entity**</span> – node from KG  connected to one of the nodes from your query
        """, unsafe_allow_html=True)
        visualize_knowledge_graph(supporting_triplets, highlight_entities=identified_entities_names)

        # st.success(f"✅ Answer to the question is {ans}")
        st.subheader("Answer")
        st.markdown(f"""
        <div style='background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745;'>
        ✅ Answer to the question is <strong>{ans}</strong>
        </div>
        """, unsafe_allow_html=True)
