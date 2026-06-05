import streamlit as st
from streamlit_ui import show_sidebar_logo
from pyvis.network import Network
import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv

# from neo4j import GraphDatabase
from pymongo import MongoClient
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
import uuid
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("QA")
logger.setLevel(logging.ERROR)


# Ensure the same user_id across all pages
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id

logger.info(f"User ID: {user_id}")

_ = load_dotenv(find_dotenv())

WIKIDATA_ONTOLOGY_DB_NAME = "wikidata_ontology"
TRIPLETS_DB_NAME = "demo"
QA_MODEL = "gpt-4.1"
mongo_client = MongoClient(os.getenv("MONGO_URI"))
api_key = os.getenv("KEY")
proxy_url = os.getenv("PROXY_URL")
triplets_db = mongo_client.get_database(TRIPLETS_DB_NAME)
ontology_db = mongo_client.get_database(WIKIDATA_ONTOLOGY_DB_NAME)
aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)
show_sidebar_logo()


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

    for t in triplets:
        s, r, o = t["subject"], t["relation"], t["object"]
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


def query_kg(inferer, question_text):
    identified_entities = inferer.identify_relevant_entities_from_question_with_llm(
        question_text, sample_id=user_id
    )
    supporting_triplets, ans = inferer.answer_question_with_llm(
        question_text, identified_entities, sample_id=user_id
    )
    return identified_entities, supporting_triplets, ans


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


question = st.text_input("Ask a question about the Knowledge Graph")
trigger = st.button("Answer question")


if trigger:
    if not question:
        st.warning("Please enter a question.")
    else:
        extractor = LLMTripletExtractor(
            model=QA_MODEL, api_key=api_key, proxy=proxy_url
        )
        inferer = StructuredInferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )

        st.markdown(f"#### Results for: *{question}*")
        identified_entities_names, supporting_triplets, ans = query_kg(
            inferer, question
        )

        st.session_state.kg = nx.DiGraph()
        for t in supporting_triplets:
            s, r, o = t["subject"], t["relation"], t["object"]
            st.session_state.kg.add_edge(
                s,
                o,
                label=r,
                highlight=s in identified_entities_names
                or o in identified_entities_names,
            )

        st.success(f"✅ Extracted {len(supporting_triplets)} supporting triplets.")

        st.subheader("Relevant Subgraph")
        st.markdown(
            """
        - 🟢 <span style='color:#B2CD9C'>**Highlighted Entity**</span> – relevant node from your query  
        - ⚪ <span style='color:#C7C8CC'>**Regular Entity**</span> – node from KG  connected to one of the nodes from your query
        """,
            unsafe_allow_html=True,
        )
        visualize_knowledge_graph(
            supporting_triplets, highlight_entities=identified_entities_names
        )

        # st.success(f"✅ Answer to the question is {ans}")
        st.subheader("Answer")
        st.markdown(
            f"""
        <div style='background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745;'>
        ✅ Answer to the question is <strong>{ans}</strong>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div style="padding: 20px 0; margin-top: 40px; 
                border-top: 1px solid #e0e0e0; text-align: center;">
        <div style="display: flex; justify-content: center; gap: 40px; align-items: center; flex-wrap: wrap;">
            <a href="https://github.com/screemix/Wikontic" target="_blank"
                style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">🔗 GitHub Repository</a>
            <a href="https://arxiv.org/abs/2512.00590" target="_blank"
                style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">📄 ArXiv Paper</a>
            <a href="https://github.com/screemix/Wikontic/blob/main/tutorial.ipynb" target="_blank"
                style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">🦜 Langchain Tutorial</a>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)
