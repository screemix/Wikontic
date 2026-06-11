# --- File: 0_KG_Extraction.py ---
import streamlit as st
from streamlit_session import get_triplets_db, get_user_id
from streamlit_kg_viz import TRIPLET_FIELDS, visualize_knowledge_graph

# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from pymongo import MongoClient
from src.wikontic.logging_config import get_logger
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
import uuid
import base64

logger = get_logger("KGExtraction")

user_id = get_user_id()
triplets_db = get_triplets_db()
logger.info(f"User ID: {user_id}")


def fetch_triplets():
    collection = triplets_db.get_collection("triplets")
    query = {"sample_id": user_id}
    results = collection.find(query, TRIPLET_FIELDS)
    return list(results)


# --- UI ---
with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Просмотр текущего графа знаний</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


subgraph = fetch_triplets()
# st.session_state.kg = nx.DiGraph()
# for s, r, o in subgraph:
# st.session_state.kg.add_edge(s, o, label=r, highlight=s in new_entities or o in new_entities)
st.success(f"✅ Найдено {len(subgraph)} триплетов.")
st.subheader("Текущий граф знаний")
visualize_knowledge_graph(subgraph, entity_color="#C7C8CC")

with st.expander("🗑 Удалить граф знаний", expanded=True):
    st.markdown(
        """⚠️ Это действие удалит граф знаний, построенный в текущей сессии."""
    )
    confirm = st.checkbox("Подтвердить удаление")
    drop_button = st.button("Удалить")
    if confirm and drop_button:
        collection = triplets_db.get_collection("triplets")
        collection.delete_many({"sample_id": user_id})
        collection = triplets_db.get_collection("filtered_triplets")
        collection.delete_many({"sample_id": user_id})
        collection = triplets_db.get_collection("ontology_filtered_triplets")
        collection.delete_many({"sample_id": user_id})
        collection = triplets_db.get_collection("initial_triplets")
        collection.delete_many({"sample_id": user_id})
        collection = triplets_db.get_collection("entity_aliases")
        collection.delete_many({"sample_id": user_id})

        st.success("Граф знаний удален.")
        logger.info(f"Граф знаний удален для пользователя {user_id}")
        st.stop()

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