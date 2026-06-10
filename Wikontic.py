import streamlit as st
import uuid
import sys
import base64
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.wikontic.logging_config import get_logger

logger = get_logger("App")

# Ensure the same user_id across all pages
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id

logger.info(f"User ID: {user_id}")


st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)

with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded_logo = base64.b64encode(img_bytes).decode()

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_logo}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Wikontic</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with open("media/wikontic-example.png", "rb") as f:
    img_bytes = f.read()
encoded_pipeline = base64.b64encode(img_bytes).decode()

st.markdown(
    """
<span style="font-size: 1.2em;">
Welcome to the <b>Wikontic</b> &mdash; an integrated tool for Knowledge
Graph (KG) construction and question answering (QA).
</span>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

st.markdown(
    f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start; justify-content: center; gap: 40px; text-align: left;">
        <img src="data:image/png;base64,{encoded_pipeline}" style="max-width: 600px; height: auto; flex-shrink: 0;">
        <div style="max-width: 400px; margin-left: 8px; font-size: 1.1em; line-height: 1.6;">
            <p style="font-size: 1.1em; margin-bottom: 12px;">
            <b>Wikontic</b> is a tool for ontology-aware construction of a Wikidata-aligned knowledge graph.
            </p>
            <ul style="padding-left: 1.4em; margin: 0;">
            <li style="font-size: 1.1em; margin-bottom: 8px;">An LLM-based triplet extractor proposes candidate (subject, relation, object) triples; </li>
            <li style="font-size: 1.1em; margin-bottom: 8px;">The LLM, guided by the Wikidata ontology, assigns entity types (node colors), merges similar nodes, and prunes or rewrites relations that violate ontology constraints.</li>
            <li style="font-size: 1.1em; margin-bottom: 8px;">
                The resulting graph is de-duplicated and fully compliant with Wikidata semantics—ready for downstream tasks and further usage.
            </li>
            </ul>
        </div>
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
