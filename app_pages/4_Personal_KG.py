import streamlit as st
from streamlit_session import (
    EXTRACTION_MODEL,
    get_extractor,
    get_inference,
    get_triplets_db,
)
from streamlit_kg_viz import TRIPLET_FIELDS, visualize_knowledge_graph

# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.wikontic.logging_config import get_logger
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from pymongo import MongoClient
import uuid
import base64

logger = get_logger("PersonalKG")

extractor = get_extractor()
inference_with_db = get_inference()
triplets_db = get_triplets_db()


def fetch_related_triplets(entities):
    collection = triplets_db.get_collection("triplets")
    query = {
        "$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
        "sample_id": "personal_kg",
    }
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
        <h1 style="margin: 0;">Постройте свой персональный граф знаний!</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
st.session_state.input_text = ""

st.subheader("Введите имя и фамилию человека, для которого вы хотите построить граф знаний")
input_text = st.text_area(
    "Введите имя и фамилию:",
    value=st.session_state.input_text,
    placeholder="Введите имя и фамилию человека, для которого вы хотите построить граф знаний",
    height=68,
    key="name_surname",
)

trigger = st.button("Построить и визуализировать граф знаний для человека")

if trigger:
    if not input_text:
        st.warning(
            "Пожалуйста, введите имя и фамилию человека, для которого вы хотите построить граф знаний."
        )
    else:
        response = extractor.client.responses.create(
            model=EXTRACTION_MODEL,
            tools=[{"type": "web_search"}],
            input=f"Найдите и извлеките из интернета свежую и актуальную информацию о {input_text} и верните параграф, который суммирует эту информацию. Верните только параграф, никакого другого текста.",
        )
        personal_text = response.output_text

        logger.info(f"Personal text: {personal_text}")
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
            f"✅ Найдено {len(final_triplets)} триплетов и визуализировано {len(subgraph)} связанных."
        )

        st.subheader("Дополненный новыми триплетами граф знаний")
        visualize_knowledge_graph(
            subgraph,
            highlight_entities=new_entities,
            highlight_color="#2fbeac",
            entity_color="#C7C8CC",
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
