# --- File: 0_KG_Extraction.py ---
import streamlit as st
from streamlit_session import (
    USE_UNIDECODE,
    get_inference,
    get_triplets_db,
    get_user_id,
)
from streamlit_kg_viz import TRIPLET_FIELDS, visualize_knowledge_graph
from streamlit_session import EXTRACTION_MODEL
from streamlit_token_stats import compare_text_and_triplets

# import networkx as nx
<<<<<<< HEAD
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
=======
import logging
import sys
>>>>>>> demo
import base64

logger = get_logger("KGExtraction")

user_id = get_user_id()
triplets_db = get_triplets_db()
inference_with_db = get_inference()
logger.info(f"User ID: {user_id}")


def fetch_related_triplets(entities):
    collection = triplets_db.get_collection("triplets")
    query = {
        "$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
        "sample_id": user_id,
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
        <h1 style="margin: 0;">Извлечь и визуализировать граф знаний</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Predefined Wikipedia texts
WIKIPEDIA_TEXTS = {
    "Юрий Гагарин": "Юрий Алексеевич Гагарин (9 марта 1934, Клушино — 27 марта 1968, село Новосёлово, Владимирская область) — советский космонавт и военный лётчик, первый человек, совершивший космический полёт. Герой Советского Союза, кавалер высших знаков отличия ряда государств, почётный гражданин многих российских и зарубежных городов.",
    "Алексей Леонов": "Алексей Архипович Леонов (30 мая 1934, Листвянка, Западно-Сибирский край — 11 октября 2019, Басманный район, Москва) — лётчик-космонавт СССР № 11, первый человек в мире, вышедший в открытый космос. Дважды Герой Советского Союза (1965, 1975), генерал-майор авиации (1975), лауреат Государственной премии СССР (1981), член Высшего совета партии «Единая Россия» (2002—2019), военный лётчик 1-го класса (1965).",
    


}

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "selected_predefined" not in st.session_state:
    st.session_state.selected_predefined = None

# Create two columns: left for predefined texts, right for text area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Примеры текстов")

    # Add option for custom text
    predefined_options = ["Произвольный текст"] + list(WIKIPEDIA_TEXTS.keys())

    # Determine initial index
    if st.session_state.selected_predefined is None:
        initial_index = 0
    elif st.session_state.selected_predefined in predefined_options:
        initial_index = predefined_options.index(st.session_state.selected_predefined)
    else:
        initial_index = 0

    selected_predefined = st.radio(
        "Выберите текст:",
        predefined_options,
        index=initial_index,
        key="predefined_selector",
    )

    # Handle selection change
    if selected_predefined != st.session_state.selected_predefined:
        st.session_state.selected_predefined = selected_predefined
        if (
            selected_predefined != "Произвольный текст"
            and selected_predefined in WIKIPEDIA_TEXTS
        ):
            st.session_state.input_text = WIKIPEDIA_TEXTS[selected_predefined]
            st.rerun()
        elif selected_predefined == "Произвольный текст":
            # Don't clear text when switching to custom - let user keep their edits
            pass

with col2:
    st.subheader("Входной текст")
    input_text = st.text_area(
        "Введите текст:",
        value=st.session_state.input_text,
        placeholder="Введите текст или выберите текст из списка слева...",
        height=300,
        key="text_area",
    )
    # Update session state when user manually edits
    st.session_state.input_text = input_text

trigger = st.button("Извлечь и визуализировать граф знаний")

if trigger:
    if not input_text:
        st.warning("Пожалуйста, введите текст для извлечения графа знаний.")
    else:
        (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        ) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
            text=input_text, sample_id=user_id, source_text_id=None, use_unidecode=USE_UNIDECODE
        )
        logger.info("Initial triplets: ", initial_triplets)
        logger.info("-" * 100)
        logger.info("Refined triplets: ", final_triplets)
        logger.info("-" * 100)
        logger.info("filtered_triplets: ", filtered_triplets)
        logger.info("-" * 100)
        logger.info("ontology_filtered_triplets: ", ontology_filtered_triplets)
        logger.info("-" * 100)
        new_entities = {t["subject"] for t in final_triplets} | {
            t["object"] for t in final_triplets
        }
        subgraph = fetch_related_triplets(list(new_entities))
        st.success(
            f"✅ Extracted {len(final_triplets)} triplets and visualized {len(subgraph)} related ones."
        )

        token_stats = compare_text_and_triplets(
            input_text, final_triplets, model=EXTRACTION_MODEL
        )
        st.subheader("Счётчик токенов")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Исходный текст", f"{token_stats['source_tokens']:,}")
        metric_col2.metric(
            "Факты (триплеты)",
            f"{token_stats['triplet_tokens']:,}",
        )
        metric_col3.metric(
            "Экономия",
            f"{token_stats['savings_pct']:.1f}%",
            delta=f"{token_stats['triplet_tokens'] - token_stats['source_tokens']:,}",
            delta_color="inverse",
        )
        st.caption(
            "Токены для триплетов считаются по вербализации вида "
            "`(subject, relation, object | [qualifiers])`."
        )
        with st.expander("Вербализация триплетов"):
            st.text(token_stats["triplet_text"])

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Извлеченные факты")
            initial_entities = {t["subject"] for t in initial_triplets} | {
                t["object"] for t in initial_triplets
            }
            visualize_knowledge_graph(
                initial_triplets,
                highlight_entities=initial_entities,
                highlight_color="#2fbeac",
                entity_color="#2fbeac",
            )

        with col2:
            st.subheader("Дополненный фактами граф знаний")
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
