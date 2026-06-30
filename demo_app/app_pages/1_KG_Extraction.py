import logging

import streamlit as st
from dotenv import load_dotenv

from streamlit_app_config import ENV_PATH, BACKEND_LANGUAGE, EXTRACTION_MODEL
from streamlit_examples import get_example_texts
from streamlit_i18n import t
from streamlit_inference import extract_triplets_for_demo
from streamlit_kg_viz import TRIPLET_FIELDS, visualize_knowledge_graph
from streamlit_session import get_inference, get_triplets_db, get_user_id
from streamlit_token_stats import compare_text_and_triplets
from streamlit_ui import render_footer, render_page_header
from wikontic.logging_config import get_logger

load_dotenv(ENV_PATH)
logger = get_logger("KGExtraction")

user_id = get_user_id()
triplets_db = get_triplets_db()
inference_with_db = get_inference()
logger.info("User ID: %s", user_id)


def fetch_related_triplets(entities):
    collection = triplets_db.get_collection("triplets")
    query = {
        "$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
        "sample_id": user_id,
    }
    results = collection.find(query, TRIPLET_FIELDS)
    return list(results)


render_page_header(t("extract.title"))

example_texts = get_example_texts(BACKEND_LANGUAGE)
custom_text_label = t("extract.custom_text")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "selected_predefined" not in st.session_state:
    st.session_state.selected_predefined = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(t("extract.examples_header"))
    predefined_options = [custom_text_label] + list(example_texts.keys())

    if st.session_state.selected_predefined in predefined_options:
        initial_index = predefined_options.index(st.session_state.selected_predefined)
    else:
        initial_index = 0

    selected_predefined = st.radio(
        t("extract.choose_text"),
        predefined_options,
        index=initial_index,
        key="predefined_selector",
    )

    if selected_predefined != st.session_state.selected_predefined:
        st.session_state.selected_predefined = selected_predefined
        if selected_predefined != custom_text_label and selected_predefined in example_texts:
            st.session_state.input_text = example_texts[selected_predefined]
            st.rerun()

with col2:
    st.subheader(t("extract.input_header"))
    input_text = st.text_area(
        t("extract.input_label"),
        placeholder=t("extract.input_placeholder"),
        height=300,
        key="input_text",
    )

trigger = st.button(t("extract.button"))

if trigger:
    if not input_text:
        st.warning(t("extract.empty_warning"))
    else:
        (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        ) = extract_triplets_for_demo(
            inference_with_db,
            text=input_text,
            sample_id=user_id,
            source_text_id=None,
        )
        logger.info("Initial triplets: %s", initial_triplets)
        logger.info("-" * 100)
        logger.info("Refined triplets: %s", final_triplets)
        logger.info("-" * 100)
        logger.info("filtered_triplets: %s", filtered_triplets)
        logger.info("-" * 100)
        logger.info("ontology_filtered_triplets: %s", ontology_filtered_triplets)
        logger.info("-" * 100)

        new_entities = {t_["subject"] for t_ in final_triplets} | {
            t_["object"] for t_ in final_triplets
        }
        subgraph = fetch_related_triplets(list(new_entities))
        st.success(
            "✅ "
            + t(
                "extract.success",
                final_count=len(final_triplets),
                subgraph_count=len(subgraph),
            )
        )

        token_stats = compare_text_and_triplets(
            input_text, final_triplets, model=EXTRACTION_MODEL
        )
        st.subheader(t("extract.token_header"))
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(t("extract.source_tokens"), f"{token_stats['source_tokens']:,}")
        metric_col2.metric(t("extract.triplets_count"), f"{len(final_triplets):,}")
        st.caption(t("extract.token_caption"))
        with st.expander(t("extract.triplet_text")):
            st.text(token_stats["triplet_text"])

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.subheader(t("extract.initial_facts"))
            initial_entities = {t_["subject"] for t_ in initial_triplets} | {
                t_["object"] for t_ in initial_triplets
            }
            visualize_knowledge_graph(
                initial_triplets,
                highlight_entities=initial_entities,
                highlight_color="#2fbeac",
                entity_color="#2fbeac",
            )

        with graph_col2:
            st.subheader(t("extract.enriched_graph"))
            visualize_knowledge_graph(
                subgraph,
                highlight_entities=new_entities,
                highlight_color="#2fbeac",
                entity_color="#C7C8CC",
            )

render_footer()
