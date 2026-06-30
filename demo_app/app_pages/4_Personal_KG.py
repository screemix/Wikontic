import streamlit as st
from dotenv import load_dotenv

from streamlit_app_config import ENV_PATH, BACKEND_LANGUAGE, EXTRACTION_MODEL
from streamlit_examples import personal_search_prompt
from streamlit_i18n import t
from streamlit_inference import extract_triplets_for_demo
from streamlit_kg_viz import TRIPLET_FIELDS, visualize_knowledge_graph
from streamlit_session import get_base_url, get_extractor, get_inference, get_triplets_db
from streamlit_ui import render_footer, render_page_header
from wikontic.logging_config import get_logger

load_dotenv(ENV_PATH)
logger = get_logger("PersonalKG")

extractor = get_extractor()
inference_with_db = get_inference()
triplets_db = get_triplets_db()
base_url = get_base_url()


def supports_openai_web_search(api_base_url: str) -> bool:
    return "api.openai.com" in (api_base_url or "").lower()


def fetch_related_triplets(entities):
    collection = triplets_db.get_collection("triplets")
    query = {
        "$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
        "sample_id": "personal_kg",
    }
    results = collection.find(query, TRIPLET_FIELDS)
    return list(results)


render_page_header(t("personal.title"))

st.subheader(t("personal.subheader"))
input_text = st.text_area(
    t("personal.input_label"),
    placeholder=t("personal.input_placeholder"),
    height=68,
    key="name_surname",
)

trigger = st.button(t("personal.button"))

if trigger:
    if not input_text:
        st.warning(t("personal.empty_warning"))
    else:
        if not supports_openai_web_search(base_url):
            st.error(t("personal.unsupported_api"))
            st.stop()

        try:
            response = extractor.client.responses.create(
                model=EXTRACTION_MODEL,
                tools=[{"type": "web_search"}],
                input=personal_search_prompt(BACKEND_LANGUAGE, input_text),
            )
        except Exception as exc:
            logger.exception("Personal KG web search failed")
            st.error(t("personal.unsupported_api_details", details=exc))
            st.stop()

        personal_text = response.output_text

        logger.info("Personal text: %s", personal_text)
        (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        ) = extract_triplets_for_demo(
            inference_with_db,
            text=personal_text,
            sample_id="personal_kg",
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
                "personal.success",
                final_count=len(final_triplets),
                subgraph_count=len(subgraph),
            )
        )

        st.subheader(t("personal.graph_header"))
        visualize_knowledge_graph(
            subgraph,
            highlight_entities=new_entities,
            highlight_color="#2fbeac",
            entity_color="#C7C8CC",
        )

render_footer()
