import streamlit as st
from dotenv import load_dotenv

from streamlit_app_config import ENV_PATH, USE_ONTOLOGY
from streamlit_i18n import t
from streamlit_kg_viz import TRIPLET_FIELDS, visualize_knowledge_graph
from streamlit_session import get_triplets_db, get_user_id
from streamlit_ui import render_footer, render_page_header
from wikontic.logging_config import get_logger

load_dotenv(ENV_PATH)
logger = get_logger("CurrentKG")

user_id = get_user_id()
triplets_db = get_triplets_db()
logger.info("User ID: %s", user_id)


def fetch_triplets():
    collection = triplets_db.get_collection("triplets")
    query = {"sample_id": user_id}
    results = collection.find(query, TRIPLET_FIELDS)
    return list(results)


def delete_session_graph():
    collection_names = [
        "triplets",
        "filtered_triplets",
        "initial_triplets",
        "entity_aliases",
    ]
    if USE_ONTOLOGY:
        collection_names.append("ontology_filtered_triplets")
    else:
        collection_names.append("property_aliases")

    for collection_name in collection_names:
        triplets_db.get_collection(collection_name).delete_many({"sample_id": user_id})


render_page_header(t("current.title"))

subgraph = fetch_triplets()
st.success("✅ " + t("current.success", count=len(subgraph)))
st.subheader(t("current.graph_header"))
visualize_knowledge_graph(subgraph, entity_color="#C7C8CC")

with st.expander("🗑 " + t("current.delete_expander"), expanded=True):
    st.markdown("⚠️ " + t("current.delete_warning"))
    confirm = st.checkbox(t("current.delete_confirm"))
    drop_button = st.button(t("current.delete_button"))
    if confirm and drop_button:
        delete_session_graph()
        st.success(t("current.delete_success"))
        logger.info("Knowledge graph deleted for user %s", user_id)
        st.stop()

render_footer()
