# --- File: 0_KG_Extraction.py ---
import streamlit as st
from streamlit_ui import show_sidebar_logo
from streamlit_session import get_triplets_db, get_user_id, init_session
from pyvis.network import Network

# import networkx as nx
import tempfile
import os
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("KGExtraction")
logger.setLevel(logging.INFO)

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)
show_sidebar_logo()
init_session()

user_id = get_user_id()
triplets_db = get_triplets_db()
logger.info(f"User ID: {user_id}")


def fetch_triplets():
    collection = triplets_db.get_collection("triplets")
    query = {"sample_id": user_id}
    results = collection.find(
        query, {"_id": 0, "subject": 1, "relation": 1, "object": 1}
    )
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


# --- Visualize ---
def visualize_knowledge_graph(
    triplets,
):
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )
    added_nodes = set()

    for s, r, o in triplets:
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(node, label=node, color="#C7C8CC")
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


# --- UI ---
with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">KG Viewer</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


subgraph = fetch_triplets()
# st.session_state.kg = nx.DiGraph()
# for s, r, o in subgraph:
# st.session_state.kg.add_edge(s, o, label=r, highlight=s in new_entities or o in new_entities)
st.success(f"✅ Retrieved {len(subgraph)} triplets.")
st.subheader("Current Knowledge Graph")
visualize_knowledge_graph(subgraph)

with st.expander("🗑 Drop Knowledge Graph", expanded=True):
    st.markdown(
        """⚠️ This action button will drop the knowledge graph built in current session."""
    )
    confirm = st.checkbox("Confirm Drop")
    drop_button = st.button("Drop")
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

        st.success("Knowledge Graph dropped.")
        logger.info(f"Knowledge Graph dropped for user {user_id}")
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