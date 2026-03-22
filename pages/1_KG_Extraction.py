# --- File: 0_KG_Extraction.py ---
import streamlit as st
from pyvis.network import Network

# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from pymongo import MongoClient
import uuid
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("KGExtraction")
logger.setLevel(logging.INFO)


# Ensure the same user_id across all pages
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id
logger.info(f"User ID: {user_id}")

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)

WIKIDATA_ONTOLOGY_DB_NAME = "wikidata_ontology"
TRIPLETS_DB_NAME = "demo"
# --- Mongo Setup ---
_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
api_key = os.getenv("KEY")
proxy_url = os.getenv("PROXY_URL")
ontology_db = mongo_client.get_database(WIKIDATA_ONTOLOGY_DB_NAME)
triplets_db = mongo_client.get_database(TRIPLETS_DB_NAME)


aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)


def fetch_related_triplets(entities):
    collection = triplets_db.get_collection("triplets")
    query = {
        "$or": [{"subject": {"$in": entities}}, {"object": {"$in": entities}}],
        "sample_id": user_id,
    }
    results = collection.find(
        query, {"_id": 0, "subject": 1, "relation": 1, "object": 1}
    )
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]


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

    for s, r, o in triplets:
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


def visualize_initial_knowledge_graph(initial_triplets):
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )

    for t in initial_triplets:
        s, r, o = t["subject"], t["relation"], t["object"]
        logger.info(f"Initial triplet: {s} {r} {o}")
        net.add_node(s, label=s, color="#B2CD9C")
        net.add_node(o, label=o, color="#B2CD9C")
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
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
        <h1 style="margin: 0;">KG Extraction + Visualization</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
selected_model = st.selectbox(
    "Choose a model for KG extraction:", model_options, index=0
)

# Predefined Wikipedia texts
WIKIPEDIA_TEXTS = {
    "Albert Einstein": "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics. His mass–energy equivalence formula E = mc², which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.",
    "AAAI": "The Association for the Advancement of Artificial Intelligence (AAAI) is an international scientific society devoted to promote research in, and responsible use of, artificial intelligence (AI). AAAI also aims to increase public understanding of AI, improve the teaching and training of AI practitioners, and provide guidance for research planners and funders concerning the importance and potential of current AI developments and future directions.",
    "Singapore": "Singapore, officially the Republic of Singapore, is a sovereign country as well as a city-state. It is nicknamed as 'The Lion City', 'The Garden City' or 'The Little Red Dot'. It is an island state at the southern end of the Malay Peninsula in Southeast Asia, between the Straits of Malacca and the South China Sea. Singapore is about one degree of latitude (137 kilometres or 85 miles) north of the equator. About 5.70 million people live in Singapore. About 3.31 million are citizens. Most of them are ethnically Chinese, Malay, or Indian, as well as a smaller number of other Asians and Europeans.",
    "AAAI-2026": "In 2026, the AAAI Conference on Artificial Intelligence was held in Singapore, bringing together researchers and practitioners from academia, industry, and government. The conference featured peer-reviewed technical papers, invited talks, workshops, tutorials, and poster sessions covering a broad range of topics in artificial intelligence. Singapore served as the host location for the event, providing conference facilities and infrastructure to support international participation. The 2026 edition continued AAAI’s annual conference series and contributed to the dissemination of current research results and ongoing developments in the field of artificial intelligence.",
    "TP53": "p53, also known as tumor protein p53 (TP53), is a regulatory transcription factor protein that is often mutated in human cancers. p53 has been described as 'the guardian of the genome' because of its role in conserving stability by preventing genome mutation. Hence TP53 is classified as a tumor suppressor gene. The TP53 gene is the most frequently mutated gene (>50%) in human cancer, indicating that the TP53 gene plays a crucial role in preventing cancer formation. TP53 gene encodes proteins that bind to DNA and regulate gene expression to prevent mutations of the genome.",
    "p21": "p21Cip1 (alternatively p21Waf1), also known as cyclin-dependent kinase inhibitor 1 or CDK-interacting protein 1, is a cyclin-dependent kinase inhibitor (CKI) that is capable of inhibiting all cyclin/CDK complexes. p21 represents a major target of p53 activity and thus is associated with linking DNA damage to cell cycle arrest. This protein is encoded by the CDKN1A gene located on chromosome 6 (6p21.2) in humans.",
}

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "selected_predefined" not in st.session_state:
    st.session_state.selected_predefined = None

# Create two columns: left for predefined texts, right for text area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Text Examples")

    # Add option for custom text
    predefined_options = ["Custom Text"] + list(WIKIPEDIA_TEXTS.keys())

    # Determine initial index
    if st.session_state.selected_predefined is None:
        initial_index = 0
    elif st.session_state.selected_predefined in predefined_options:
        initial_index = predefined_options.index(st.session_state.selected_predefined)
    else:
        initial_index = 0

    selected_predefined = st.radio(
        "Choose a text option:",
        predefined_options,
        index=initial_index,
        key="predefined_selector",
    )

    # Handle selection change
    if selected_predefined != st.session_state.selected_predefined:
        st.session_state.selected_predefined = selected_predefined
        if (
            selected_predefined != "Custom Text"
            and selected_predefined in WIKIPEDIA_TEXTS
        ):
            st.session_state.input_text = WIKIPEDIA_TEXTS[selected_predefined]
            st.rerun()
        elif selected_predefined == "Custom Text":
            # Don't clear text when switching to custom - let user keep their edits
            pass

with col2:
    st.subheader("Text Input")
    input_text = st.text_area(
        "Enter or modify text:",
        value=st.session_state.input_text,
        placeholder="Paste your text here or select a text option from the left...",
        height=300,
        key="text_area",
    )
    # Update session state when user manually edits
    st.session_state.input_text = input_text

trigger = st.button("Extract and Visualize")

if trigger:
    if not input_text:
        st.warning("Please enter a text to extract KG.")
    elif not selected_model:
        st.warning("Please select a model for KG extraction.")
    else:
        extractor = LLMTripletExtractor(
            model=selected_model, api_key=api_key, proxy=proxy_url
        )
        inference_with_db = StructuredInferenceWithDB(
            extractor=extractor, aligner=aligner, triplets_db=triplets_db
        )
        (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        ) = inference_with_db.extract_triplets_with_ontology_filtering_and_add_to_db(
            text=input_text, sample_id=user_id, source_text_id=None
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

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Extracted Triplets")
            visualize_initial_knowledge_graph(initial_triplets)

        with col2:
            st.subheader("Expanded KG Subgraph")
            visualize_knowledge_graph(subgraph, highlight_entities=new_entities)

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