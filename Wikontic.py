import streamlit as st
import uuid
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger('App')
logger.setLevel(logging.ERROR)

# Ensure the same user_id across all pages
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id

logger.info(f"User ID: {user_id}")

st.set_page_config(
    page_title="Wikontic",
    page_icon="media/wikotic-wo-text.png",
    layout="wide"
)

with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded_logo = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_logo}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Wikontic</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with open("media/wikontic_pipeline_wo_logo.png", "rb") as f:
    img_bytes = f.read()
encoded_pipeline = base64.b64encode(img_bytes).decode()

st.markdown("""
Welcome to the **Wikontic** - an integrated tool for Knowledge
Graph (KG) construction and question answering (QA).
""")

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_pipeline}" style="margin-right: 15px;">
    </div>

    <p><small>Wikontic: a tool for ontology-aware construction of a Wikidata-aligned knowledge graphs. First, an LLM-based triplet extractor proposes candidate (subject, relation, object) triples (center, grey). Next, the LLM guided by the Wikidata ontology, assigns entity types (node colors), merges similar nodes, and prunes or rewrites relations that violate ontology constraints. The resulting graph (right) is de-duplicated and fully compliant with the semantics of Wikidata, ready for downstream tasks and further usage.</small></p>
    """,
    unsafe_allow_html=True,
)

st.markdown("""

            
Use the menu on the left to switch between:
- **KG Extraction**: Extract and visualize triples from text.
- **QA**: Ask questions based on the current knowledge graph.
- **Current KG**: View the KG you built
""")