import streamlit as st
import uuid
import logging
import sys

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
    page_title="Knowledge Graph Demo",
    page_icon="🧠",
)

st.title("🧠 Knowledge Graph Demo")

st.markdown("""
Welcome to the **KG Demo App**.

Use the menu on the left to switch between:
- **KG Extraction**: Extract and visualize triples from text.
- **QA**: Ask questions based on the current knowledge graph.
""")