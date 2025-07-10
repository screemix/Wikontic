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
    page_icon="media/wikontic.png",
    layout="wide"
)

with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Wikontic</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
Welcome to the **KG Demo App**.

Use the menu on the left to switch between:
- **KG Extraction**: Extract and visualize triples from text.
- **QA**: Ask questions based on the current knowledge graph.
""")