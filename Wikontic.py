import logging
import sys

import streamlit as st

from streamlit_navigation import run_app
from streamlit_session import init_session
from streamlit_ui import show_sidebar_logo

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("App")
logger.setLevel(logging.ERROR)

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)
show_sidebar_logo()
init_session()
run_app()
