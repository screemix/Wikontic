import streamlit as st
from dotenv import load_dotenv

from streamlit_app_config import ENV_PATH, MEDIA_DIR
from streamlit_navigation import run_app
from streamlit_session import init_session
from streamlit_ui import show_sidebar_logo
from wikontic.logging_config import get_logger

load_dotenv(ENV_PATH)

logger = get_logger("App")

st.set_page_config(
    page_title="Wikontic", page_icon=str(MEDIA_DIR / "wikotic-wo-text.png"), layout="wide"
)
show_sidebar_logo()
init_session()
run_app()