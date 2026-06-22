import streamlit as st
from dotenv import load_dotenv, find_dotenv

from streamlit_navigation import run_app
from streamlit_session import init_session
from streamlit_ui import show_sidebar_logo
from src.wikontic.logging_config import get_logger

load_dotenv(find_dotenv())

logger = get_logger("App")

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)
show_sidebar_logo()
init_session()
run_app()
