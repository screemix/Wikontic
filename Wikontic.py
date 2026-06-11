import streamlit as st
import uuid
import sys
import base64
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.wikontic.logging_config import get_logger

logger = get_logger("App")

st.set_page_config(
    page_title="Wikontic", page_icon="media/wikotic-wo-text.png", layout="wide"
)
show_sidebar_logo()
init_session()
run_app()
