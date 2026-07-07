import streamlit as st

st.set_page_config(layout="wide")

from streamlit_ui import show_sidebar_logo
from streamlit_session import init_session
from streamlit_navigation import run_app

init_session()
show_sidebar_logo()
run_app()
