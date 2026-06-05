import streamlit as st

SIDEBAR_LOGO = "media/Logo.png"
ACCENT_COLOR = "#2fbeac"


def show_sidebar_logo():
    st.logo(SIDEBAR_LOGO)
    st.markdown(
        f"""
        <style>
            [data-testid="stDecoration"],
            #stDecoration,
            .stDecoration {{
                display: block !important;
                background: {ACCENT_COLOR} !important;
                background-color: {ACCENT_COLOR} !important;
                background-image: none !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
