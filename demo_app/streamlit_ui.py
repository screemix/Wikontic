import base64

import streamlit as st

from streamlit_app_config import MEDIA_DIR
from streamlit_i18n import t

SIDEBAR_LOGO = MEDIA_DIR / "sidebar_logo.png"
HEADER_LOGO = MEDIA_DIR / "wikontic.png"
ACCENT_COLOR = "#2fbeac"


def show_sidebar_logo():
    st.logo(str(SIDEBAR_LOGO), icon_image=str(SIDEBAR_LOGO))
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


def _base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_page_header(title: str) -> None:
    encoded_logo = _base64_image(HEADER_LOGO)
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{encoded_logo}" width="50" style="margin-right: 15px;">
            <h1 style="margin: 0;">{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div style="padding: 20px 0; margin-top: 40px;
                    border-top: 1px solid #e0e0e0; text-align: center;">
            <div style="display: flex; justify-content: center; gap: 40px; align-items: center; flex-wrap: wrap;">
                <a href="https://github.com/screemix/Wikontic" target="_blank"
                    style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">🔗 {t('footer.github')}</a>
                <a href="https://arxiv.org/abs/2512.00590" target="_blank"
                    style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">📄 {t('footer.paper')}</a>
                <a href="https://github.com/screemix/Wikontic/blob/main/tutorial.ipynb" target="_blank"
                    style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">🦜 {t('footer.tutorial')}</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
