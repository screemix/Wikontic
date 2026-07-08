import base64
import logging
import sys

import streamlit as st

from streamlit_app_config import MEDIA_DIR
from streamlit_i18n import t
from streamlit_session import get_user_id
from streamlit_ui import render_footer, render_page_header

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("App")
logger.setLevel(logging.ERROR)

user_id = get_user_id()
logger.info("User ID: %s", user_id)

render_page_header(t("home.title"))

with open(MEDIA_DIR / "wikontic-example.png", "rb") as f:
    encoded_pipeline = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
<span style="font-size: 1.2em;">
{t("home.welcome")}
</span>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

st.markdown(
    f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start; justify-content: center; gap: 32px; text-align: left;">
        <img src="data:image/png;base64,{encoded_pipeline}" style="flex: 0 1 42%; max-width: 460px; height: auto;">
        <div style="flex: 1 1 58%; max-width: 560px; margin-left: 8px; font-size: 1.1em; line-height: 1.6;">
            <p style="font-size: 1em; margin-bottom: 12px;">
            {t("home.description")}
            </p>
            <ul style="padding-left: 1.4em; margin: 0;">
            <li style="font-size: 0.9em; margin-bottom: 8px;">{t("home.bullet.extract")}</li>
            <li style="font-size: 0.9em; margin-bottom: 8px;">{t("home.bullet.refine")}</li>
            <li style="font-size: 0.9em; margin-bottom: 8px;">{t("home.bullet.result")}</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_footer()
