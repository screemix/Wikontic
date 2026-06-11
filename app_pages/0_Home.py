import base64
import logging
import sys

import streamlit as st

from streamlit_session import get_user_id

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("App")
logger.setLevel(logging.ERROR)

user_id = get_user_id()
logger.info(f"User ID: {user_id}")

with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded_logo = base64.b64encode(img_bytes).decode()

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_logo}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Wikontic</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with open("media/wikontic-example.png", "rb") as f:
    img_bytes = f.read()
encoded_pipeline = base64.b64encode(img_bytes).decode()

st.markdown(
    """
<span style="font-size: 1.2em;">
Добро пожаловать в <b>Wikontic</b> &mdash; 
интегрированный инструмент для построения графов знаний (KG)
и ответа на вопросы (QA).
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
            <b>Wikontic</b> предназначен для построения графов знаний, согласованных с онтологией Wikidata.
            </p>
            <ul style="padding-left: 1.4em; margin: 0;">
            <li style="font-size: 0.9em; margin-bottom: 8px;">LLM-модель извлекает кандидатные триплеты вида (сущность — отношение — сущность); </li>
            <li style="font-size: 0.9em; margin-bottom: 8px;">LLM c учётом онтологии Wikidata определяет типы сущностей (отражаемые цветами узлов), объединяет семантически близкие узлы, а также удаляет или переформулирует отношения, нарушающие онтологические правила;</li>
            <li style="font-size: 0.9em; margin-bottom: 8px;">
                В итоге получаемый граф не включает дубликаты и соответствует семантике Wikidata, что делает его готовым для последующего анализа и использования в прикладных задачах.
            </li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding: 20px 0; margin-top: 40px; 
                border-top: 1px solid #e0e0e0; text-align: center;">
        <div style="display: flex; justify-content: center; gap: 40px; align-items: center; flex-wrap: wrap;">
            <a href="https://github.com/screemix/Wikontic" target="_blank"
                style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">🔗 GitHub Repository</a>
            <a href="https://arxiv.org/abs/2512.00590" target="_blank"
                style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">📄 ArXiv Paper</a>
            <a href="https://github.com/screemix/Wikontic/blob/main/tutorial.ipynb" target="_blank"
                style="text-decoration: none; color: #1f77b4; font-size: 1.2em; font-weight: 500;">🦜 Langchain Tutorial</a>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)
