import streamlit as st
from streamlit_session import get_inference, get_user_id
from streamlit_kg_viz import visualize_knowledge_graph
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("QA")
logger.setLevel(logging.ERROR)

user_id = get_user_id()
inference = get_inference()
logger.info(f"User ID: {user_id}")


def query_kg(inferer, question_text):
    identified_entities = inferer.identify_relevant_entities_from_question_with_llm(
        question_text, sample_id=user_id
    )
    supporting_triplets, ans = inferer.answer_question_with_llm(
        question_text,
        identified_entities,
        sample_id=user_id,
        use_qualifiers=True,
    )
    return identified_entities, supporting_triplets, ans


with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Поиск ответа на вопрос по графу знаний</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


question = st.text_input("Введите вопрос:")
trigger = st.button("Ответить на вопрос")


if trigger:
    if not question:
        st.warning("Пожалуйста, введите вопрос.")
    else:
        st.markdown(f"#### Результат для вопроса: *{question}*")
        identified_entities_names, supporting_triplets, ans = query_kg(
            inference, question
        )

        st.success(f"✅ Найдено {len(supporting_triplets)} релевантных триплетов.")

        st.subheader("Релевантный граф знаний")
        st.markdown(
            """
        - 🟢 <span style='color:#B2CD9C'>**Выделенная сущность**</span> – релевантная сущность из вопроса  
        - ⚪ <span style='color:#C7C8CC'>**Невыделенная сущность**</span> – сущность из графа знаний, связанная с одной из сущностей из вопроса
        """,
            unsafe_allow_html=True,
        )
        visualize_knowledge_graph(
            supporting_triplets,
            highlight_entities=set(identified_entities_names),
            highlight_color="#2fbeac",
            entity_color="#C7C8CC",
        )

        # st.success(f"✅ Answer to the question is {ans}")
        st.subheader("Ответ")
        st.markdown(
            f"""
        <div style='background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745;'>
        ✅ <strong>{ans}</strong>
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
