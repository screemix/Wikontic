import streamlit as st
from dotenv import load_dotenv

from streamlit_app_config import ENV_PATH
from streamlit_i18n import t
from streamlit_kg_viz import visualize_knowledge_graph
from streamlit_session import get_inference, get_user_id
from streamlit_ui import render_footer, render_page_header
from wikontic.logging_config import get_logger

load_dotenv(ENV_PATH)
logger = get_logger("QA")

user_id = get_user_id()
inference = get_inference()
logger.info("User ID: %s", user_id)


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


render_page_header(t("qa.title"))

question = st.text_input(t("qa.input_label"))
trigger = st.button(t("qa.button"))

if trigger:
    if not question:
        st.warning(t("qa.empty_warning"))
    else:
        st.markdown("#### " + t("qa.result", question=question))
        identified_entities_names, supporting_triplets, ans = query_kg(
            inference, question
        )

        st.success("✅ " + t("qa.success", count=len(supporting_triplets)))

        st.subheader(t("qa.graph_header"))
        st.markdown(t("qa.legend"), unsafe_allow_html=True)
        visualize_knowledge_graph(
            supporting_triplets,
            highlight_entities=set(identified_entities_names),
            highlight_color="#2fbeac",
            entity_color="#C7C8CC",
        )

        st.subheader(t("qa.answer_header"))
        st.markdown(
            f"""
        <div style='background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745;'>
        ✅ <strong>{ans}</strong>
        </div>
        """,
            unsafe_allow_html=True,
        )

render_footer()
