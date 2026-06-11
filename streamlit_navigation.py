import streamlit as st

# Edit titles here without renaming page scripts.
PAGES = [
    st.Page("pages/0_Home.py", title="Главная", icon="🏠", default=True),
    st.Page("pages/1_KG_Extraction.py", title="Извлечь граф знаний"),
    st.Page("pages/2_QA.py", title="Вопрос по графу знаний"),
    st.Page("pages/3_Current_KG.py", title="Текущий граф"),
    st.Page("pages/4_Personal_KG.py", title="Персональный граф"),
]


def run_app() -> None:
    pg = st.navigation(PAGES)
    pg.run()
