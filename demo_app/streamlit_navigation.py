from pathlib import Path

import streamlit as st

from streamlit_i18n import t

PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

# Edit page keys here without renaming page scripts.
PAGES = [
    st.Page(str(PAGES_DIR / "0_Home.py"), title=t("nav.home"), icon="🏠", default=True),
    st.Page(str(PAGES_DIR / "1_KG_Extraction.py"), title=t("nav.extract")),
    st.Page(str(PAGES_DIR / "2_QA.py"), title=t("nav.qa")),
    st.Page(str(PAGES_DIR / "3_Current_KG.py"), title=t("nav.current")),
    st.Page(str(PAGES_DIR / "4_Personal_KG.py"), title=t("nav.personal")),
]


def run_app() -> None:
    pg = st.navigation(PAGES)
    pg.run()
