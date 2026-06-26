import os
import uuid

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

from src.wikontic.utils.language_config import prompt_folder_for_language
from src.wikontic.utils.openai_utils import LLMTripletExtractor
from src.wikontic.utils.structured_aligner import Aligner
from src.wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB

LANGUAGE = "ru"
WIKIDATA_ONTOLOGY_DB_NAME = "wikidata_ontology_ru"
TRIPLETS_DB_NAME = "demo_ru"
EXTRACTION_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"

_INIT_KEY = "wikontic_initialized"


@st.cache_resource(show_spinner=False)
def get_shared_resources(mongo_uri, ontology_db_name, triplets_db_name):
    mongo_client = MongoClient(mongo_uri)
    ontology_db = mongo_client.get_database(ontology_db_name)
    triplets_db = mongo_client.get_database(triplets_db_name)
    aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)
    return mongo_client, ontology_db, triplets_db, aligner


def init_session() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    if st.session_state.get(_INIT_KEY):
        return

    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("OPENROUTER_KEY") or os.getenv("KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
    proxy_url = os.getenv("PROXY_URL")
    mongo_client, ontology_db, triplets_db, aligner = get_shared_resources(
        os.getenv("MONGO_URI"),
        WIKIDATA_ONTOLOGY_DB_NAME,
        TRIPLETS_DB_NAME,
    )
    extractor = LLMTripletExtractor(
        model=EXTRACTION_MODEL,
        api_key=api_key,
        proxy=proxy_url,
        base_url=base_url,
        prompt_folder_path=str(prompt_folder_for_language(LANGUAGE)),
    )
    inference = StructuredInferenceWithDB(
        extractor=extractor,
        aligner=aligner,
        triplets_db=triplets_db,
        language=LANGUAGE,
    )

    st.session_state.mongo_client = mongo_client
    st.session_state.ontology_db = ontology_db
    st.session_state.triplets_db = triplets_db
    st.session_state.aligner = aligner
    st.session_state.extractor = extractor
    st.session_state.inference = inference
    st.session_state.api_key = api_key
    st.session_state.base_url = base_url
    st.session_state.proxy_url = proxy_url
    st.session_state[_INIT_KEY] = True


def get_user_id() -> str:
    init_session()
    return st.session_state.user_id


def get_triplets_db():
    init_session()
    return st.session_state.triplets_db


def get_aligner():
    init_session()
    return st.session_state.aligner


def get_extractor() -> LLMTripletExtractor:
    init_session()
    return st.session_state.extractor


def get_inference() -> StructuredInferenceWithDB:
    init_session()
    return st.session_state.inference


def get_base_url() -> str:
    init_session()
    return st.session_state.base_url
