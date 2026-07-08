import uuid

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

from streamlit_app_config import (
    ENV_PATH,
    API_KEY,
    BACKEND_LANGUAGE,
    BASE_URL,
    EXTRACTION_MODEL,
    INFERENCE_MODE,
    MONGO_URI,
    ONTOLOGY_DB_NAME,
    PROXY_URL,
    TRIPLETS_DB_NAME,
    USE_ONTOLOGY,
)
from streamlit_i18n import t
from wikontic.utils.language_config import prompt_folder_for_language
from wikontic.utils.inference_with_db import InferenceWithDB
from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.dynamic_aligner import Aligner as DynamicAligner
from wikontic.utils.structured_aligner import Aligner as StructuredAligner
from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB

_INIT_KEY = "wikontic_initialized"

TRIPLETS_COLLECTIONS = {
    "triplets",
    "initial_triplets",
    "filtered_triplets",
    "entity_aliases",
}
DYNAMIC_COLLECTIONS = TRIPLETS_COLLECTIONS | {"property_aliases"}
STRUCTURED_COLLECTIONS = TRIPLETS_COLLECTIONS | {"ontology_filtered_triplets"}
ONTOLOGY_COLLECTIONS = {
    "entity_types",
    "entity_type_aliases",
    "properties",
    "property_aliases",
}


def _format_missing_collections_message(db_name: str, missing: list[str]) -> str:
    return t(
        "config.missing_collections",
        db_name=db_name,
        collections=", ".join(f"`{name}`" for name in missing),
    )


def _triplets_init_command() -> str:
    module = (
        "wikontic.create_ontological_triplets_db"
        if USE_ONTOLOGY
        else "wikontic.create_triplets_db"
    )
    return (
        f'python -m {module} --backend mongodb --mongo_uri "{MONGO_URI}" '
        f"--db_name {TRIPLETS_DB_NAME}"
    )


def _ontology_init_command() -> str:
    return (
        'python -m wikontic.create_wikidata_ontology_db --backend mongodb '
        f'--mongo_uri "{MONGO_URI}" --database {ONTOLOGY_DB_NAME} '
        f"--language {BACKEND_LANGUAGE}"
    )


def _validate_collections(db, db_name: str, required: set[str], command: str) -> None:
    existing = set(db.list_collection_names())
    missing = sorted(required - existing)
    if not missing:
        return

    st.error(_format_missing_collections_message(db_name, missing))
    st.caption(t("config.init_command"))
    st.code(command, language="bash")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_shared_resources(
    mongo_uri,
    ontology_db_name,
    triplets_db_name,
    use_ontology,
    backend_language,
    extraction_model,
):
    mongo_client = MongoClient(mongo_uri)
    triplets_db = mongo_client.get_database(triplets_db_name)
    ontology_db = None

    if use_ontology:
        ontology_db = mongo_client.get_database(ontology_db_name)
        aligner = StructuredAligner(ontology_db=ontology_db, triplets_db=triplets_db)
    else:
        aligner = DynamicAligner(triplets_db=triplets_db)

    return mongo_client, ontology_db, triplets_db, aligner


def init_session() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    if st.session_state.get(_INIT_KEY):
        return

    _ = load_dotenv(ENV_PATH)
    mongo_client, ontology_db, triplets_db, aligner = get_shared_resources(
        MONGO_URI,
        ONTOLOGY_DB_NAME,
        TRIPLETS_DB_NAME,
        USE_ONTOLOGY,
        BACKEND_LANGUAGE,
        EXTRACTION_MODEL,
    )
    _validate_collections(
        triplets_db,
        TRIPLETS_DB_NAME,
        STRUCTURED_COLLECTIONS if USE_ONTOLOGY else DYNAMIC_COLLECTIONS,
        _triplets_init_command(),
    )
    if USE_ONTOLOGY:
        _validate_collections(
            ontology_db,
            ONTOLOGY_DB_NAME,
            ONTOLOGY_COLLECTIONS,
            _ontology_init_command(),
        )

    extractor = LLMTripletExtractor(
        model=EXTRACTION_MODEL,
        api_key=API_KEY,
        proxy=PROXY_URL,
        base_url=BASE_URL,
        prompt_folder_path=str(prompt_folder_for_language(BACKEND_LANGUAGE)),
    )
    if USE_ONTOLOGY:
        inference = StructuredInferenceWithDB(
            extractor=extractor,
            aligner=aligner,
            triplets_db=triplets_db,
            language=BACKEND_LANGUAGE,
        )
    else:
        inference = InferenceWithDB(
            extractor=extractor,
            aligner=aligner,
            triplets_db=triplets_db,
            language=BACKEND_LANGUAGE,
        )

    st.session_state.mongo_client = mongo_client
    st.session_state.ontology_db = ontology_db
    st.session_state.triplets_db = triplets_db
    st.session_state.aligner = aligner
    st.session_state.extractor = extractor
    st.session_state.inference = inference
    st.session_state.api_key = API_KEY
    st.session_state.base_url = BASE_URL
    st.session_state.proxy_url = PROXY_URL
    st.session_state.backend_language = BACKEND_LANGUAGE
    st.session_state.use_ontology = USE_ONTOLOGY
    st.session_state.inference_mode = INFERENCE_MODE
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


def get_inference():
    init_session()
    return st.session_state.inference


def get_base_url() -> str:
    init_session()
    return st.session_state.base_url


def get_backend_language() -> str:
    init_session()
    return st.session_state.backend_language


def get_use_ontology() -> bool:
    init_session()
    return st.session_state.use_ontology
