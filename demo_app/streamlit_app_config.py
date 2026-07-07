from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DEMO_APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_APP_DIR.parent
MEDIA_DIR = REPO_ROOT / "media"
ENV_PATH = REPO_ROOT / ".env"

load_dotenv(ENV_PATH if ENV_PATH.exists() else find_dotenv())

SUPPORTED_LANGUAGES = {"en", "ru"}
DEFAULT_MONGO_URI = "mongodb://localhost:27018/?directConnection=true"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _read_language(name: str, default: str = "en") -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in SUPPORTED_LANGUAGES else default


def _read_bool(name: str, default: bool = True) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


FRONTEND_LANGUAGE = _read_language("WIKONTIC_FRONTEND_LANGUAGE", "en")
BACKEND_LANGUAGE = _read_language("WIKONTIC_BACKEND_LANGUAGE", "en")
USE_ONTOLOGY = _read_bool("WIKONTIC_USE_ONTOLOGY", True)

MONGO_URI = os.getenv("MONGO_URI", DEFAULT_MONGO_URI)
API_KEY = os.getenv("OPENROUTER_KEY") or os.getenv("KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
PROXY_URL = os.getenv("PROXY_URL")
EXTRACTION_MODEL = os.getenv("WIKONTIC_MODEL", "gpt-4.1")


def _default_ontology_db_name(language: str) -> str:
    return "wikidata_ontology_ru" if language == "ru" else "wikidata_ontology"


def _default_triplets_db_name(language: str, use_ontology: bool) -> str:
    if use_ontology:
        return "demo_ru" if language == "ru" else "demo"
    return "demo_ru_dynamic" if language == "ru" else "demo_dynamic"


ONTOLOGY_DB_NAME = os.getenv(
    "WIKONTIC_ONTOLOGY_DB_NAME", _default_ontology_db_name(BACKEND_LANGUAGE)
)
TRIPLETS_DB_NAME = os.getenv(
    "WIKONTIC_TRIPLETS_DB_NAME",
    _default_triplets_db_name(BACKEND_LANGUAGE, USE_ONTOLOGY),
)

INFERENCE_MODE = "ontology" if USE_ONTOLOGY else "dynamic"
