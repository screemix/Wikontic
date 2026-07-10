"""
Shared pytest fixtures for the Wikontic test suite.
All config / API keys are read from .env in the repository root.
"""

import os
import sys
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env", override=True)
sys.path.insert(0, str(REPO_ROOT / "src"))

from wikontic.logging_config import configure_logging, resolve_log_level_name

configure_logging()


def pytest_configure(config):
    config.option.log_cli_level = resolve_log_level_name()

from pymongo.mongo_client import MongoClient

from wikontic.db.factory import create_backend
from wikontic.create_triplets_db import create_triplets_database
from wikontic.create_ontological_triplets_db import create_ontological_triplets_database
from wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database
from wikontic.utils.embedding_model import EMBEDDING_DIMS
from wikontic.utils.language_config import default_embedding_model_for_language

# ── config ─────────────────────────────────────────────────────────────────────
MONGO_URI        = os.environ.get("MONGO_URI", "mongodb://localhost:27018/?directConnection=true")
OPENROUTER_KEY   = os.environ.get("OPENROUTER_KEY") or os.environ.get("KEY")
OPENROUTER_URL   = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
PROXY_URL        = os.environ.get("PROXY_URL")
LLM_MODEL        = "gpt-4o-mini"
DEVICE           = "cpu"

TRIPLETS_DB      = "test_triplets_db"
ONTO_TRIPLETS_DB = "test_onto_triplets_db"

SAMPLE_TEXTS = [
    "Marie Curie was born in Warsaw, Poland, and later moved to Paris where she conducted "
    "her research at the University of Paris. She discovered polonium and radium.",
    "Albert Einstein developed the theory of relativity while working at the Swiss Patent "
    "Office in Bern. He was awarded the Nobel Prize in Physics in 1921.",
    "The Eiffel Tower was designed by Gustave Eiffel and constructed between 1887 and 1889 "
    "in Paris, France. It served as the entrance arch to the 1889 World's Fair.",
]

# ── ru config ──────────────────────────────────────────────────────────────────
LANGUAGE_RU        = "ru"
EMBEDDING_MODEL_RU  = default_embedding_model_for_language(LANGUAGE_RU)  # "frida"
EMBEDDING_DIMS_RU   = EMBEDDING_DIMS[EMBEDDING_MODEL_RU]                 # 1536

TRIPLETS_DB_RU        = "test_triplets_db_ru"
ONTO_TRIPLETS_DB_RU   = "test_onto_triplets_db_ru"
ONTOLOGY_DB_MONGO_RU  = "test_wikidata_ontology_mongo_ru"

SAMPLE_TEXTS_RU = [
    "Мария Кюри родилась в Варшаве, Польша, а позже переехала в Париж, где проводила "
    "исследования в Парижском университете. Она открыла полоний и радий.",
    "Альберт Эйнштейн разработал теорию относительности, работая в патентном бюро Берна, "
    "Швейцария. В 1921 году он был удостоен Нобелевской премии по физике.",
    "Эйфелева башня была спроектирована Гюставом Эйфелем и построена между 1887 и 1889 годом "
    "в Париже, Франция. Она служила входной аркой для Всемирной выставки 1889 года.",
]


# ── session-scoped fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mongo_client():
    client = MongoClient(MONGO_URI)
    client.admin.command("ping")
    return client


@pytest.fixture(scope="session")
def triplets_db_mongo(mongo_client):
    backend = create_triplets_database(
        backend="mongodb",
        mongo_uri=MONGO_URI,
        db_name=TRIPLETS_DB,
        drop_collections=True,
    )
    yield backend
    mongo_client.drop_database(TRIPLETS_DB)


@pytest.fixture(scope="session")
def onto_triplets_db_mongo(mongo_client):
    backend = create_ontological_triplets_database(
        backend="mongodb",
        mongo_uri=MONGO_URI,
        db_name=ONTO_TRIPLETS_DB,
        drop_collections=True,
    )
    yield backend
    mongo_client.drop_database(ONTO_TRIPLETS_DB)


@pytest.fixture(scope="session")
def triplets_db_qdrant():
    yield create_triplets_database(backend="qdrant", qdrant_url=":memory:")


@pytest.fixture(scope="session")
def onto_triplets_db_qdrant():
    yield create_ontological_triplets_database(backend="qdrant", qdrant_url=":memory:")


ONTOLOGY_DB_MONGO  = "test_wikidata_ontology_mongo"
ONTOLOGY_DB_QDRANT = "test_wikidata_ontology_qdrant"


@pytest.fixture(scope="session")
def ontology_db_mongo(mongo_client):
    backend = create_wikidata_ontology_database(
        backend="mongodb",
        mongo_uri=MONGO_URI,
        database=ONTOLOGY_DB_MONGO,
        drop_collections=True,
    )
    yield backend
    mongo_client.drop_database(ONTOLOGY_DB_MONGO)


@pytest.fixture(scope="session")
def ontology_db_qdrant():
    backend = create_wikidata_ontology_database(
        backend="qdrant", qdrant_url=":memory:"
    )
    # Qdrant :memory: is fully ephemeral — no explicit teardown needed.
    yield backend


@pytest.fixture(scope="session")
def dynamic_aligner_mongo(triplets_db_mongo):
    from wikontic.utils.dynamic_aligner import Aligner as DynamicAligner
    return DynamicAligner(triplets_db_mongo, device=DEVICE)


@pytest.fixture(scope="session")
def dynamic_aligner_qdrant(triplets_db_qdrant):
    from wikontic.utils.dynamic_aligner import Aligner as DynamicAligner
    return DynamicAligner(triplets_db_qdrant, device=DEVICE)


@pytest.fixture(scope="session")
def structured_aligner_mongo(ontology_db_mongo, onto_triplets_db_mongo):
    from wikontic.utils.structured_aligner import Aligner as StructuredAligner
    return StructuredAligner(ontology_db_mongo, onto_triplets_db_mongo, device=DEVICE)


@pytest.fixture(scope="session")
def structured_aligner_qdrant(ontology_db_qdrant, onto_triplets_db_qdrant):
    from wikontic.utils.structured_aligner import Aligner as StructuredAligner
    return StructuredAligner(ontology_db_qdrant, onto_triplets_db_qdrant, device=DEVICE)


@pytest.fixture(scope="session")
def llm_extractor():
    if not OPENROUTER_KEY:
        pytest.skip("No OPENROUTER_KEY / KEY in .env")
    from wikontic.utils.openai_utils import LLMTripletExtractor
    return LLMTripletExtractor(
        api_key=OPENROUTER_KEY,
        model=LLM_MODEL,
        base_url=OPENROUTER_URL,
        proxy=PROXY_URL,
    )


@pytest.fixture(scope="session")
def inference_with_db_mongo(llm_extractor, dynamic_aligner_mongo, triplets_db_mongo):
    from wikontic.utils.inference_with_db import InferenceWithDB
    return InferenceWithDB(llm_extractor, dynamic_aligner_mongo, triplets_db_mongo)


# ── ru session-scoped fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def triplets_db_mongo_ru(mongo_client):
    backend = create_triplets_database(
        backend="mongodb",
        mongo_uri=MONGO_URI,
        db_name=TRIPLETS_DB_RU,
        embedding_dimensions=EMBEDDING_DIMS_RU,
        drop_collections=True,
    )
    yield backend
    mongo_client.drop_database(TRIPLETS_DB_RU)


@pytest.fixture(scope="session")
def onto_triplets_db_mongo_ru(mongo_client):
    backend = create_ontological_triplets_database(
        backend="mongodb",
        mongo_uri=MONGO_URI,
        db_name=ONTO_TRIPLETS_DB_RU,
        embedding_dimensions=EMBEDDING_DIMS_RU,
        drop_collections=True,
    )
    yield backend
    mongo_client.drop_database(ONTO_TRIPLETS_DB_RU)


@pytest.fixture(scope="session")
def triplets_db_qdrant_ru():
    yield create_triplets_database(
        backend="qdrant", qdrant_url=":memory:", embedding_dimensions=EMBEDDING_DIMS_RU
    )


@pytest.fixture(scope="session")
def onto_triplets_db_qdrant_ru():
    yield create_ontological_triplets_database(
        backend="qdrant", qdrant_url=":memory:", embedding_dimensions=EMBEDDING_DIMS_RU
    )


@pytest.fixture(scope="session")
def ontology_db_mongo_ru(mongo_client):
    backend = create_wikidata_ontology_database(
        backend="mongodb",
        mongo_uri=MONGO_URI,
        database=ONTOLOGY_DB_MONGO_RU,
        language=LANGUAGE_RU,
        drop_collections=True,
    )
    yield backend
    mongo_client.drop_database(ONTOLOGY_DB_MONGO_RU)


@pytest.fixture(scope="session")
def ontology_db_qdrant_ru():
    backend = create_wikidata_ontology_database(
        backend="qdrant", qdrant_url=":memory:", language=LANGUAGE_RU
    )
    # Qdrant :memory: is fully ephemeral — no explicit teardown needed.
    yield backend


@pytest.fixture(scope="session")
def dynamic_aligner_mongo_ru(triplets_db_mongo_ru):
    from wikontic.utils.dynamic_aligner import Aligner as DynamicAligner
    return DynamicAligner(triplets_db_mongo_ru, device=DEVICE, embedding_model=EMBEDDING_MODEL_RU)


@pytest.fixture(scope="session")
def dynamic_aligner_qdrant_ru(triplets_db_qdrant_ru):
    from wikontic.utils.dynamic_aligner import Aligner as DynamicAligner
    return DynamicAligner(triplets_db_qdrant_ru, device=DEVICE, embedding_model=EMBEDDING_MODEL_RU)


@pytest.fixture(scope="session")
def structured_aligner_mongo_ru(ontology_db_mongo_ru, onto_triplets_db_mongo_ru):
    from wikontic.utils.structured_aligner import Aligner as StructuredAligner
    return StructuredAligner(
        ontology_db_mongo_ru, onto_triplets_db_mongo_ru, device=DEVICE, language=LANGUAGE_RU
    )


@pytest.fixture(scope="session")
def structured_aligner_qdrant_ru(ontology_db_qdrant_ru, onto_triplets_db_qdrant_ru):
    from wikontic.utils.structured_aligner import Aligner as StructuredAligner
    return StructuredAligner(
        ontology_db_qdrant_ru, onto_triplets_db_qdrant_ru, device=DEVICE, language=LANGUAGE_RU
    )


@pytest.fixture(scope="session")
def llm_extractor_ru():
    if not OPENROUTER_KEY:
        pytest.skip("No OPENROUTER_KEY / KEY in .env")
    from wikontic.utils.openai_utils import LLMTripletExtractor
    from wikontic.utils.language_config import prompt_folder_for_language
    return LLMTripletExtractor(
        api_key=OPENROUTER_KEY,
        model=LLM_MODEL,
        base_url=OPENROUTER_URL,
        proxy=PROXY_URL,
        prompt_folder_path=str(prompt_folder_for_language(LANGUAGE_RU)),
    )


@pytest.fixture(scope="session")
def inference_with_db_qdrant(llm_extractor, dynamic_aligner_qdrant, triplets_db_qdrant):
    from wikontic.utils.inference_with_db import InferenceWithDB
    return InferenceWithDB(llm_extractor, dynamic_aligner_qdrant, triplets_db_qdrant)


@pytest.fixture(scope="session")
def structured_inference_with_db_mongo(llm_extractor, structured_aligner_mongo, onto_triplets_db_mongo):
    from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
    return StructuredInferenceWithDB(llm_extractor, structured_aligner_mongo, onto_triplets_db_mongo)


@pytest.fixture(scope="session")
def structured_inference_with_db_qdrant(llm_extractor, structured_aligner_qdrant, onto_triplets_db_qdrant):
    from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
    return StructuredInferenceWithDB(llm_extractor, structured_aligner_qdrant, onto_triplets_db_qdrant)


# ── ru inference fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def inference_with_db_mongo_ru(llm_extractor_ru, dynamic_aligner_mongo_ru, triplets_db_mongo_ru):
    from wikontic.utils.inference_with_db import InferenceWithDB
    return InferenceWithDB(
        llm_extractor_ru, dynamic_aligner_mongo_ru, triplets_db_mongo_ru, language=LANGUAGE_RU
    )


@pytest.fixture(scope="session")
def inference_with_db_qdrant_ru(llm_extractor_ru, dynamic_aligner_qdrant_ru, triplets_db_qdrant_ru):
    from wikontic.utils.inference_with_db import InferenceWithDB
    return InferenceWithDB(
        llm_extractor_ru, dynamic_aligner_qdrant_ru, triplets_db_qdrant_ru, language=LANGUAGE_RU
    )


@pytest.fixture(scope="session")
def structured_inference_with_db_mongo_ru(
    llm_extractor_ru, structured_aligner_mongo_ru, onto_triplets_db_mongo_ru
):
    from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
    return StructuredInferenceWithDB(
        llm_extractor_ru, structured_aligner_mongo_ru, onto_triplets_db_mongo_ru, language=LANGUAGE_RU
    )


@pytest.fixture(scope="session")
def structured_inference_with_db_qdrant_ru(
    llm_extractor_ru, structured_aligner_qdrant_ru, onto_triplets_db_qdrant_ru
):
    from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB
    return StructuredInferenceWithDB(
        llm_extractor_ru, structured_aligner_qdrant_ru, onto_triplets_db_qdrant_ru, language=LANGUAGE_RU
    )


# ── timing helper ──────────────────────────────────────────────────────────────
_timing: dict = {}


def timed(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    _timing.setdefault(label, []).append(time.perf_counter() - t0)
    return result


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _timing:
        return
    terminalreporter.write_sep("=", "Mean search times")
    for label, times in sorted(_timing.items()):
        mean_ms = 1000 * sum(times) / len(times)
        terminalreporter.write_line(f"  {label:<72s}  {mean_ms:>8.1f} ms  (n={len(times)})")
