"""
InferenceWithDB and StructuredInferenceWithDB on Russian sample texts — MongoDB and Qdrant.
Mirrors test_05_inference_pipeline.py, but exercises language='ru' end to end:
Russian prompts, FRIDA embeddings, and the ru ontology mappings.
Requires OPENROUTER_KEY / KEY in .env — tests are skipped otherwise.
"""

import json

import pytest
from wikontic.logging_config import get_logger
from conftest import timed, SAMPLE_TEXTS_RU

logger = get_logger(__name__)

DYNAMIC_IDENTITY = ["subject", "relation", "object", "sample_id"]
STRUCTURED_IDENTITY = [
    "subject",
    "relation",
    "object",
    "subject_type",
    "object_type",
    "sample_id",
]


def _log_triplets(sid: str, backend: str, initial, final, filtered, onto_filtered=None):
    logger.info("\n%s\n  backend   : %s\n  sample_id : %s", "─" * 60, backend, sid)
    logger.info("  initial   (%d):\n%s", len(initial),
                json.dumps(initial, indent=4, ensure_ascii=False))
    logger.info("  final     (%d):\n%s", len(final),
                json.dumps(final, indent=4, ensure_ascii=False))
    logger.info("  filtered  (%d):\n%s", len(filtered),
                json.dumps(filtered, indent=4, ensure_ascii=False))
    if onto_filtered is not None:
        logger.info("  onto_filt (%d):\n%s", len(onto_filtered),
                    json.dumps(onto_filtered, indent=4, ensure_ascii=False))


def _triplet_query(triplet: dict, sample_id: str, identity_keys) -> dict:
    query = {
        key: triplet[key]
        for key in identity_keys
        if key != "sample_id" and key in triplet
    }
    query["sample_id"] = sample_id
    return query


def _count_docs(db, collection: str, query: dict) -> int:
    return len(db.match_documents(collection, query))


def _assert_triplets_stored_once(db, collection, sample_id, triplets, identity_keys):
    stored = db.match_documents(collection, {"sample_id": sample_id})
    assert len(stored) == len(triplets)
    for triplet in triplets:
        query = _triplet_query(triplet, sample_id, identity_keys)
        assert _count_docs(db, collection, query) == 1, (
            f"expected one stored triplet in {collection} for {query}, "
            f"found {_count_docs(db, collection, query)}"
        )


def _assert_no_duplicates_on_readd(aligner, db, collection, sample_id, triplets, identity_keys, add_fn):
    if not triplets:
        return
    add_fn(triplets, sample_id)
    _assert_triplets_stored_once(db, collection, sample_id, triplets, identity_keys)


def _verify_inference_storage(inference, db, sample_id, initial, final, filtered, *, ontology_filtered=None):
    aligner = inference.aligner
    identity_keys = (
        STRUCTURED_IDENTITY if ontology_filtered is not None else DYNAMIC_IDENTITY
    )

    collections = [
        ("initial_triplets", initial, aligner.add_initial_triplets),
        ("triplets", final, aligner.add_triplets),
        ("filtered_triplets", filtered, aligner.add_filtered_triplets),
    ]
    if ontology_filtered is not None:
        collections.append(
            (
                "ontology_filtered_triplets",
                ontology_filtered,
                aligner.add_ontology_filtered_triplets,
            )
        )

    for collection, triplets, add_fn in collections:
        _assert_triplets_stored_once(db, collection, sample_id, triplets, identity_keys)
        _assert_no_duplicates_on_readd(
            aligner, db, collection, sample_id, triplets, identity_keys, add_fn
        )


# ══════════════════════════════════════════════════════════════════════════════
# InferenceWithDB (non-structured) — MongoDB and Qdrant, ru
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=["mongo", "qdrant"])
def inference_backend_ru(request, inference_with_db_mongo_ru, inference_with_db_qdrant_ru,
                          triplets_db_mongo_ru, triplets_db_qdrant_ru):
    if request.param == "mongo":
        return request.param, inference_with_db_mongo_ru, triplets_db_mongo_ru
    return request.param, inference_with_db_qdrant_ru, triplets_db_qdrant_ru


@pytest.mark.parametrize("idx,text", list(enumerate(SAMPLE_TEXTS_RU)))
def test_inference_extract_and_store(inference_backend_ru, idx, text):
    backend_label, inference, db = inference_backend_ru
    sid = f"test_inference_ru_{backend_label}_{idx}"

    initial, final, filtered = timed(
        f"InferenceWithDB-ru({backend_label}).extract_triplets_and_add_to_db",
        inference.extract_triplets_and_add_to_db,
        text, source_text_id=f"src_ru_{idx}", sample_id=sid,
    )

    assert isinstance(initial, list)
    assert isinstance(final, list)
    assert isinstance(filtered, list)
    assert len(final) > 0

    _log_triplets(sid, backend_label, initial, final, filtered)
    _verify_inference_storage(inference, db, sid, initial, final, filtered)


# ══════════════════════════════════════════════════════════════════════════════
# StructuredInferenceWithDB — MongoDB and Qdrant, ru
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=["mongo", "qdrant"])
def structured_inference_backend_ru(request,
                                     structured_inference_with_db_mongo_ru,
                                     structured_inference_with_db_qdrant_ru,
                                     onto_triplets_db_mongo_ru,
                                     onto_triplets_db_qdrant_ru):
    if request.param == "mongo":
        return request.param, structured_inference_with_db_mongo_ru, onto_triplets_db_mongo_ru
    return request.param, structured_inference_with_db_qdrant_ru, onto_triplets_db_qdrant_ru


@pytest.mark.parametrize("idx,text", list(enumerate(SAMPLE_TEXTS_RU)))
def test_structured_inference_extract_and_store(structured_inference_backend_ru, idx, text):
    backend_label, inference, db = structured_inference_backend_ru
    sid = f"test_struct_inference_ru_{backend_label}_{idx}"

    initial, final, filtered, onto_filtered = timed(
        f"StructuredInferenceWithDB-ru({backend_label}).extract_triplets_with_ontology_filtering_and_add_to_db",
        inference.extract_triplets_with_ontology_filtering_and_add_to_db,
        text, source_text_id=f"src_ru_{idx}", sample_id=sid,
    )

    assert isinstance(initial, list)
    assert isinstance(final, list)
    assert isinstance(filtered, list)
    assert isinstance(onto_filtered, list)
    assert len(final) > 0

    _log_triplets(sid, backend_label, initial, final, filtered, onto_filtered)
    _verify_inference_storage(
        inference,
        db,
        sid,
        initial,
        final,
        filtered,
        ontology_filtered=onto_filtered,
    )
