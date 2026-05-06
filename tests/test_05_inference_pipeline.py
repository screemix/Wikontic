"""
InferenceWithDB and StructuredInferenceWithDB on sample texts — MongoDB and Qdrant.
Results are written to the `test_pipeline` collection in the respective DB.
Requires OPENROUTER_KEY / KEY in .env — tests are skipped otherwise.
"""

import json
import logging
import pytest
from conftest import timed, SAMPLE_TEXTS

logger = logging.getLogger(__name__)


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


# ══════════════════════════════════════════════════════════════════════════════
# InferenceWithDB (non-structured) — MongoDB and Qdrant
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=["mongo", "qdrant"])
def inference_backend(request, inference_with_db_mongo, inference_with_db_qdrant,
                      triplets_db_mongo, triplets_db_qdrant):
    if request.param == "mongo":
        return request.param, inference_with_db_mongo, triplets_db_mongo
    return request.param, inference_with_db_qdrant, triplets_db_qdrant


@pytest.mark.parametrize("idx,text", list(enumerate(SAMPLE_TEXTS)))
def test_inference_extract_and_store(inference_backend, idx, text):
    backend_label, inference, db = inference_backend
    sid = f"test_inference_{backend_label}_{idx}"

    initial, final, filtered = timed(
        f"InferenceWithDB({backend_label}).extract_triplets_and_add_to_db",
        inference.extract_triplets_and_add_to_db,
        text, source_text_id=f"src_{idx}", sample_id=sid,
    )

    assert isinstance(initial, list)
    assert isinstance(final, list)
    assert isinstance(filtered, list)

    _log_triplets(sid, backend_label, initial, final, filtered)

    db.upsert_many(
        "test_pipeline",
        [{
            "sample_id": sid,
            "text": text,
            "initial_triplets": initial,
            "final_triplets": final,
            "filtered_triplets": filtered,
        }],
        unique_fields=["sample_id"],
    )

    stored = db.match_documents("test_pipeline", {"sample_id": sid})
    assert len(stored) == 1
    assert stored[0]["text"] == text


# ══════════════════════════════════════════════════════════════════════════════
# StructuredInferenceWithDB — MongoDB and Qdrant
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=["mongo", "qdrant"])
def structured_inference_backend(request,
                                  structured_inference_with_db_mongo,
                                  structured_inference_with_db_qdrant,
                                  onto_triplets_db_mongo,
                                  onto_triplets_db_qdrant):
    if request.param == "mongo":
        return request.param, structured_inference_with_db_mongo, onto_triplets_db_mongo
    return request.param, structured_inference_with_db_qdrant, onto_triplets_db_qdrant


@pytest.mark.parametrize("idx,text", list(enumerate(SAMPLE_TEXTS)))
def test_structured_inference_extract_and_store(structured_inference_backend, idx, text):
    backend_label, inference, db = structured_inference_backend
    sid = f"test_struct_inference_{backend_label}_{idx}"

    initial, final, filtered, onto_filtered = timed(
        f"StructuredInferenceWithDB({backend_label}).extract_triplets_with_ontology_filtering_and_add_to_db",
        inference.extract_triplets_with_ontology_filtering_and_add_to_db,
        text, source_text_id=f"src_{idx}", sample_id=sid,
    )

    assert isinstance(initial, list)
    assert isinstance(final, list)
    assert isinstance(filtered, list)
    assert isinstance(onto_filtered, list)

    _log_triplets(sid, backend_label, initial, final, filtered, onto_filtered)

    db.upsert_many(
        "test_pipeline",
        [{
            "sample_id": sid,
            "text": text,
            "initial_triplets": initial,
            "final_triplets": final,
            "filtered_triplets": filtered,
            "ontology_filtered_triplets": onto_filtered,
        }],
        unique_fields=["sample_id"],
    )

    stored = db.match_documents("test_pipeline", {"sample_id": sid})
    assert len(stored) == 1
    assert stored[0]["text"] == text
