"""DynamicAligner (dynamic_aligner.py) — both MongoDB and Qdrant."""

import pytest
from conftest import timed

SID = "test_dyn"
TRIPLET = {"subject": "Paris", "relation": "located in", "object": "France"}


@pytest.fixture(params=["mongo", "qdrant"])
def aligner(request, dynamic_aligner_mongo, dynamic_aligner_qdrant):
    return dynamic_aligner_mongo if request.param == "mongo" else dynamic_aligner_qdrant


def _backend_label(aligner):
    return "mongo" if hasattr(aligner.db, "db") else "qdrant"


def test_add_entity(aligner):
    aligner.add_entity("Paris", "the city of light", SID)
    aligner.add_entity("France", "French republic", SID)


def test_add_property(aligner):
    aligner.add_property("located in", "is located in", SID)


def test_add_triplets(aligner):
    aligner.add_triplets([TRIPLET.copy()], SID)


def test_add_initial_triplets(aligner):
    aligner.add_initial_triplets([TRIPLET.copy()], SID)


def test_add_filtered_triplets(aligner):
    aligner.add_filtered_triplets(
        [{"subject": "Rome", "relation": "capital of", "object": "Italy"}], SID
    )


def test_retrieve_similar_entity_names(aligner):
    # Seed two entities with clearly different aliases so ranking is meaningful.
    aligner.add_entity("Paris", "the city of light", SID)
    aligner.add_entity("France", "French republic", SID)

    results = timed(
        f"DynamicAligner({_backend_label(aligner)}).retrieve_similar_entity_names",
        aligner.retrieve_similar_entity_names,
        "city of light", sample_id=SID, k=3,
    )

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "at least one result expected"
    assert all(isinstance(r, str) for r in results), "each result must be a label string"
    # The alias 'the city of light' was added for 'Paris' — it must rank first.
    assert results[0] == "Paris", f"expected 'Paris' as top result, got {results[0]!r}"


def test_retrieve_similar_properties(aligner):
    aligner.add_property("located in", "is located in", SID)
    aligner.add_property("capital of", "is the capital of", SID)

    results = timed(
        f"DynamicAligner({_backend_label(aligner)}).retrieve_similar_properties",
        aligner.retrieve_similar_properties,
        "located in", sample_id=SID, k=3,
    )

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "at least one result expected"
    assert all(isinstance(r, str) for r in results), "each result must be a label string"
    # The alias 'is located in' was added for 'located in' — it must rank first.
    assert results[0] == "located in", f"expected 'located in' as top result, got {results[0]!r}"
