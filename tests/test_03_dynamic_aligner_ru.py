"""DynamicAligner (dynamic_aligner.py) for language='ru' — both MongoDB and Qdrant.

Mirrors test_03_dynamic_aligner.py but exercises the FRIDA embedding model
instead of Contriever, via the ru-suffixed fixtures in conftest.py.
"""

import pytest
from conftest import timed

SAMPLE_ID = "test_dyn_ru"
TRIPLET = {"subject": "Париж", "relation": "находится в", "object": "Франция"}


def _count_docs(aligner, collection, query):
    return len(aligner.db.match_documents(collection, query))


def _assert_single_doc(aligner, collection, query, expected_fields):
    docs = aligner.db.match_documents(collection, query)
    assert len(docs) == 1
    for key, value in expected_fields.items():
        assert docs[0].get(key) == value
    return docs[0]


@pytest.fixture(params=["mongo", "qdrant"])
def aligner(request, dynamic_aligner_mongo_ru, dynamic_aligner_qdrant_ru):
    return dynamic_aligner_mongo_ru if request.param == "mongo" else dynamic_aligner_qdrant_ru


def _backend_label(aligner):
    return "mongo" if hasattr(aligner.db, "db") else "qdrant"


def test_add_entity(aligner):
    sample_id = "test_dyn_ru_add_entity"
    entities = [
        ("Париж", "город огней"),
        ("Франция", "Французская Республика"),
    ]
    for label, alias in entities:
        aligner.add_entity(label, alias, sample_id)
        _assert_single_doc(
            aligner,
            "entity_aliases",
            {"label": label, "alias": alias, "sample_id": sample_id},
            {"label": label, "alias": alias, "sample_id": sample_id},
        )
        aligner.add_entity(label, alias, sample_id)
        assert _count_docs(
            aligner,
            "entity_aliases",
            {"label": label, "alias": alias, "sample_id": sample_id},
        ) == 1


def test_add_property(aligner):
    label, alias = "находится в", "расположен в"
    aligner.add_property(label, alias, SAMPLE_ID)
    _assert_single_doc(
        aligner,
        "property_aliases",
        {"label": label, "alias": alias},
        {"label": label, "alias": alias},
    )
    aligner.add_property(label, alias, SAMPLE_ID)
    assert _count_docs(
        aligner, "property_aliases", {"label": label, "alias": alias}
    ) == 1


def test_add_triplets(aligner):
    sample_id = "test_dyn_ru_add_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    aligner.add_triplets([triplet.copy()], sample_id)
    _assert_single_doc(aligner, "triplets", query, query)
    aligner.add_triplets([triplet.copy()], sample_id)
    assert _count_docs(aligner, "triplets", query) == 1


def test_add_initial_triplets(aligner):
    sample_id = "test_dyn_ru_add_initial_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    aligner.add_initial_triplets([triplet.copy()], sample_id)
    _assert_single_doc(aligner, "initial_triplets", query, query)
    aligner.add_initial_triplets([triplet.copy()], sample_id)
    assert _count_docs(aligner, "initial_triplets", query) == 1


def test_add_filtered_triplets(aligner):
    sample_id = "test_dyn_ru_add_filtered_triplets"
    triplet = {"subject": "Рим", "relation": "столица", "object": "Италия"}
    query = {**triplet, "sample_id": sample_id}
    aligner.add_filtered_triplets([triplet.copy()], sample_id)
    _assert_single_doc(aligner, "filtered_triplets", query, query)
    aligner.add_filtered_triplets([triplet.copy()], sample_id)
    assert _count_docs(aligner, "filtered_triplets", query) == 1


def _assert_unique_labels(results, label_name="labels"):
    assert len(results) == len(set(results)), (
        f"expected unique {label_name}, got duplicates in {results!r}"
    )


def test_retrieve_similar_entity_names(aligner):
    # Seed two entities with clearly different aliases so ranking is meaningful.
    aligner.add_entity("Париж", "город огней", SAMPLE_ID)
    aligner.add_entity("Франция", "Французская Республика", SAMPLE_ID)

    results = timed(
        f"DynamicAligner-ru({_backend_label(aligner)}).retrieve_similar_entity_names",
        aligner.retrieve_similar_entity_names,
        "город огней", sample_id=SAMPLE_ID, k=3,
    )

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "at least one result expected"
    assert all(isinstance(r, str) for r in results), "each result must be a label string"
    _assert_unique_labels(results, label_name="entity labels")
    # The alias 'город огней' was added for 'Париж' — it must rank first.
    assert results[0] == "Париж", f"expected 'Париж' as top result, got {results[0]!r}"


def test_retrieve_similar_properties(aligner):
    aligner.add_property("находится в", "расположен в", SAMPLE_ID)
    aligner.add_property("столица", "является столицей", SAMPLE_ID)

    results = timed(
        f"DynamicAligner-ru({_backend_label(aligner)}).retrieve_similar_properties",
        aligner.retrieve_similar_properties,
        "находится в", sample_id=SAMPLE_ID, k=3,
    )

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "at least one result expected"
    assert all(isinstance(r, str) for r in results), "each result must be a label string"
    _assert_unique_labels(results, label_name="property labels")
    # The alias 'расположен в' was added for 'находится в' — it must rank first.
    assert results[0] == "находится в", f"expected 'находится в' as top result, got {results[0]!r}"
