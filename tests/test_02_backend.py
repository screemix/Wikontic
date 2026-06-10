"""Backend CRUD and vector_search for both MongoDB and Qdrant."""

import pytest
from wikontic.db.interfaces import VectorQuery
from conftest import timed

DUMMY_EMBEDDING = [0.01] * 768
SAMPLE_ID = "test_backend"
VECTOR_SAMPLE_ID = "test_backend_vector"
ENTITY_DOC = {
    "label": "Test Entity",
    "alias": "TE",
    "sample_id": SAMPLE_ID,
    "alias_text_embedding": DUMMY_EMBEDDING,
}
VECTOR_ENTITY_DOC = {
    "label": "Test Entity",
    "alias": "TE",
    "sample_id": VECTOR_SAMPLE_ID,
    "alias_text_embedding": DUMMY_EMBEDDING,
}
OTHER_VECTOR_ENTITY_DOC = {
    "label": "Other Entity",
    "alias": "OE",
    "sample_id": VECTOR_SAMPLE_ID,
    "alias_text_embedding": [1.0] + [0.0] * (len(DUMMY_EMBEDDING) - 1),
}
UNIQUE_FIELDS = ["label", "alias", "sample_id"]


def _doc_identity_query():
    return {field: ENTITY_DOC[field] for field in UNIQUE_FIELDS}


def _assert_doc_matches(actual, expected):
    for key, value in expected.items():
        assert actual.get(key) == value


def _assert_vector_search_returns_expected(backend, timed_label):
    backend.upsert_many(
        "entity_aliases",
        [VECTOR_ENTITY_DOC, OTHER_VECTOR_ENTITY_DOC],
        unique_fields=UNIQUE_FIELDS,
    )
    results = timed(
        timed_label,
        backend.vector_search,
        VectorQuery(
            collection_name="entity_aliases",
            index_name="entity_aliases",
            query_vector=DUMMY_EMBEDDING,
            vector_field="alias_text_embedding",
            limit=1,
            filters={"sample_id": {"$eq": VECTOR_SAMPLE_ID}},
        ),
    )
    assert len(results) >= 1
    _assert_doc_matches(results[0], VECTOR_ENTITY_DOC)


@pytest.fixture(params=["mongo", "qdrant"])
def backend(request, triplets_db_mongo, triplets_db_qdrant):
    return triplets_db_mongo if request.param == "mongo" else triplets_db_qdrant


def test_upsert_many(backend):
    backend.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=UNIQUE_FIELDS
    )
    docs = backend.match_documents("entity_aliases", _doc_identity_query())
    assert len(docs) == 1
    _assert_doc_matches(docs[0], ENTITY_DOC)

    backend.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=UNIQUE_FIELDS
    )
    docs = backend.match_documents("entity_aliases", _doc_identity_query())
    assert len(docs) == 1
    _assert_doc_matches(docs[0], ENTITY_DOC)


def test_match_documents(backend):
    backend.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=UNIQUE_FIELDS
    )
    docs = backend.match_documents("entity_aliases", {"sample_id": SAMPLE_ID})
    assert len(docs) >= 1


def test_get_collection_find_one(backend):
    backend.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=UNIQUE_FIELDS
    )
    coll = backend.get_collection("entity_aliases")
    doc = coll.find_one({"label": "Test Entity"})
    assert doc is not None


def test_vector_search_mongo(triplets_db_mongo):
    _assert_vector_search_returns_expected(
        triplets_db_mongo, "MongoBackend.vector_search entity_aliases"
    )


def test_vector_search_qdrant(triplets_db_qdrant):
    _assert_vector_search_returns_expected(
        triplets_db_qdrant, "QdrantBackend.vector_search entity_aliases"
    )
