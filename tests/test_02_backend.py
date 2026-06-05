"""Backend CRUD and vector_search for both MongoDB and Qdrant."""

import pytest
from wikontic.db.interfaces import VectorQuery
from conftest import timed

DUMMY_EMBEDDING = [0.01] * 768
SAMPLE_ID = "test_backend"
ENTITY_DOC = {
    "label": "Test Entity",
    "alias": "TE",
    "sample_id": SAMPLE_ID,
    "alias_text_embedding": DUMMY_EMBEDDING,
}


@pytest.fixture(params=["mongo", "qdrant"])
def backend(request, triplets_db_mongo, triplets_db_qdrant):
    return triplets_db_mongo if request.param == "mongo" else triplets_db_qdrant


def test_upsert_many(backend):
    backend.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=["label", "alias", "sample_id"]
    )


def test_match_documents(backend):
    backend.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=["label", "alias", "sample_id"]
    )
    docs = backend.match_documents("entity_aliases", {"sample_id": SAMPLE_ID})
    assert len(docs) >= 1


def test_get_collection_find_one(backend):
    backend.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=["label", "alias", "sample_id"]
    )
    coll = backend.get_collection("entity_aliases")
    doc = coll.find_one({"label": "Test Entity"})
    assert doc is not None


def test_vector_search_mongo(triplets_db_mongo):
    triplets_db_mongo.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=["label", "alias", "sample_id"]
    )
    results = timed(
        "MongoBackend.vector_search entity_aliases",
        triplets_db_mongo.vector_search,
        VectorQuery(
            collection_name="entity_aliases",
            index_name="entity_aliases",
            query_vector=DUMMY_EMBEDDING,
            vector_field="alias_text_embedding",
            limit=5,
        ),
    )
    assert isinstance(results, list)


def test_vector_search_qdrant(triplets_db_qdrant):
    triplets_db_qdrant.upsert_many(
        "entity_aliases", [ENTITY_DOC], unique_fields=["label", "alias", "sample_id"]
    )
    results = timed(
        "QdrantBackend.vector_search entity_aliases",
        triplets_db_qdrant.vector_search,
        VectorQuery(
            collection_name="entity_aliases",
            index_name="entity_aliases",
            query_vector=DUMMY_EMBEDDING,
            vector_field="alias_text_embedding",
            limit=5,
        ),
    )
    assert isinstance(results, list)
