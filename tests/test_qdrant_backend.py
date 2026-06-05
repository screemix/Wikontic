from wikontic.db.factory import create_backend
from wikontic.db.interfaces import VectorQuery


def test_qdrant_upsert_and_match_documents():
    backend = create_backend(backend_type="qdrant", qdrant_url=":memory:")
    backend.ensure_collection("triplets")
    backend.upsert_many(
        collection_name="triplets",
        documents=[
            {"subject": "a", "relation": "r", "object": "b", "sample_id": "s1"},
            {"subject": "a", "relation": "r", "object": "b", "sample_id": "s1"},
        ],
        unique_fields=["subject", "relation", "object", "sample_id"],
    )
    matches = backend.match_documents(
        "triplets", {"sample_id": "s1", "$or": [{"subject": "a"}, {"object": "a"}]}
    )
    assert len(matches) == 1


def test_qdrant_match_documents_with_in_and_and():
    backend = create_backend(backend_type="qdrant", qdrant_url=":memory:")
    backend.upsert_many(
        collection_name="triplets",
        documents=[
            {"subject": "a", "relation": "r1", "object": "x", "sample_id": "s1"},
            {"subject": "a", "relation": "r2", "object": "y", "sample_id": "s1"},
            {"subject": "b", "relation": "r1", "object": "x", "sample_id": "s2"},
        ],
        unique_fields=["subject", "relation", "object", "sample_id"],
    )
    matches = backend.match_documents(
        "triplets",
        {
            "$and": [
                {"sample_id": {"$eq": "s1"}},
                {"relation": {"$in": ["r1", "r2"]}},
                {"subject": "a"},
            ]
        },
    )
    assert len(matches) == 2


def test_qdrant_vector_search():
    backend = create_backend(backend_type="qdrant", qdrant_url=":memory:")
    backend.upsert_many(
        collection_name="entity_aliases",
        documents=[
            {"label": "Paris", "sample_id": "s1", "alias_text_embedding": [1.0, 0.0]},
            {"label": "Berlin", "sample_id": "s1", "alias_text_embedding": [0.0, 1.0]},
        ],
        unique_fields=["label", "sample_id"],
    )
    results = backend.vector_search(
        VectorQuery(
            collection_name="entity_aliases",
            index_name="entity_aliases",
            query_vector=[0.9, 0.1],
            vector_field="alias_text_embedding",
            limit=1,
            filters={"sample_id": {"$eq": "s1"}},
            projection={"label": 1, "_id": 0},
        )
    )
    assert results[0]["label"] == "Paris"
