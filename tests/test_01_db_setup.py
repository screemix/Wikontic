"""DB setup: collections, regular indexes, vector indexes (Mongo Atlas / Qdrant)."""

import pytest

from conftest import (
    MONGO_URI,
    TRIPLETS_DB,
    ONTO_TRIPLETS_DB,
    ONTOLOGY_DB_MONGO,
    TRIPLETS_DB_RU,
    ONTO_TRIPLETS_DB_RU,
    ONTOLOGY_DB_MONGO_RU,
    EMBEDDING_DIMS_RU,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _index_field_sets(mongo_db, collection_name):
    coll = mongo_db.get_collection(collection_name)
    return [
        frozenset(idx["key"].keys()) - {"_id"}
        for idx in coll.list_indexes()
    ]


def _has_index(mongo_db, collection_name, *fields):
    expected = frozenset(fields)
    return any(expected.issubset(s) for s in _index_field_sets(mongo_db, collection_name))


def _vector_index_names(mongo_db, collection_name):
    coll = mongo_db.get_collection(collection_name)
    return [idx.get("name") for idx in coll.list_search_indexes()]


def _vector_index_dimensions(mongo_db, collection_name, index_name, vector_field):
    coll = mongo_db.get_collection(collection_name)
    for idx in coll.list_search_indexes():
        if idx.get("name") != index_name:
            continue
        definition = idx.get("latestDefinition") or idx.get("definition") or {}
        fields = definition.get("mappings", {}).get("fields", {})
        return fields.get(vector_field, {}).get("dimensions")
    return None


def _qdrant_payload_fields(backend, collection_name):
    list_fields = getattr(backend, "list_payload_index_fields", None)
    if callable(list_fields):
        # Local :memory: Qdrant does not expose payload_schema; backend tracks fields.
        return list_fields(collection_name)
    client = getattr(backend, "client", None)
    if client is None:
        pytest.skip("Not a Qdrant backend")
    schema = client.get_collection(collection_name).payload_schema or {}
    return set(schema.keys())


def _has_payload_index(backend, collection_name, *fields):
    indexed = _qdrant_payload_fields(backend, collection_name)
    return all(field in indexed for field in fields)


def _qdrant_named_vectors(backend, collection_name):
    client = getattr(backend, "client", None)
    if client is None:
        pytest.skip("Not a Qdrant backend")
    vectors = client.get_collection(collection_name).config.params.vectors
    if vectors is None:
        return set()
    if isinstance(vectors, dict):
        return set(vectors.keys())
    return set()


# ── collections ────────────────────────────────────────────────────────────────

TRIPLETS_COLLS      = {"entity_aliases", "property_aliases", "triplets",
                        "initial_triplets", "filtered_triplets"}
ONTO_TRIPLETS_COLLS = {"entity_aliases", "triplets", "initial_triplets",
                        "filtered_triplets", "ontology_filtered_triplets"}
ONTOLOGY_COLLS      = {"entity_types", "entity_type_aliases", "properties",
                        "property_aliases"}


@pytest.mark.parametrize("coll", sorted(TRIPLETS_COLLS))
def test_triplets_db_collection(triplets_db_mongo, coll):
    assert coll in triplets_db_mongo.list_collection_names()


@pytest.mark.parametrize("coll", sorted(ONTO_TRIPLETS_COLLS))
def test_onto_triplets_db_collection(onto_triplets_db_mongo, coll):
    assert coll in onto_triplets_db_mongo.list_collection_names()


@pytest.mark.parametrize("coll", sorted(ONTOLOGY_COLLS))
def test_ontology_db_collection_mongo(ontology_db_mongo, coll):
    assert coll in ontology_db_mongo.list_collection_names()


@pytest.mark.parametrize("coll", sorted(ONTOLOGY_COLLS))
def test_ontology_db_collection_qdrant(ontology_db_qdrant, coll):
    assert coll in ontology_db_qdrant.list_collection_names()


@pytest.mark.parametrize("coll", sorted(TRIPLETS_COLLS))
def test_triplets_db_collection_qdrant(triplets_db_qdrant, coll):
    assert coll in triplets_db_qdrant.list_collection_names()


@pytest.mark.parametrize("coll", sorted(ONTO_TRIPLETS_COLLS))
def test_onto_triplets_db_collection_qdrant(onto_triplets_db_qdrant, coll):
    assert coll in onto_triplets_db_qdrant.list_collection_names()


# ── regular indexes (MongoDB) ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def triplets_raw(mongo_client, triplets_db_mongo):
    return mongo_client.get_database(TRIPLETS_DB)


@pytest.fixture(scope="module")
def onto_raw(mongo_client, onto_triplets_db_mongo):
    return mongo_client.get_database(ONTO_TRIPLETS_DB)


@pytest.fixture(scope="module")
def ontology_raw(mongo_client, ontology_db_mongo):
    return mongo_client.get_database(ONTOLOGY_DB_MONGO)


@pytest.mark.parametrize("fields", [
    ["sample_id"],
    ["label"],
])
def test_triplets_entity_aliases_index(triplets_raw, fields):
    assert _has_index(triplets_raw, "entity_aliases", *fields)


@pytest.mark.parametrize("coll", ["triplets", "initial_triplets", "filtered_triplets"])
def test_triplets_sample_id_index(triplets_raw, coll):
    assert _has_index(triplets_raw, coll, "sample_id")


@pytest.mark.parametrize("fields", [
    ["entity_type", "sample_id"],
    ["label"],
])
def test_onto_triplets_entity_aliases_index(onto_raw, fields):
    assert _has_index(onto_raw, "entity_aliases", *fields)


@pytest.mark.parametrize("coll", [
    "triplets", "initial_triplets", "filtered_triplets", "ontology_filtered_triplets"
])
def test_onto_triplets_sample_id_index(onto_raw, coll):
    assert _has_index(onto_raw, coll, "sample_id")


@pytest.mark.parametrize("collection,fields", [
    ("entity_types",        ["entity_type_id"]),
    ("entity_types",        ["label"]),
    ("entity_type_aliases", ["entity_type_id"]),
    ("entity_type_aliases", ["alias_label"]),
    ("properties",          ["property_id"]),
])
def test_ontology_regular_indexes(ontology_raw, collection, fields):
    assert _has_index(ontology_raw, collection, *fields)


# ── vector (Atlas Search) indexes ─────────────────────────────────────────────

def test_triplets_vector_indexes(triplets_raw):
    assert "entity_aliases" in _vector_index_names(triplets_raw, "entity_aliases")
    assert "property_aliases" in _vector_index_names(triplets_raw, "property_aliases")


def test_onto_triplets_vector_index(onto_raw):
    assert "entity_aliases" in _vector_index_names(onto_raw, "entity_aliases")


@pytest.mark.parametrize("collection,index_name", [
    ("entity_type_aliases", "entity_type_aliases"),
    ("property_aliases",    "property_aliases"),
])
def test_ontology_vector_indexes(ontology_raw, collection, index_name):
    assert index_name in _vector_index_names(ontology_raw, collection)


# ── payload indexes (Qdrant) ─────────────────────────────────────────────────

@pytest.mark.parametrize("fields", [
    ["sample_id"],
    ["label"],
])
def test_triplets_entity_aliases_payload_index_qdrant(triplets_db_qdrant, fields):
    assert _has_payload_index(triplets_db_qdrant, "entity_aliases", *fields)


@pytest.mark.parametrize("coll", ["triplets", "initial_triplets", "filtered_triplets"])
def test_triplets_sample_id_payload_index_qdrant(triplets_db_qdrant, coll):
    assert _has_payload_index(triplets_db_qdrant, coll, "sample_id")


@pytest.mark.parametrize("fields", [
    ["entity_type", "sample_id"],
    ["label"],
])
def test_onto_triplets_entity_aliases_payload_index_qdrant(onto_triplets_db_qdrant, fields):
    assert _has_payload_index(onto_triplets_db_qdrant, "entity_aliases", *fields)


@pytest.mark.parametrize("coll", [
    "triplets", "initial_triplets", "filtered_triplets", "ontology_filtered_triplets"
])
def test_onto_triplets_sample_id_payload_index_qdrant(onto_triplets_db_qdrant, coll):
    assert _has_payload_index(onto_triplets_db_qdrant, coll, "sample_id")


@pytest.mark.parametrize("collection,fields", [
    ("entity_types",        ["entity_type_id"]),
    ("entity_types",        ["label"]),
    ("entity_type_aliases", ["entity_type_id"]),
    ("entity_type_aliases", ["alias_label"]),
    ("properties",          ["property_id"]),
])
def test_ontology_payload_indexes_qdrant(ontology_db_qdrant, collection, fields):
    assert _has_payload_index(ontology_db_qdrant, collection, *fields)


# ── named vector indexes (Qdrant) ─────────────────────────────────────────────

def test_triplets_named_vectors_qdrant(triplets_db_qdrant):
    assert "entity_aliases" in _qdrant_named_vectors(triplets_db_qdrant, "entity_aliases")
    assert "property_aliases" in _qdrant_named_vectors(triplets_db_qdrant, "property_aliases")


def test_onto_triplets_named_vector_qdrant(onto_triplets_db_qdrant):
    assert "entity_aliases" in _qdrant_named_vectors(onto_triplets_db_qdrant, "entity_aliases")


@pytest.mark.parametrize("collection,index_name", [
    ("entity_type_aliases", "entity_type_aliases"),
    ("property_aliases",    "property_aliases"),
])
def test_ontology_named_vectors_qdrant(ontology_db_qdrant, collection, index_name):
    assert index_name in _qdrant_named_vectors(ontology_db_qdrant, collection)


# ── ru: collections, and vector index dimensions (Mongo) ──────────────────────
# FRIDA (ru) embeds at 1536 dims vs Contriever (en) at 768 — these confirm the
# ru vector indexes were actually provisioned at the right size, not silently
# left at the 768-dim default.

@pytest.fixture(scope="module")
def triplets_raw_ru(mongo_client, triplets_db_mongo_ru):
    return mongo_client.get_database(TRIPLETS_DB_RU)


@pytest.fixture(scope="module")
def onto_raw_ru(mongo_client, onto_triplets_db_mongo_ru):
    return mongo_client.get_database(ONTO_TRIPLETS_DB_RU)


@pytest.fixture(scope="module")
def ontology_raw_ru(mongo_client, ontology_db_mongo_ru):
    return mongo_client.get_database(ONTOLOGY_DB_MONGO_RU)


@pytest.mark.parametrize("coll", sorted(TRIPLETS_COLLS))
def test_triplets_db_collection_ru(triplets_db_mongo_ru, coll):
    assert coll in triplets_db_mongo_ru.list_collection_names()


@pytest.mark.parametrize("coll", sorted(ONTO_TRIPLETS_COLLS))
def test_onto_triplets_db_collection_ru(onto_triplets_db_mongo_ru, coll):
    assert coll in onto_triplets_db_mongo_ru.list_collection_names()


@pytest.mark.parametrize("coll", sorted(ONTOLOGY_COLLS))
def test_ontology_db_collection_ru(ontology_db_mongo_ru, coll):
    assert coll in ontology_db_mongo_ru.list_collection_names()


def test_triplets_vector_index_dims_ru(triplets_raw_ru):
    assert _vector_index_dimensions(
        triplets_raw_ru, "entity_aliases", "entity_aliases", "alias_text_embedding"
    ) == EMBEDDING_DIMS_RU
    assert _vector_index_dimensions(
        triplets_raw_ru, "property_aliases", "property_aliases", "alias_text_embedding"
    ) == EMBEDDING_DIMS_RU


def test_onto_triplets_vector_index_dims_ru(onto_raw_ru):
    assert _vector_index_dimensions(
        onto_raw_ru, "entity_aliases", "entity_aliases", "alias_text_embedding"
    ) == EMBEDDING_DIMS_RU


@pytest.mark.parametrize("collection,index_name", [
    ("entity_type_aliases", "entity_type_aliases"),
    ("property_aliases",    "property_aliases"),
])
def test_ontology_vector_index_dims_ru(ontology_raw_ru, collection, index_name):
    assert _vector_index_dimensions(
        ontology_raw_ru, collection, index_name, "alias_text_embedding"
    ) == EMBEDDING_DIMS_RU
