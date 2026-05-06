"""DB setup: collections, regular indexes, vector (Atlas Search) indexes."""

import pytest
from pymongo.mongo_client import MongoClient

from conftest import MONGO_URI, TRIPLETS_DB, ONTO_TRIPLETS_DB, ONTOLOGY_DB_MONGO, ONTOLOGY_DB_QDRANT


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


# ── collections ────────────────────────────────────────────────────────────────

TRIPLETS_COLLS      = {"entity_aliases", "triplets",
                        "initial_triplets", "filtered_triplets"}
ONTO_TRIPLETS_COLLS = {"entity_aliases", "triplets", "initial_triplets",
                        "filtered_triplets", "ontology_filtered_triplets"}
ONTOLOGY_COLLS      = {"entity_types", "entity_type_aliases", "properties",
                        "property_aliases", "entity_aliases", "triplets"}


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
def triplets_raw(mongo_client):
    return mongo_client.get_database(TRIPLETS_DB)


@pytest.fixture(scope="module")
def onto_raw(mongo_client):
    return mongo_client.get_database(ONTO_TRIPLETS_DB)


@pytest.fixture(scope="module")
def ontology_raw(mongo_client):
    return mongo_client.get_database(ONTOLOGY_DB_MONGO)


@pytest.mark.parametrize("fields", [
    ["entity_type", "sample_id"],
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
    ("entity_aliases",      ["entity_type", "sample_id"]),
    ("entity_aliases",      ["label"]),
    ("triplets",            ["sample_id"]),
])
def test_ontology_regular_indexes(ontology_raw, collection, fields):
    assert _has_index(ontology_raw, collection, *fields)


# ── vector (Atlas Search) indexes ─────────────────────────────────────────────

def test_triplets_vector_indexes(triplets_raw):
    assert "entity_aliases" in _vector_index_names(triplets_raw, "entity_aliases")


def test_onto_triplets_vector_index(onto_raw):
    assert "entity_aliases" in _vector_index_names(onto_raw, "entity_aliases")


@pytest.mark.parametrize("collection,index_name", [
    ("entity_type_aliases", "entity_type_aliases"),
    ("property_aliases",    "property_aliases"),
])
def test_ontology_vector_indexes(ontology_raw, collection, index_name):
    assert index_name in _vector_index_names(ontology_raw, collection)
