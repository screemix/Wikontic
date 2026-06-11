"""StructuredAligner (structured_aligner.py) — MongoDB only (requires ontology DB)."""

import pytest
from conftest import timed

SAMPLE_ID = "test_struct"
TRIPLET = {
    "subject": "Paris", "relation": "located in", "object": "France",
    "subject_type": "city", "object_type": "country",
}


def _count_docs(aligner, collection, query):
    return len(aligner.triplets_db.match_documents(collection, query))


def _assert_single_doc(aligner, collection, query, expected_fields):
    docs = aligner.triplets_db.match_documents(collection, query)
    assert len(docs) == 1
    for key, value in expected_fields.items():
        assert docs[0].get(key) == value
    return docs[0]


def _assert_unique_values(values, name):
    assert len(values) == len(set(values)), (
        f"expected unique {name}, got duplicates in {values!r}"
    )


def test_add_entity(structured_aligner_mongo):
    sample_id = "test_struct_add_entity"
    label, alias, entity_type = "Paris", "city of light", "city"
    query = {
        "label": label,
        "entity_type": entity_type,
        "alias": alias,
        "sample_id": sample_id,
    }
    structured_aligner_mongo.add_entity(label, alias, entity_type, sample_id)
    _assert_single_doc(structured_aligner_mongo, "entity_aliases", query, query)
    structured_aligner_mongo.add_entity(label, alias, entity_type, sample_id)
    assert _count_docs(structured_aligner_mongo, "entity_aliases", query) == 1


def test_add_triplets(structured_aligner_mongo):
    sample_id = "test_struct_add_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    structured_aligner_mongo.add_triplets([triplet.copy()], sample_id)
    _assert_single_doc(structured_aligner_mongo, "triplets", query, query)
    structured_aligner_mongo.add_triplets([triplet.copy()], sample_id)
    assert _count_docs(structured_aligner_mongo, "triplets", query) == 1


def test_add_initial_triplets(structured_aligner_mongo):
    sample_id = "test_struct_add_initial_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    structured_aligner_mongo.add_initial_triplets([triplet.copy()], sample_id)
    _assert_single_doc(structured_aligner_mongo, "initial_triplets", query, query)
    structured_aligner_mongo.add_initial_triplets([triplet.copy()], sample_id)
    assert _count_docs(structured_aligner_mongo, "initial_triplets", query) == 1


def test_add_filtered_triplets(structured_aligner_mongo):
    sample_id = "test_struct_add_filtered_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    structured_aligner_mongo.add_filtered_triplets([triplet.copy()], sample_id)
    _assert_single_doc(structured_aligner_mongo, "filtered_triplets", query, query)
    structured_aligner_mongo.add_filtered_triplets([triplet.copy()], sample_id)
    assert _count_docs(structured_aligner_mongo, "filtered_triplets", query) == 1


def test_add_ontology_filtered_triplets(structured_aligner_mongo):
    sample_id = "test_struct_add_ontology_filtered_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    structured_aligner_mongo.add_ontology_filtered_triplets([triplet.copy()], sample_id)
    _assert_single_doc(
        structured_aligner_mongo, "ontology_filtered_triplets", query, query
    )
    structured_aligner_mongo.add_ontology_filtered_triplets([triplet.copy()], sample_id)
    assert _count_docs(
        structured_aligner_mongo, "ontology_filtered_triplets", query
    ) == 1


def test_retrieve_similar_entity_names(structured_aligner_mongo):
    # Seed two entities with different aliases so ranking is meaningful.
    structured_aligner_mongo.add_entity("Paris", "city of light", "city", SAMPLE_ID)
    structured_aligner_mongo.add_entity("France", "French republic", "country", SAMPLE_ID)

    results = timed(
        "StructuredAligner.retrieve_similar_entity_names",
        structured_aligner_mongo.retrieve_similar_entity_names,
        "city of light", k=3, sample_id=SAMPLE_ID,
    )

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "at least one result expected"
    assert all(isinstance(r, dict) for r in results), "each result must be a dict"
    assert all("entity" in r for r in results), "each result must have an 'entity' key"
    _assert_unique_values([r["entity"] for r in results], "entity labels")
    # The alias 'city of light' was added for 'Paris' — it must rank first.
    assert results[0]["entity"] == "Paris", (
        f"expected 'Paris' as top result, got {results[0]['entity']!r}"
    )


def test_retrieve_similar_entity_types(structured_aligner_mongo):
    results = timed(
        "StructuredAligner.retrieve_similar_entity_types",
        structured_aligner_mongo.retrieve_similar_entity_types,
        {"subject_type": "city", "object_type": "country"}, k=3,
    )
    subject_types, object_types = results

    assert isinstance(subject_types, list), "subject_types must be a list"
    assert isinstance(object_types, list), "object_types must be a list"
    assert len(subject_types) > 0, "at least one subject type expected"
    assert len(object_types) > 0, "at least one object type expected"
    # Results are Wikidata IDs (e.g. Q515 for city, Q6256 for country).
    assert all(isinstance(t, str) for t in subject_types)
    assert all(isinstance(t, str) for t in object_types)
    _assert_unique_values(subject_types, "subject entity types")
    _assert_unique_values(object_types, "object entity types")
    assert "Q515" in subject_types, (
        f"expected Q515 (city) among subject types, got {subject_types!r}"
    )
    assert "Q6256" in object_types, (
        f"expected Q6256 (country) among object types, got {object_types!r}"
    )


def test_retrieve_properties_for_entity_type(structured_aligner_mongo):
    results = timed(
        "StructuredAligner.retrieve_properties_for_entity_type",
        structured_aligner_mongo.retrieve_properties_for_entity_type,
        "is located in", object_types=["Q515"], subject_types=["Q6256"], k=3,
    )

    assert isinstance(results, list), "result must be a list"
    # Each element is a (property_id, direction) tuple.
    assert all(
        isinstance(r, tuple) and len(r) == 2 for r in results
    ), "each result must be a (property_id, direction) tuple"
    assert all(
        r[1] in ("direct", "inverse") for r in results
    ), "direction must be 'direct' or 'inverse'"
    _assert_unique_values(results, "property (id, direction) pairs")
    # P131 has the alias "is located in" and is valid for city (Q515) / country (Q6256).
    assert results[0][0] == "P131", (
        f"expected P131 as top property for 'is located in', got {results[0][0]!r}"
    )
    assert "P276" in [r[0] for r in results], (
        f"expected P276 (location) to be similar to 'is located in', got {results}"
    )


def test_retrieve_entity_by_type(structured_aligner_mongo):
    structured_aligner_mongo.add_entity("Paris", "city of light", "city", SAMPLE_ID)
    
    results = timed(
        "StructuredAligner.retrieve_entity_by_type",
        structured_aligner_mongo.retrieve_entity_by_type,
        "Paris", "city", SAMPLE_ID, k=3,
    )

    assert isinstance(results, dict), "result must be a dict"
    assert len(results) > 0, "at least one match expected"
    _assert_unique_values([(key, value) for key, value in results.items()], "entity (alias, label) pairs")
    # Keys are aliases, values are canonical labels — 'Paris' must appear as a value.
    assert "Paris" in results.values(), (
        f"expected 'Paris' among retrieved labels, got {list(results.values())}"
    )


def test_retrieve_entity_type_labels(structured_aligner_mongo):
    results = structured_aligner_mongo.retrieve_entity_type_labels(["Q515", "Q6256"])

    assert isinstance(results, dict), "result must be a dict"
    assert "Q515" in results, "Q515 (city) must be present in ontology"
    assert "Q6256" in results, "Q6256 (country) must be present in ontology"
    _assert_unique_values(list(results.keys()), "entity type ids")
    assert isinstance(results["Q515"], str) and len(results["Q515"]) > 0
    assert isinstance(results["Q6256"], str) and len(results["Q6256"]) > 0
    assert results['Q515'] == 'city'
    assert results['Q6256'] == 'country'



def test_retrieve_entity_type_hierarchy(structured_aligner_mongo):
    results = structured_aligner_mongo.retrieve_entity_type_hierarchy("city")

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "hierarchy must have at least the type itself"
    assert all(isinstance(t, str) for t in results), "each entry must be a Wikidata ID string"
    # Q515 is the Wikidata ID for 'city' — must be in its own hierarchy.
    assert "Q515" in results, f"Q515 (city) expected in hierarchy, got {results}"
