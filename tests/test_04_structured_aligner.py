"""StructuredAligner (structured_aligner.py) — MongoDB only (requires ontology DB)."""

import pytest
from conftest import timed

SID = "test_struct"
TRIPLET = {
    "subject": "Paris", "relation": "located in", "object": "France",
    "subject_type": "city", "object_type": "country",
}


def test_add_entity(structured_aligner_mongo):
    structured_aligner_mongo.add_entity("Paris", "city of light", "city", SID)


def test_add_triplets(structured_aligner_mongo):
    structured_aligner_mongo.add_triplets([TRIPLET.copy()], SID)


def test_add_initial_triplets(structured_aligner_mongo):
    structured_aligner_mongo.add_initial_triplets([TRIPLET.copy()], SID)


def test_add_filtered_triplets(structured_aligner_mongo):
    structured_aligner_mongo.add_filtered_triplets([TRIPLET.copy()], SID)


def test_add_ontology_filtered_triplets(structured_aligner_mongo):
    structured_aligner_mongo.add_ontology_filtered_triplets([TRIPLET.copy()], SID)


def test_retrieve_similar_entity_names(structured_aligner_mongo):
    # Seed two entities with different aliases so ranking is meaningful.
    structured_aligner_mongo.add_entity("Paris", "city of light", "city", SID)
    structured_aligner_mongo.add_entity("France", "French republic", "country", SID)

    results = timed(
        "StructuredAligner.retrieve_similar_entity_names",
        structured_aligner_mongo.retrieve_similar_entity_names,
        "city of light", k=3, sample_id=SID,
    )

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "at least one result expected"
    assert all(isinstance(r, dict) for r in results), "each result must be a dict"
    assert all("entity" in r for r in results), "each result must have an 'entity' key"
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
    assert all(
        r[0] in ("P276", "P706") for r in results
    ), "property_id must be 'P276' or 'P706'"


def _backend_label(aligner):
    return "mongo" if hasattr(aligner.db, "db") else "qdrant"


def test_retrieve_entity_by_type(structured_aligner_mongo):
    structured_aligner_mongo.add_entity("Paris", "city of light", "city", SID)

    results = timed(
        "StructuredAligner.retrieve_entity_by_type",
        structured_aligner_mongo.retrieve_entity_by_type,
        "Paris", "city", SID, k=3,
    )

    assert isinstance(results, dict), "result must be a dict"
    assert len(results) > 0, "at least one match expected"
    # Keys are aliases, values are canonical labels — 'Paris' must appear as a value.
    assert "Paris" in results.values(), (
        f"expected 'Paris' among retrieved labels, got {list(results.values())}"
    )


def test_retrieve_entity_type_labels(structured_aligner_mongo):
    results = structured_aligner_mongo.retrieve_entity_type_labels(["Q515", "Q6256"])

    assert isinstance(results, dict), "result must be a dict"
    assert "Q515" in results, "Q515 (city) must be present in ontology"
    assert "Q6256" in results, "Q6256 (country) must be present in ontology"
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
