"""StructuredAligner (structured_aligner.py) for language='ru' — MongoDB only.

Mirrors test_04_structured_aligner.py but exercises the FRIDA embedding model
and the ru ontology mappings (ontology_mappings_ru_en) instead of Contriever/en.
"""

import pytest
from conftest import timed

SAMPLE_ID = "test_struct_ru"
TRIPLET = {
    "subject": "Париж", "relation": "находится в", "object": "Франция",
    "subject_type": "город", "object_type": "страна",
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


def test_add_entity(structured_aligner_mongo_ru):
    sample_id = "test_struct_ru_add_entity"
    label, alias, entity_type = "Париж", "город огней", "город"
    query = {
        "label": label,
        "entity_type": entity_type,
        "alias": alias,
        "sample_id": sample_id,
    }
    structured_aligner_mongo_ru.add_entity(label, alias, entity_type, sample_id)
    _assert_single_doc(structured_aligner_mongo_ru, "entity_aliases", query, query)
    structured_aligner_mongo_ru.add_entity(label, alias, entity_type, sample_id)
    assert _count_docs(structured_aligner_mongo_ru, "entity_aliases", query) == 1


def test_add_triplets(structured_aligner_mongo_ru):
    sample_id = "test_struct_ru_add_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    structured_aligner_mongo_ru.add_triplets([triplet.copy()], sample_id)
    _assert_single_doc(structured_aligner_mongo_ru, "triplets", query, query)
    structured_aligner_mongo_ru.add_triplets([triplet.copy()], sample_id)
    assert _count_docs(structured_aligner_mongo_ru, "triplets", query) == 1


def test_add_initial_triplets(structured_aligner_mongo_ru):
    sample_id = "test_struct_ru_add_initial_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    structured_aligner_mongo_ru.add_initial_triplets([triplet.copy()], sample_id)
    _assert_single_doc(structured_aligner_mongo_ru, "initial_triplets", query, query)
    structured_aligner_mongo_ru.add_initial_triplets([triplet.copy()], sample_id)
    assert _count_docs(structured_aligner_mongo_ru, "initial_triplets", query) == 1


def test_add_filtered_triplets(structured_aligner_mongo_ru):
    sample_id = "test_struct_ru_add_filtered_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    structured_aligner_mongo_ru.add_filtered_triplets([triplet.copy()], sample_id)
    _assert_single_doc(structured_aligner_mongo_ru, "filtered_triplets", query, query)
    structured_aligner_mongo_ru.add_filtered_triplets([triplet.copy()], sample_id)
    assert _count_docs(structured_aligner_mongo_ru, "filtered_triplets", query) == 1


def test_add_ontology_filtered_triplets(structured_aligner_mongo_ru):
    sample_id = "test_struct_ru_add_ontology_filtered_triplets"
    triplet = TRIPLET.copy()
    query = {**triplet, "sample_id": sample_id}
    structured_aligner_mongo_ru.add_ontology_filtered_triplets([triplet.copy()], sample_id)
    _assert_single_doc(
        structured_aligner_mongo_ru, "ontology_filtered_triplets", query, query
    )
    structured_aligner_mongo_ru.add_ontology_filtered_triplets([triplet.copy()], sample_id)
    assert _count_docs(
        structured_aligner_mongo_ru, "ontology_filtered_triplets", query
    ) == 1


def test_retrieve_similar_entity_names(structured_aligner_mongo_ru):
    # Seed two entities with different aliases so ranking is meaningful.
    structured_aligner_mongo_ru.add_entity("Париж", "город огней", "город", SAMPLE_ID)
    structured_aligner_mongo_ru.add_entity("Франция", "Французская Республика", "страна", SAMPLE_ID)

    results = timed(
        "StructuredAligner(ru).retrieve_similar_entity_names",
        structured_aligner_mongo_ru.retrieve_similar_entity_names,
        "город огней", k=3, sample_id=SAMPLE_ID,
    )

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "at least one result expected"
    assert all(isinstance(r, dict) for r in results), "each result must be a dict"
    assert all("entity" in r for r in results), "each result must have an 'entity' key"
    _assert_unique_values([r["entity"] for r in results], "entity labels")
    # The alias 'город огней' was added for 'Париж' — it must rank first.
    assert results[0]["entity"] == "Париж", (
        f"expected 'Париж' as top result, got {results[0]['entity']!r}"
    )


def test_retrieve_similar_entity_types(structured_aligner_mongo_ru):
    results = timed(
        "StructuredAligner(ru).retrieve_similar_entity_types",
        structured_aligner_mongo_ru.retrieve_similar_entity_types,
        {"subject_type": "город", "object_type": "страна"}, k=3,
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
        f"expected Q515 (город) among subject types, got {subject_types!r}"
    )
    assert "Q6256" in object_types, (
        f"expected Q6256 (страна) among object types, got {object_types!r}"
    )


def test_retrieve_properties_for_entity_type(structured_aligner_mongo_ru):
    results = timed(
        "StructuredAligner(ru).retrieve_properties_for_entity_type",
        structured_aligner_mongo_ru.retrieve_properties_for_entity_type,
        "находится в административно-территориальной единице",
        object_types=["Q515"], subject_types=["Q6256"], k=3,
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
    # P131 has this exact alias and is valid for город (Q515) / страна (Q6256).
    assert results[0][0] == "P131", (
        f"expected P131 as top property, got {results[0][0]!r}"
    )
    # P150 ("contains administrative territorial entity") is P131's structural
    # inverse and is the next closest candidate valid for this type pair.
    assert "P150" in [r[0] for r in results], (
        f"expected P150 (contains administrative territorial entity) to be similar, got {results}"
    )


def test_retrieve_entity_by_type(structured_aligner_mongo_ru):
    structured_aligner_mongo_ru.add_entity("Париж", "город огней", "город", SAMPLE_ID)
    results = timed(
        "StructuredAligner(ru).retrieve_entity_by_type",
        structured_aligner_mongo_ru.retrieve_entity_by_type,
        "Париж", "город", SAMPLE_ID, k=3,
    )

    assert isinstance(results, dict), "result must be a dict"
    assert len(results) > 0, "at least one match expected"
    _assert_unique_values([(key, value) for key, value in results.items()], "entity (alias, label) pairs")
    # Keys are aliases, values are canonical labels — 'Париж' must appear as a value.
    assert "Париж" in results.values(), (
        f"expected 'Париж' among retrieved labels, got {list(results.values())}"
    )


def test_retrieve_entity_type_labels(structured_aligner_mongo_ru):
    results = structured_aligner_mongo_ru.retrieve_entity_type_labels(["Q515", "Q6256"])

    assert isinstance(results, dict), "result must be a dict"
    assert "Q515" in results, "Q515 (город) must be present in ontology"
    assert "Q6256" in results, "Q6256 (страна) must be present in ontology"
    _assert_unique_values(list(results.keys()), "entity type ids")
    assert isinstance(results["Q515"], str) and len(results["Q515"]) > 0
    assert isinstance(results["Q6256"], str) and len(results["Q6256"]) > 0
    assert results["Q515"] == "город"
    assert results["Q6256"] == "страна"


def test_retrieve_entity_type_hierarchy(structured_aligner_mongo_ru):
    results = structured_aligner_mongo_ru.retrieve_entity_type_hierarchy("город")

    assert isinstance(results, list), "result must be a list"
    assert len(results) > 0, "hierarchy must have at least the type itself"
    assert all(isinstance(t, str) for t in results), "each entry must be a Wikidata ID string"
    # Q515 is the Wikidata ID for 'город' — must be in its own hierarchy.
    assert "Q515" in results, f"Q515 (город) expected in hierarchy, got {results}"
