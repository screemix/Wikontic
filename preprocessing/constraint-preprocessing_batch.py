import argparse
import json
import logging
import re
import time
from pathlib import Path

from SPARQLWrapper import JSON, SPARQLWrapper
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


logger = logging.getLogger("ConstraintPreprocessingBatch")
logging.basicConfig(level=logging.INFO)

CHINESE_JAPANESE_PATTERN = re.compile(
    r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]"
)
DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
QUANTITY_ENTITY_ID = "Q309314"
POINT_IN_TIME_ENTITY_ID = "Q186408"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ontology_mappings_dir",
        type=str,
        default="src/wikontic/utils/ontology_mappings_ru_en",
    )
    parser.add_argument("--wikidata_endpoint", type=str, default=DEFAULT_ENDPOINT)
    parser.add_argument("--language", type=str, default="ru")
    parser.add_argument("--fallback_language", type=str, default="en")
    parser.add_argument("--property_batch_size", type=int, default=200)
    parser.add_argument("--hierarchy_batch_size", type=int, default=100)
    parser.add_argument("--label_batch_size", type=int, default=50)
    parser.add_argument("--entity_info_batch_size", type=int, default=100)
    parser.add_argument("--entity_alias_batch_size", type=int, default=100)
    parser.add_argument("--constraint_sleep_seconds", type=float, default=0.5)
    parser.add_argument("--hierarchy_sleep_seconds", type=float, default=0.3)
    parser.add_argument("--label_sleep_seconds", type=float, default=0.2)
    parser.add_argument("--entity_info_sleep_seconds", type=float, default=0.2)
    parser.add_argument("--entity_alias_sleep_seconds", type=float, default=0.2)
    return parser.parse_args()


def chunks(items, batch_size):
    items = list(items)
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def make_sparql_client(endpoint, user_agent):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", user_agent)
    return sparql


def property_sort_key(value):
    return int(value[1:])


def sort_mapping(mapping):
    return {
        key: mapping[key]
        for key in sorted(mapping.keys(), key=property_sort_key)
    }


def dump_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_property_metadata(endpoint, language, wikidata_languages):
    prop_to_label = {}
    prop_to_data_type = {}
    prop_to_aliases = {}

    logger.info("Collecting property metadata from Wikidata SPARQL endpoint...")

    sparql = make_sparql_client(endpoint, "wikontic-property-mapping/1.0")
    query = f"""
    SELECT ?property ?propertyLabel ?typeLabel ?alias WHERE {{
      ?property a wikibase:Property .
      ?property wikibase:propertyType ?type .

      VALUES ?type {{ wikibase:WikibaseItem wikibase:Quantity wikibase:Time }}

      BIND(
        IF(?type = wikibase:WikibaseItem, "Item",
          IF(?type = wikibase:Quantity, "Quantity",
            IF(?type = wikibase:Time, "Point in time", "Unknown")
          )
        ) AS ?typeLabel
      )

      OPTIONAL {{
        ?property skos:altLabel ?alias .
        FILTER(LANG(?alias) = "{language}")
      }}

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{wikidata_languages}". }}
    }}
    """

    sparql.setQuery(query)
    results = sparql.query().convert()

    for row in tqdm(results["results"]["bindings"], desc="properties"):
        prop = row["property"]["value"].split("/")[-1]
        prop_to_label[prop] = row.get("propertyLabel", {}).get("value", "No label")
        prop_to_data_type[prop] = row.get("typeLabel", {}).get("value", "Unknown")
        prop_to_aliases.setdefault(prop, [])

        if "alias" in row:
            prop_to_aliases[prop].append(row["alias"]["value"])

    prop_to_label = sort_mapping(prop_to_label)
    prop_to_data_type = sort_mapping(prop_to_data_type)
    prop_to_aliases = sort_mapping(prop_to_aliases)

    logger.info("Collected metadata for %s properties", len(prop_to_label))
    logger.info("Property data types: %s", sorted(set(prop_to_data_type.values())))

    alias_set = set()
    alias_list = []
    for aliases in prop_to_aliases.values():
        alias_set.update(aliases)
        alias_list.extend(aliases)
    logger.info("Unique property aliases: %s", len(alias_set))
    logger.info("Total property aliases: %s", len(alias_list))

    return prop_to_label, prop_to_data_type, prop_to_aliases


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(7))
def get_constraints_batch(endpoint, property_ids):
    sparql = make_sparql_client(endpoint, "constraints-collector/1.0")
    values = " ".join(f"wd:{prop_id}" for prop_id in property_ids)

    query = f"""
    SELECT ?property ?constraintType ?entity WHERE {{
      VALUES ?property {{ {values} }}

      ?property p:P2302 ?statement .
      ?statement ps:P2302 ?constraintEntity .

      VALUES ?constraintEntity {{
        wd:Q21510865
        wd:Q21503250
      }}

      ?statement pq:P2308 ?entity .

      BIND(
        IF(
          ?constraintEntity = wd:Q21510865,
          "Value-type constraint",
          "Subject type constraint"
        ) AS ?constraintType
      )
    }}
    """

    sparql.setQuery(query)
    results = sparql.query().convert()

    batch_constraints = {
        prop_id: {
            "Value-type constraint": [],
            "Subject type constraint": [],
        }
        for prop_id in property_ids
    }

    for row in results["results"]["bindings"]:
        prop_id = row["property"]["value"].split("/")[-1]
        batch_constraints[prop_id][row["constraintType"]["value"]].append(
            row["entity"]["value"].split("/")[-1]
        )

    return batch_constraints


def collect_property_constraints(
    endpoint,
    prop_to_label,
    prop_to_data_type,
    batch_size,
    sleep_seconds,
):
    constraint_dict = {}
    property_ids = list(prop_to_label.keys())

    logger.info("Collecting property constraints...")

    for batch in tqdm(
        list(chunks(property_ids, batch_size)),
        desc="constraint batches",
    ):
        constraint_dict.update(get_constraints_batch(endpoint, batch))
        time.sleep(sleep_seconds)

    without_constraints = [
        prop_id
        for prop_id, constraint in constraint_dict.items()
        if not constraint["Value-type constraint"] and not constraint["Subject type constraint"]
    ]
    logger.info("Properties without explicit constraints: %s", len(without_constraints))

    quantity_props = [
        prop_id
        for prop_id in without_constraints
        if prop_to_data_type[prop_id] == "Quantity"
    ]
    time_props = [
        prop_id
        for prop_id in without_constraints
        if prop_to_data_type[prop_id] == "Point in time"
    ]
    other_props = [
        prop_id
        for prop_id in without_constraints
        if prop_to_data_type[prop_id] not in {"Quantity", "Point in time"}
    ]
    logger.info("Constraint-free quantity properties: %s", len(quantity_props))
    logger.info("Constraint-free time properties: %s", len(time_props))
    logger.info("Constraint-free other properties: %s", len(other_props))

    for prop_id, constraint in constraint_dict.items():
        if prop_to_data_type[prop_id] == "Point in time":
            constraint["Value-type constraint"].append(POINT_IN_TIME_ENTITY_ID)
        elif prop_to_data_type[prop_id] == "Quantity":
            constraint["Value-type constraint"].append(QUANTITY_ENTITY_ID)

    for constraint in constraint_dict.values():
        if not constraint["Value-type constraint"]:
            constraint["Value-type constraint"] = ["ANY"]
        if not constraint["Subject type constraint"]:
            constraint["Subject type constraint"] = ["ANY"]

    return sort_mapping(constraint_dict)


def collect_constrained_entities(constraint_dict):
    entities = set()
    for constraint in constraint_dict.values():
        for entity_ids in constraint.values():
            entities.update(entity_id for entity_id in entity_ids if entity_id != "ANY")
    entities = sorted(entities)
    logger.info("Unique entity types in constraints: %s", len(entities))
    return entities


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(7))
def get_subclass_hierarchy_batch(endpoint, entity_ids):
    sparql = make_sparql_client(endpoint, "hierarchy-collector/1.0")
    values = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)

    query = f"""
    SELECT DISTINCT ?entity ?ancestor WHERE {{
      VALUES ?entity {{ {values} }}

      {{
        ?entity wdt:P31/wdt:P279* ?ancestor .
      }}
      UNION
      {{
        ?entity wdt:P279* ?ancestor .
      }}
    }}
    """

    sparql.setQuery(query)
    results = sparql.query().convert()

    batch_hierarchy = {entity_id: [] for entity_id in entity_ids}
    for row in results["results"]["bindings"]:
        entity_id = row["entity"]["value"].split("/")[-1]
        batch_hierarchy[entity_id].append(row["ancestor"]["value"].split("/")[-1])

    return batch_hierarchy


def collect_entity_hierarchy(endpoint, entities, batch_size, sleep_seconds):
    entity_to_hierarchy = {}

    logger.info("Collecting entity hierarchies...")

    for batch in tqdm(list(chunks(entities, batch_size)), desc="hierarchy batches"):
        entity_to_hierarchy.update(get_subclass_hierarchy_batch(endpoint, batch))
        time.sleep(sleep_seconds)

    entity_set = set(entities)
    for entity_id, ancestors in entity_to_hierarchy.items():
        entity_to_hierarchy[entity_id] = [ancestor for ancestor in ancestors if ancestor in entity_set]

    logger.info("Entity hierarchies collected: %s", len(entity_to_hierarchy))
    return entity_to_hierarchy


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(7))
def fetch_labels_batch(endpoint, entity_ids, wikidata_languages):
    sparql = make_sparql_client(endpoint, "wikontic-label-collector/1.0")
    values = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)

    query = f"""
    SELECT ?entity ?entityLabel WHERE {{
      VALUES ?entity {{ {values} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{wikidata_languages}". }}
    }}
    """

    sparql.setQuery(query)
    results = sparql.query().convert()

    batch_labels = {entity_id: "No label" for entity_id in entity_ids}
    for row in results["results"]["bindings"]:
        entity_id = row["entity"]["value"].split("/")[-1]
        batch_labels[entity_id] = row.get("entityLabel", {}).get("value", "No label")

    return batch_labels


def collect_entity_labels(endpoint, entities, wikidata_languages, batch_size, sleep_seconds):
    entity_to_label = {}

    logger.info("Collecting entity labels...")

    for batch in tqdm(list(chunks(entities, batch_size)), desc="label batches"):
        entity_to_label.update(fetch_labels_batch(endpoint, batch, wikidata_languages))
        time.sleep(sleep_seconds)

    logger.info(
        "Entity labels collected: %s entities, %s unique labels",
        len(entity_to_label),
        len(set(entity_to_label.values())),
    )
    return entity_to_label


def build_label_to_entities(entity_to_label):
    label_to_entities = {}
    for entity_id, label in entity_to_label.items():
        label_to_entities.setdefault(label, []).append(entity_id)
    return label_to_entities


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(7))
def get_entity_info_batch(endpoint, entity_ids, language, fallback_language):
    sparql = make_sparql_client(endpoint, "wikontic-entity-info-collector/1.0")
    values = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)

    query = f"""
    SELECT ?entity ?entityLabel ?entityDescription WHERE {{
      VALUES ?entity {{ {values} }}

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{language},{fallback_language}".
      }}

      OPTIONAL {{
        ?entity schema:description ?entityDescription .
        FILTER(LANG(?entityDescription) IN ("{language}", "{fallback_language}"))
      }}
    }}
    """

    sparql.setQuery(query)
    results = sparql.query().convert()

    entity_info = {
        entity_id: {"label": None, "description": None}
        for entity_id in entity_ids
    }
    for row in results["results"]["bindings"]:
        entity_id = row["entity"]["value"].split("/")[-1]
        entity_info[entity_id] = {
            "label": row.get("entityLabel", {}).get("value"),
            "description": row.get("entityDescription", {}).get("value"),
        }

    return entity_info


def disambiguate_duplicate_labels(
    endpoint,
    entity_to_label,
    language,
    fallback_language,
    batch_size,
    sleep_seconds,
):
    logger.info("Checking for duplicate entity labels...")

    label_to_entities = build_label_to_entities(entity_to_label)
    ambiguous_entities = sorted(
        {
            entity_id
            for entity_ids in label_to_entities.values()
            if len(entity_ids) > 1
            for entity_id in entity_ids
        }
    )

    if not ambiguous_entities:
        return entity_to_label

    entity_to_info = {}
    for batch in tqdm(list(chunks(ambiguous_entities, batch_size)), desc="entity info batches"):
        entity_to_info.update(
            get_entity_info_batch(endpoint, batch, language, fallback_language)
        )
        time.sleep(sleep_seconds)

    for entity_id, info in entity_to_info.items():
        label = info.get("label")
        description = info.get("description")
        if label and description:
            entity_to_label[entity_id] = f"{label} ({description})"
        elif label:
            entity_to_label[entity_id] = label

    logger.info("Disambiguated %s entities with duplicate labels", len(ambiguous_entities))
    return entity_to_label


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(7))
def get_entity_aliases_batch(endpoint, entity_ids, language):
    sparql = make_sparql_client(endpoint, "wikontic-entity-alias-collector/1.0")
    values = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)

    query = f"""
    SELECT ?entity ?alias WHERE {{
      VALUES ?entity {{ {values} }}
      ?entity skos:altLabel ?alias .
      FILTER(LANG(?alias) = "{language}")
    }}
    """

    sparql.setQuery(query)
    results = sparql.query().convert()

    batch_aliases = {entity_id: [] for entity_id in entity_ids}
    for row in results["results"]["bindings"]:
        entity_id = row["entity"]["value"].split("/")[-1]
        alias = row["alias"]["value"]
        if not CHINESE_JAPANESE_PATTERN.search(alias):
            batch_aliases[entity_id].append(alias)

    return batch_aliases


def collect_entity_aliases(endpoint, entity_ids, language, batch_size, sleep_seconds):
    entity_to_aliases = {}

    logger.info("Collecting entity aliases...")

    for batch in tqdm(list(chunks(entity_ids, batch_size)), desc="entity alias batches"):
        entity_to_aliases.update(get_entity_aliases_batch(endpoint, batch, language))
        time.sleep(sleep_seconds)

    logger.info("Entity aliases collected: %s", len(entity_to_aliases))
    return entity_to_aliases


def build_inverse_constraint_mappings(constraint_dict, prop_to_data_type):
    subj_to_prop_constraints = {"<ANY SUBJECT>": []}
    obj_to_prop_constraint = {
        "<ANY OBJECT>": [],
        QUANTITY_ENTITY_ID: [],
        POINT_IN_TIME_ENTITY_ID: [],
    }

    for prop_id, constraint in constraint_dict.items():
        data_type = prop_to_data_type[prop_id]

        if data_type == "Point in time":
            obj_to_prop_constraint[POINT_IN_TIME_ENTITY_ID].append(prop_id)
        elif data_type == "Quantity":
            obj_to_prop_constraint[QUANTITY_ENTITY_ID].append(prop_id)
        elif constraint["Value-type constraint"] == ["ANY"]:
            obj_to_prop_constraint["<ANY OBJECT>"].append(prop_id)
        else:
            for entity_id in constraint["Value-type constraint"]:
                obj_to_prop_constraint.setdefault(entity_id, []).append(prop_id)

        if constraint["Subject type constraint"] == ["ANY"]:
            subj_to_prop_constraints["<ANY SUBJECT>"].append(prop_id)
        else:
            for entity_id in constraint["Subject type constraint"]:
                subj_to_prop_constraints.setdefault(entity_id, []).append(prop_id)

    logger.info("Subject constraint buckets: %s", len(subj_to_prop_constraints))
    logger.info("Object constraint buckets: %s", len(obj_to_prop_constraint))
    return subj_to_prop_constraints, obj_to_prop_constraint


def resolve_output_dir(raw_output_dir):
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(raw_output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.ontology_mappings_dir)
    wikidata_languages = f"{args.language},{args.fallback_language}"

    logger.info("Writing outputs to %s", output_dir)

    prop_to_label, prop_to_data_type, prop_to_aliases = collect_property_metadata(
        args.wikidata_endpoint,
        args.language,
        wikidata_languages,
    )

    dump_json(output_dir / "prop2label.json", prop_to_label)
    dump_json(output_dir / "prop2aliases.json", prop_to_aliases)
    dump_json(output_dir / "prop2data_type.json", prop_to_data_type)

    constraint_dict = collect_property_constraints(
        args.wikidata_endpoint,
        prop_to_label,
        prop_to_data_type,
        args.property_batch_size,
        args.constraint_sleep_seconds,
    )
    dump_json(output_dir / "prop2constraints.json", constraint_dict)

    entities = collect_constrained_entities(constraint_dict)
    entity_to_hierarchy = collect_entity_hierarchy(
        args.wikidata_endpoint,
        entities,
        args.hierarchy_batch_size,
        args.hierarchy_sleep_seconds,
    )
    dump_json(output_dir / "entity_type2hierarchy.json", entity_to_hierarchy)

    entity_to_label = collect_entity_labels(
        args.wikidata_endpoint,
        entities,
        wikidata_languages,
        args.label_batch_size,
        args.label_sleep_seconds,
    )
    entity_to_label = disambiguate_duplicate_labels(
        args.wikidata_endpoint,
        entity_to_label,
        args.language,
        args.fallback_language,
        args.entity_info_batch_size,
        args.entity_info_sleep_seconds,
    )
    dump_json(output_dir / "entity_type2label.json", entity_to_label)

    entity_to_aliases = collect_entity_aliases(
        args.wikidata_endpoint,
        list(entity_to_label.keys()),
        args.language,
        args.entity_alias_batch_size,
        args.entity_alias_sleep_seconds,
    )
    dump_json(output_dir / "entity_type2aliases.json", entity_to_aliases)

    subj_to_prop_constraints, obj_to_prop_constraint = build_inverse_constraint_mappings(
        constraint_dict,
        prop_to_data_type,
    )
    dump_json(output_dir / "subj_constraint2prop.json", subj_to_prop_constraints)
    dump_json(output_dir / "obj_constraint2prop.json", obj_to_prop_constraint)


if __name__ == "__main__":
    main()
