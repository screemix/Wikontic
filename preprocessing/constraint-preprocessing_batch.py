from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from SPARQLWrapper import JSON, SPARQLWrapper
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
import urllib.request


logger = logging.getLogger("ConstraintPreprocessingBatch")
logging.basicConfig(level=logging.INFO)

CHINESE_JAPANESE_PATTERN = re.compile(
    r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]"
)
DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
QUANTITY_ENTITY_ID = "Q309314"
POINT_IN_TIME_ENTITY_ID = "Q186408"


@dataclass
class MappingStats:
    summary_lines: list[str] = field(default_factory=list)
    properties_without_label: list[str] = field(default_factory=list)
    entities_without_label: list[str] = field(default_factory=list)

    def info(self, message: str, *args) -> None:
        line = message % args if args else message
        logger.info(line)
        self.summary_lines.append(line)

    def warn_property_without_label(self, property_id: str) -> None:
        if property_id in self.properties_without_label:
            return
        logger.warning(
            "Property %s has no retrieved label; excluding from mappings",
            property_id,
        )
        self.properties_without_label.append(property_id)

    def warn_entity_without_label(self, entity_id: str) -> None:
        if entity_id in self.entities_without_label:
            return
        logger.warning(
            "Entity %s has no retrieved label; excluding from mappings",
            entity_id,
        )
        self.entities_without_label.append(entity_id)


def binding_label(row: dict, key: str) -> str | None:
    if key not in row:
        return None
    value = row.get(key, {}).get("value")
    if value is None:
        return None
    label = str(value).strip()
    return label if label else None


def filter_hierarchy_to_labeled_entities(
    entity_to_hierarchy: dict,
    labeled_entity_ids: set[str],
) -> dict:
    filtered = {}
    for entity_id in entity_to_hierarchy:
        if entity_id not in labeled_entity_ids:
            continue
        filtered[entity_id] = [
            ancestor_id
            for ancestor_id in entity_to_hierarchy[entity_id]
            if ancestor_id in labeled_entity_ids
        ]
    return filtered


def filter_inverse_constraint_mappings(
    subj_to_prop_constraints: dict,
    obj_to_prop_constraint: dict,
    labeled_entity_ids: set[str],
) -> tuple[dict, dict]:
    special_keys = {"<ANY SUBJECT>", "<ANY OBJECT>"}
    filtered_subj = {
        key: subj_to_prop_constraints[key]
        for key in subj_to_prop_constraints
        if key in special_keys or key in labeled_entity_ids
    }
    filtered_obj = {
        key: obj_to_prop_constraint[key]
        for key in obj_to_prop_constraint
        if key in special_keys or key in labeled_entity_ids
    }
    return filtered_subj, filtered_obj


def strip_unlabeled_entities_from_constraints(
    constraint_dict: dict,
    labeled_entity_ids: set[str],
) -> dict:
    filtered: dict = {}
    for prop_id, constraint in constraint_dict.items():
        filtered[prop_id] = {
            "Value-type constraint": [
                entity_id
                for entity_id in constraint["Value-type constraint"]
                if entity_id == "ANY" or entity_id in labeled_entity_ids
            ],
            "Subject type constraint": [
                entity_id
                for entity_id in constraint["Subject type constraint"]
                if entity_id == "ANY" or entity_id in labeled_entity_ids
            ],
        }
    return sort_mapping(filtered)


def write_summary_files(output_dir: Path, stats: MappingStats) -> None:
    dump_json(
        output_dir / "properties_without_label.json",
        sorted(stats.properties_without_label, key=property_sort_key),
    )
    dump_json(
        output_dir / "entities_without_label.json",
        sorted(stats.entities_without_label, key=property_sort_key),
    )

    summary_path = output_dir / "summary_collected_mappings.txt"
    extra_lines = [
        "",
        "Items excluded due to missing labels",
        f"Properties without label: {len(stats.properties_without_label)}",
        f"Entities without label: {len(stats.entities_without_label)}",
    ]
    summary_path.write_text(
        "\n".join(stats.summary_lines + extra_lines) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote summary to %s", summary_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikidata_endpoint", type=str, default=DEFAULT_ENDPOINT)
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--language", type=str, default="en")
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


def make_sparql_client(endpoint, user_agent, proxy=None):
    if proxy:
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy,
            "https": proxy,
        })
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)

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

@retry(wait=wait_random_exponential(multiplier=1, max=60))
def collect_property_metadata(endpoint, language, wikidata_languages, stats: MappingStats):
    prop_to_label = {}
    prop_to_data_type = {}
    prop_to_aliases = {}

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
        label = binding_label(row, "propertyLabel")
        if label is None:
            stats.warn_property_without_label(prop)
            continue

        prop_to_label[prop] = label
        prop_to_data_type[prop] = row.get("typeLabel", {}).get("value", "Unknown")
        prop_to_aliases.setdefault(prop, [])

        if "alias" in row:
            prop_to_aliases[prop].append(row["alias"]["value"])

    prop_to_label = sort_mapping(prop_to_label)
    prop_to_data_type = sort_mapping(prop_to_data_type)
    prop_to_aliases = sort_mapping(prop_to_aliases)

    stats.info("Collected metadata for %s properties with labels", len(prop_to_label))
    stats.info("Property data types: %s", sorted(set(prop_to_data_type.values())))

    alias_set = set()
    alias_list = []
    for aliases in prop_to_aliases.values():
        alias_set.update(aliases)
        alias_list.extend(aliases)
    stats.info("Unique property aliases: %s", len(alias_set))
    stats.info("Total property aliases: %s", len(alias_list))

    return prop_to_label, prop_to_data_type, prop_to_aliases


@retry(wait=wait_random_exponential(multiplier=1, max=60))
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
    stats: MappingStats,
):
    constraint_dict = {}
    property_ids = list(prop_to_label.keys())

    stats.info("Collecting property constraints...")

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
    stats.info("Properties without explicit constraints: %s", len(without_constraints))

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
    stats.info("Constraint-free quantity properties: %s", len(quantity_props))
    stats.info("Constraint-free time properties: %s", len(time_props))
    stats.info("Constraint-free other properties: %s", len(other_props))

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


def collect_constrained_entities(constraint_dict, stats: MappingStats):
    entities = set()
    for constraint in constraint_dict.values():
        for entity_ids in constraint.values():
            entities.update(entity_id for entity_id in entity_ids if entity_id != "ANY")
    entities = sorted(entities)
    stats.info("Unique entity types in constraints: %s", len(entities))
    return entities


@retry(wait=wait_random_exponential(multiplier=1, max=60))
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


def collect_entity_hierarchy(endpoint, entities, batch_size, sleep_seconds, stats: MappingStats):
    entity_to_hierarchy = {}

    stats.info("Collecting entity hierarchies...")

    for batch in tqdm(list(chunks(entities, batch_size)), desc="hierarchy batches"):
        entity_to_hierarchy.update(get_subclass_hierarchy_batch(endpoint, batch))
        time.sleep(sleep_seconds)

    entity_set = set(entities)
    for entity_id, ancestors in entity_to_hierarchy.items():
        entity_to_hierarchy[entity_id] = [ancestor for ancestor in ancestors if ancestor in entity_set]

    stats.info("Entity hierarchies collected: %s", len(entity_to_hierarchy))
    return entity_to_hierarchy


@retry(wait=wait_random_exponential(multiplier=1, max=60))
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

    batch_labels: dict[str, str] = {}
    for row in results["results"]["bindings"]:
        entity_id = row["entity"]["value"].split("/")[-1]
        label = binding_label(row, "entityLabel")
        if label is not None:
            batch_labels[entity_id] = label

    return batch_labels


def collect_entity_labels(
    endpoint,
    entities,
    wikidata_languages,
    batch_size,
    sleep_seconds,
    stats: MappingStats,
):
    entity_to_label = {}

    stats.info("Collecting entity labels...")

    for batch in tqdm(list(chunks(entities, batch_size)), desc="label batches"):
        entity_to_label.update(fetch_labels_batch(endpoint, batch, wikidata_languages))
        time.sleep(sleep_seconds)

    for entity_id in entities:
        if entity_id not in entity_to_label:
            stats.warn_entity_without_label(entity_id)

    stats.info(
        "Entity labels collected: %s entities with labels, %s unique labels",
        len(entity_to_label),
        len(set(entity_to_label.values())),
    )
    return entity_to_label


def build_label_to_entities(entity_to_label):
    label_to_entities = {}
    for entity_id, label in entity_to_label.items():
        label_to_entities.setdefault(label, []).append(entity_id)
    return label_to_entities


@retry(wait=wait_random_exponential(multiplier=1, max=60))
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
    stats: MappingStats,
):
    stats.info("Checking for duplicate entity labels...")

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

    stats.info("Disambiguated %s entities with duplicate labels", len(ambiguous_entities))
    return entity_to_label


@retry(wait=wait_random_exponential(multiplier=1, max=60))
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

def collect_entity_aliases(
    endpoint,
    entity_ids,
    language,
    batch_size,
    sleep_seconds,
    stats: MappingStats,
):
    entity_to_aliases = {}

    stats.info("Collecting entity aliases...")

    for batch in tqdm(list(chunks(entity_ids, batch_size)), desc="entity alias batches"):
        entity_to_aliases.update(get_entity_aliases_batch(endpoint, batch, language))
        time.sleep(sleep_seconds)

    stats.info("Entity aliases collected: %s", len(entity_to_aliases))
    return entity_to_aliases


def build_inverse_constraint_mappings(constraint_dict, prop_to_data_type, stats: MappingStats):
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

    stats.info("Subject constraint buckets: %s", len(subj_to_prop_constraints))
    stats.info("Object constraint buckets: %s", len(obj_to_prop_constraint))
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
    ontology_mappings_dir = f"src/wikontic/utils/ontology_mappings_{args.language}_{args.fallback_language}"
    output_dir = resolve_output_dir(ontology_mappings_dir)
    wikidata_languages = f"{args.language},{args.fallback_language}"
    stats = MappingStats()

    stats.info("Writing outputs to %s", output_dir)

    stats.info("Collecting property metadata from Wikidata SPARQL endpoint...")
    prop_to_label, prop_to_data_type, prop_to_aliases = collect_property_metadata(
        args.wikidata_endpoint,
        args.language,
        wikidata_languages,
        stats,
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
        stats,
    )

    entities = collect_constrained_entities(constraint_dict, stats)
    entity_to_hierarchy = collect_entity_hierarchy(
        args.wikidata_endpoint,
        entities,
        args.hierarchy_batch_size,
        args.hierarchy_sleep_seconds,
        stats,
    )

    entity_to_label = collect_entity_labels(
        args.wikidata_endpoint,
        entities,
        wikidata_languages,
        args.label_batch_size,
        args.label_sleep_seconds,
        stats,
    )
    entity_to_label = disambiguate_duplicate_labels(
        args.wikidata_endpoint,
        entity_to_label,
        args.language,
        args.fallback_language,
        args.entity_info_batch_size,
        args.entity_info_sleep_seconds,
        stats,
    )

    labeled_entity_ids = set(entity_to_label.keys())
    entity_to_hierarchy = filter_hierarchy_to_labeled_entities(
        entity_to_hierarchy,
        labeled_entity_ids,
    )
    constraint_dict = strip_unlabeled_entities_from_constraints(
        constraint_dict,
        labeled_entity_ids,
    )

    dump_json(output_dir / "entity_type2label.json", entity_to_label)
    dump_json(output_dir / "entity_type2hierarchy.json", entity_to_hierarchy)
    dump_json(output_dir / "prop2constraints.json", constraint_dict)

    entity_to_aliases = collect_entity_aliases(
        args.wikidata_endpoint,
        list(entity_to_label.keys()),
        args.language,
        args.entity_alias_batch_size,
        args.entity_alias_sleep_seconds,
        stats,
    )
    dump_json(output_dir / "entity_type2aliases.json", entity_to_aliases)

    subj_to_prop_constraints, obj_to_prop_constraint = build_inverse_constraint_mappings(
        constraint_dict,
        prop_to_data_type,
        stats,
    )
    subj_to_prop_constraints, obj_to_prop_constraint = filter_inverse_constraint_mappings(
        subj_to_prop_constraints,
        obj_to_prop_constraint,
        labeled_entity_ids,
    )
    dump_json(output_dir / "subj_constraint2prop.json", subj_to_prop_constraints)
    dump_json(output_dir / "obj_constraint2prop.json", obj_to_prop_constraint)

    write_summary_files(output_dir, stats)


if __name__ == "__main__":
    main()
