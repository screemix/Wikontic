import json
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import re
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, before_sleep_log
import argparse
import os
import logging
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ontology_mappings_dir", type=str, default="src/wikontic/utils/ontology_mappings/"
    )
    return parser.parse_args()


def get_property_label_and_data_type(sparql: SPARQLWrapper):
    # SPARQL query for properties with data types: Item, Quantity, Point in time
    query = """
    SELECT ?property ?propertyLabel ?typeLabel WHERE {
    ?property a wikibase:Property .
    ?property wikibase:propertyType ?type .
    
    VALUES ?type { wikibase:WikibaseItem wikibase:Quantity wikibase:Time }
    
    BIND(
        IF(?type = wikibase:WikibaseItem, "Item",
        IF(?type = wikibase:Quantity, "Quantity",
            IF(?type = wikibase:Time, "Point in time", "Unknown")
        )
        ) AS ?typeLabel
    )
    
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    PROP_2_LABEL = {}
    PROP_2_DATA_TYPE = {}

    try:
        results = sparql.query().convert()

        for result in results["results"]["bindings"]:
            prop = result["property"]["value"].split("/")[-1]
            label = result.get("propertyLabel", {}).get("value", "No label")
            data_type = result.get("typeLabel", {}).get("value", "Unknown")

            PROP_2_LABEL[prop] = label
            PROP_2_DATA_TYPE[prop] = data_type

    except Exception as e:
        logger.error(f"Error executing SPARQL query: {e}")

    return PROP_2_LABEL, PROP_2_DATA_TYPE


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def get_property_aliases(sparql: SPARQLWrapper, property_id: str):
    query = f"""
    SELECT ?alias WHERE {{
      wd:{property_id} skos:altLabel ?alias .
      FILTER (lang(?alias) = "en")
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    aliases = [result["alias"]["value"] for result in results["results"]["bindings"]]
    return aliases


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def get_constraints(sparql: SPARQLWrapper, property_id: str):
    """Retrieve value-type and subject-type constraints for a specified Wikidata property."""

    query = f"""
    SELECT ?constraintType ?entity WHERE {{
      VALUES ?property {{ wd:{property_id} }}  

      ?property p:P2302 ?statement.  # Property constraints
      ?statement ps:P2302 ?constraintEntity.  # Constraint type

      VALUES ?constraintEntity {{ wd:Q21510865 wd:Q21503250 }}  # Value-type & Subject-type constraints

      ?statement pq:P2308 ?entity.  # The constrained entity type (allowed type)

      BIND(
        IF(?constraintEntity = wd:Q21510865, "Value-type constraint", "Subject type constraint")
        AS ?constraintType
      )
    }}
    """
    # SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    constraints = {"Value-type constraint": [], "Subject type constraint": []}
    for result in results["results"]["bindings"]:
        constraints[result["constraintType"]["value"]].append(
            result["entity"]["value"].split("/")[-1]
        )

    return constraints

@retry(wait=wait_random_exponential(multiplier=1, max=60))
def get_entity_hierarchy(sparql: SPARQLWrapper, entity_id: str):
    query = f"""
    SELECT DISTINCT ?subclass ?subclassLabel WHERE {{
      wd:{entity_id} wdt:P31/wdt:P279* ?subclass.
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    hierarchy = [result["subclass"]["value"].split("/")[-1] for result in results["results"]["bindings"]]
    return hierarchy


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def fetch_labels(sparql: SPARQLWrapper, batch: list, batch_size: int = 50):
    entity_values = " ".join(f"wd:{entity}" for entity in batch)

    query = f"""
    SELECT ?entity ?entityLabel WHERE {{
      VALUES ?entity {{ {entity_values} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        return {
            result["entity"]["value"]
            .split("/")[-1]: result.get("entityLabel", {})
            .get("value", "")
            for result in results["results"]["bindings"]
        }
    except Exception as e:
        logger.error(f"Error with batch {batch[:5]}...: {e}")
        return {}

    # Collecting descriptions for entity types with duplicated labels
    @retry(wait=wait_random_exponential(multiplier=1, max=60))
    def get_entity_info(entity_id):
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f"""
        SELECT ?entityLabel ?entityDescription WHERE {{
        wd:{entity_id} rdfs:label ?entityLabel .
        wd:{entity_id} schema:description ?entityDescription .
        FILTER (lang(?entityLabel) = "en")
        FILTER (lang(?entityDescription) = "en")
        }}
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if results["results"]["bindings"]:
            result = results["results"]["bindings"][0]
            return {
                "label": result["entityLabel"]["value"],
                "description": result["entityDescription"]["value"],
            }
        else:
            return None
    
    @retry(wait=wait_random_exponential(multiplier=1, max=60))
    def get_entity_aliases(sparql: SPARQLWrapper, entity_id: str):
        chinese_japanese_pattern = re.compile(
            r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]"
        )

        query = f"""
        SELECT ?alias WHERE {{
        wd:{entity_id} skos:altLabel ?alias .
        FILTER (lang(?alias) = "en")
        }}
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        aliases = []
        for result in results["results"]["bindings"]:

            alias = result["alias"]["value"]
            if not chinese_japanese_pattern.search(alias):
                aliases.append(alias)
            # except Exception as e:
            #    continue

        return aliases

if __name__ == "__main__":
    args = get_args()
    ONTOLOGY_MAPPINGS_DIR = args.ontology_mappings_dir
    if not os.path.exists(ONTOLOGY_MAPPINGS_DIR):
        os.makedirs(ONTOLOGY_MAPPINGS_DIR)
    logger.info(f"Ontology mappings directory: {ONTOLOGY_MAPPINGS_DIR}")

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    ######## Collecting properties' metadata ########
    PROP_2_LABEL, PROP_2_DATA_TYPE = get_property_label_and_data_type(sparql)
    logger.info(f"Successfully collected property data types and labels")
    logger.info(f"Number of properties: {len(PROP_2_LABEL)}")
    logger.info(f"Number of data types: {len(PROP_2_DATA_TYPE)}")

    PROP2ALIASES = {}
    for property_id in tqdm(PROP_2_LABEL.keys()):
        PROP2ALIASES[property_id] = get_property_aliases(sparql, property_id)
    logger.info(f"Successfully collected property aliases")
    logger.info(f"Number of properties with aliases: {len(PROP2ALIASES)}")

    alias_set = set()
    alias_list = []
    for prop, aliases in PROP2ALIASES.items():
        alias_set.update(aliases)
        alias_list.extend(aliases)
    logger.info(f"Successfully counted unique and total alias numbers for properties")
    logger.info(f"Number of unique aliases for properties: {len(alias_set)}")
    logger.info(f"Number of total aliases for properties: {len(alias_list)}")

    constraint_dict = {}

    for prop in tqdm(PROP_2_LABEL.keys()):
        constraint_dict[prop] = get_constraints(sparql, prop)
        time.sleep(0.1)
    logger.info(f"Number of properties with constraints: {len(constraint_dict)}")

    wo_constraint = []
    for prop in constraint_dict:
        if (
            len(constraint_dict[prop]["Value-type constraint"]) == 0
            and len(constraint_dict[prop]["Subject type constraint"]) == 0
        ):
            wo_constraint.append(prop)
    logger.info(f"Number of properties without constraints: {len(wo_constraint)}")

    quantity_props = []
    time_props = []
    other_props = []
    for prop in wo_constraint:
        if PROP_2_DATA_TYPE[prop] == "Quantity":
            quantity_props.append(prop)
        elif PROP_2_DATA_TYPE[prop] == "Point in time":
            time_props.append(prop)
        else:
            other_props.append(prop)
    logger.info(f"Number of properties without constraints with quantity constraints: {len(quantity_props)}")
    logger.info(f"Number of properties without constraints with time constraints: {len(time_props)}")

    for prop in constraint_dict:
        if PROP_2_DATA_TYPE[prop] == "Point in time":
            constraint_dict[prop]["Value-type constraint"].append("Q186408")

        elif PROP_2_DATA_TYPE[prop] == "Quantity":
            constraint_dict[prop]["Value-type constraint"].append("Q309314")


    wo_constraint = []
    for prop in constraint_dict:
        if (
            len(constraint_dict[prop]["Value-type constraint"]) == 0
            and len(constraint_dict[prop]["Subject type constraint"]) == 0
        ):
            wo_constraint.append(prop)
    logger.info(f"Number of properties without constraints and without data type constraints: {len(wo_constraint)}")

    wo_constraint = []
    for prop in constraint_dict:
        if len(constraint_dict[prop]["Value-type constraint"]) == 0:
            constraint_dict[prop]["Value-type constraint"] = ["ANY"]
        if len(constraint_dict[prop]["Subject type constraint"]) == 0:
            constraint_dict[prop]["Subject type constraint"] = ["ANY"]
    logger.info(f"Number of properties with constraints and with ANY type constraints: {len(constraint_dict)}")

    ######## Collecting entities' metadata ########
    entities = set()
    for prop, constraint in constraint_dict.items():
        for const_type in constraint:
            for entity in constraint[const_type]:
                entities.add(entity)
    entities = list(entities)
    logger.info(f"Number of unique entity types: {len(entities)}")

    ENTITY_2_HIERARCHY = {}
    for entity_id in tqdm(entities):
        hierarchy = get_entity_hierarchy(sparql, entity_id)
        ENTITY_2_HIERARCHY[entity_id] = hierarchy
    logger.info(f"Number of unique entity types with hierarchy: {len(ENTITY_2_HIERARCHY)}")

    # leaving only entity types that are used in constraints
    for entity in tqdm(ENTITY_2_HIERARCHY):
        filtered_super_entities = [
            item for item in ENTITY_2_HIERARCHY[entity] if item in entities
        ]
        ENTITY_2_HIERARCHY[entity] = filtered_super_entities
    logger.info(f"Number of unique entity types with hierarchy after filtering: {len(ENTITY_2_HIERARCHY)}")

    ENTITY_2_LABEL = {}
    BATCH_SIZE = 50
    for i in range(0, len(entities), BATCH_SIZE):
        batch = entities[i : i + BATCH_SIZE]
        logger.info(f"Processing batch {i // BATCH_SIZE + 1}/{(len(entities) // BATCH_SIZE) + 1}")
        labels = fetch_labels(sparql, batch, BATCH_SIZE)
        ENTITY_2_LABEL.update(labels)

    logger.info(f"Number of unique entity types with labels and unique labels: {len(set(ENTITY_2_LABEL.keys()))}, {len(set(ENTITY_2_LABEL.values()))}")

    label2entity = {}
    for entity, label in ENTITY_2_LABEL.items():
        if label not in label2entity:
            label2entity[label] = []
        label2entity[label].append(entity)

    # Collecting descriptions for entity types with duplicated labels   
    for label, entities in label2entity.items():
        if len(entities) > 1:
            for entity in entities:
                info = get_entity_info(entity)
                ENTITY_2_LABEL[entity] = info["label"] + " (" + info["description"] + ")"
    logger.info(f"Number of unique entity types with labels and unique labels with descriptions: {len(set(ENTITY_2_LABEL.keys()))}, {len(set(ENTITY_2_LABEL.values()))}")
    
    ENTITY_2_ALIASES = {}
    for entity in tqdm(ENTITY_2_LABEL.keys()):
        ENTITY_2_ALIASES[entity] = get_entity_aliases(sparql, entity)
    logger.info(f"Number of unique entity types with aliases: {len(ENTITY_2_ALIASES)}")
    
    subj2prop_constraints = {"<ANY SUBJECT>": []}
    # Q309314 - quantity, Q186408 -  point in time
    obj2prop_constraint = {"<ANY OBJECT>": [], "Q309314": [], "Q186408": []}

    for prop, constraint in constraint_dict.items():

        if PROP_2_DATA_TYPE[prop] == "Point in time":
            obj2prop_constraint["Q186408"].append(prop)

        elif PROP_2_DATA_TYPE[prop] == "Quantity":
            obj2prop_constraint["Q309314"].append(prop)

        elif constraint["Value-type constraint"] == ["ANY"]:
            obj2prop_constraint["<ANY OBJECT>"].append(prop)

        else:
            for entity in constraint["Value-type constraint"]:
                if entity not in obj2prop_constraint:
                    obj2prop_constraint[entity] = []
                obj2prop_constraint[entity].append(prop)

        if constraint["Subject type constraint"] == ["ANY"]:
            subj2prop_constraints["<ANY SUBJECT>"].append(prop)

        else:
            for entity in constraint["Subject type constraint"]:
                if entity not in subj2prop_constraints:
                    subj2prop_constraints[entity] = []
                subj2prop_constraints[entity].append(prop)


    logger.info(f"Number of subject property constraints: {len(subj2prop_constraints)}")
    logger.info(f"Number of value property constraints: {len(obj2prop_constraint)}")

    # ------------------------------ PROPERTIES ------------------------------

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2constraints.json"), "w") as f:
        json.dump(constraint_dict, f)

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2aliases.json"), "w") as f:
        json.dump(PROP2ALIASES, f)

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2data_type.json"), "w") as f:
        json.dump(PROP_2_DATA_TYPE, f)

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2label.json"), "w") as f:
        json.dump(PROP_2_LABEL, f)

    # ------------------------------ ENTITY TYPES ------------------------------
    
    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "entity_type2label.json"), "w") as f:
        json.dump(ENTITY_2_LABEL, f)

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "entity_type2aliases.json"), "w") as f:
        json.dump(ENTITY_2_ALIASES, f)

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "entity_type2hierarchy.json"), "w") as f:
        json.dump(ENTITY_2_HIERARCHY, f)

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "subj_constraint2prop.json"), "w") as f:
        json.dump(subj2prop_constraints, f)
    
    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "obj_constraint2prop.json"), "w") as f:
        json.dump(obj2prop_constraint, f)