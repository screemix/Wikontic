"""Utilities for fetching Wikidata head/tail triplets for an entity."""

from SPARQLWrapper import JSON, SPARQLWrapper
import requests
from bs4 import BeautifulSoup
import re


def get_head_triplets(entity_id: str) -> list[dict[str, str]]:
    """Return triplets where the given entity is the subject."""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?subjectLabel ?propertyLabel ?objectLabel ?object   WHERE {{

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .
      }}
      VALUES (?subject) {{(wd:{entity_id})}}
      ?subject ?predicate ?object .
      ?property wikibase:directClaim ?predicate.

      FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/direct/")) .
      FILTER(STRSTARTS(STR(?object), "http://www.wikidata.org/entity/")) .

    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    output_triplets: list[dict[str, str]] = []
    for result in results["results"]["bindings"]:
        obj_id = result["object"]["value"].split("/")[-1]
        subject = result["subjectLabel"]["value"]
        predicate = result["propertyLabel"]["value"]
        object_ = result["objectLabel"]["value"]

        output_triplets.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "subj_id": entity_id,
                "obj_id": obj_id,
            }
        )

    return output_triplets


def get_tail_triplets(entity_id: str) -> list[dict[str, str]]:
    """Return triplets where the given entity is the object."""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?subjectLabel ?propertyLabel ?objectLabel ?subject WHERE {{

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .
      }}

      VALUES (?object) {{(wd:{entity_id})}}
      ?subject ?predicate ?object .
      ?property wikibase:directClaim ?predicate.

      FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/direct/")) .
      FILTER(STRSTARTS(STR(?object), "http://www.wikidata.org/entity/")) .

    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    output_triplets: list[dict[str, str]] = []
    for result in results["results"]["bindings"]:
        subject = result["subjectLabel"]["value"]
        predicate = result["propertyLabel"]["value"]
        object_ = result["objectLabel"]["value"]
        subj_id = result["subject"]["value"].split("/")[-1]

        output_triplets.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "subj_id": subj_id,
                "obj_id": entity_id,
            }
        )

    return output_triplets


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def get_wiki_paragraphs_by_entity(entity_name: str) -> list[tuple[str, list[str]]]:
    """Return paragraphs from Wikipedia about the given entity."""

    url = f"https://en.wikipedia.org/wiki/{entity_name}"
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find(id="firstHeading")

    text_div = soup.find("div", class_='mw-content-ltr mw-parser-output')

    regex = re.compile('infobox.*')
    last_marked = text_div.find("table", {"class" : regex})

    texts = []
    last_marked = text_div.find("h2")

    for text in last_marked.find_all_previous('p'):
        texts.append(text)

    texts.reverse()
    
    text_metadata = []
    for text in texts:
        external_entities = []
        for entity in text.find_all("a"):
            title = entity.get('title')
            if title:
                external_entities.append(title)
        
        content = text.text.strip()
        if len(content) > 0:
            text_metadata.append((content, external_entities))
    
    return text_metadata
