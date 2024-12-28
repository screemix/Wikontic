import requests
from urllib.parse import urlencode
import json
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from itertools import product
from tenacity import retry, wait_random_exponential, before_sleep_log
import logging
import sys
import jsonlines


logging.basicConfig(stream=sys.stderr)

logger = logging.getLogger('Verifier')
logger.setLevel(logging.ERROR)


class TripletFilter:
    def __init__(self, constrained_dict_path='subject_object_constraints.json'):

        with open(constrained_dict_path, 'r') as f:
            self.constrained_dict = json.load(f)

        self.cached_hirerachy = {}
        with open('logs/hierarchy_cache.jsonl', 'r') as f:
            for line in f:
                hirerachy_map = json.loads(line)
                item = list(hirerachy_map.keys())[0]
                hirerachy = list(hirerachy_map.values())[0]
                self.cached_hirerachy[item] = hirerachy

        # self.constraint_id2rel = {"Q21514624": ["P279"], "Q21503252": ["P31"], "Q30208840": ["P279", "P31"]}

    @retry(wait=wait_random_exponential(multiplier=1, max=60), before_sleep=before_sleep_log(logger, logging.ERROR))
    def get_subclass_hierarchy(self, entity_id):
        # Wikidata SPARQL endpoint
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        # SPARQL query to get all subclasses (direct and indirect) of the given entity
        query = f"""
        SELECT DISTINCT ?subclass ?subclassLabel WHERE {{
            {{
                wd:{entity_id} p:P31/ps:P31/wdt:P279* ?subclass.
            }}
              UNION
            {{
                wd:{entity_id} p:P279/ps:P279/wdt:P279* ?subclass.
            }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        # Execute the query and parse the results
        results = sparql.query().convert()

        subclass_hierarchy = []

        # Extract subclass information from the query results
        for result in results["results"]["bindings"]:
            subclass_id = result["subclass"]["value"].split("/")[-1]
            subclass_hierarchy.append(subclass_id)

        with jsonlines.open('logs/hierarchy_cache.jsonl', mode='a') as writer:
            writer.write({entity_id: subclass_hierarchy})

        return subclass_hierarchy

        
    def check_triplet_validity(self, subject, relation, value):
        
        subject_validity = True
        object_validity = True

        if relation in self.constrained_dict:
            constraints = self.constrained_dict[relation]

            if 'subject type constraint' in constraints:
                # constraint_relations = self.constraint_id2rel[constraints['subject type constraint']['P2309'][0]]
                constraint_entities = constraints['subject type constraint']['P2308']
                subject_validity = self.check_entity_validity(subject, constraint_entities)
                        
            if 'value-type constraint' in constraints:
                # constraint_relations = self.constraint_id2rel[constraints['value-type constraint']['P2309'][0]]
                constraint_entities = constraints['value-type constraint']['P2308']
                object_validity = self.check_entity_validity(value, constraint_entities)
            
        return subject_validity and object_validity


    def check_entity_validity(self, entity, constraint_entities):
        if entity in self.cached_hirerachy:
            res = self.cached_hirerachy[entity]
        else:
            res = self.get_subclass_hierarchy(entity)
            self.cached_hirerachy[entity] = res.copy()
        
        try:
            if len(set(constraint_entities) & set(res)) > 0:
                return True
            else:
                return False
        except Exception as e:
            print(e, '\n', res)



            






