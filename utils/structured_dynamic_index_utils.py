import json
import faiss
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


class Aligner:
    def __init__(self, k=5, device='cuda:0'):

        # parameters for model for retrieving 
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)

        # dim = 768
        # metric = faiss.METRIC_INNER_PRODUCT

        # constraints mapping object/subject types -> properties

        with open('utils/ontology_mappings/obj_constraint2prop.json', 'r') as f:
            self.obj2prop = json.load(f)

        with open('utils/ontology_mappings/subj_constraint2prop.json', 'r') as f:
            self.subj2prop = json.load(f)

        # mapping properties -> labels
        with open("utils/ontology_mappings/prop2label.json", 'r') as f:
            self.prop2label = json.load(f)

        # mapping entities -> labels
        with open('utils/ontology_mappings/ontology_entity2label.json', 'r') as f:
            self.entity2label = json.load(f)

        with open('utils/ontology_mappings/entity_hierarchy.json', 'r') as f:
            self.entity2hierarchy = json.load(f)

        with open('utils/ontology_mappings/subject_object_constraints.json', 'r') as f:
            self.prop2constraint = json.load(f)

        # entity type index
        self.entity_type_index = faiss.read_index("utils/ontology_mappings/wikidata_ontology_entities.index")


    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


    def embed_batch(self, names):
        inputs = self.tokenizer(names, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**inputs.to(self.device))
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        return np.array(embeddings.detach().cpu().numpy())


    def rank_relations_by_similarity(self, candidates_labels, target_relation):

        candidate_embeddings = self.embed_batch(candidates_labels)
        target_embedding = self.embed_batch([target_relation])
        similarities = cosine_similarity(candidate_embeddings, target_embedding).flatten()
        return similarities


    def retrieve_similar_entity_types(self, triplet, k=10):
        # retrieve k most similar entity types to the given triplet
        # using the entity type index
        # return the wikidata ids of the most similar entity types
        subject_type = triplet['subject_type']
        object_type = triplet['object_type']

        _, sim_object_type_idxs =  self.entity_type_index.search(self.embed_batch([subject_type]), k)
        _, sim_subject_type_idxs = self.entity_type_index.search(self.embed_batch([object_type]), k)

        # print([self.entity2label["Q"+str(i)] for i in sim_object_type_idxs[0]])
        # print([self.entity2label["Q"+str(i)] for i in sim_subject_type_idxs[0]])

        sim_object_type_ids = ["Q"+str(i) for i in sim_object_type_idxs[0]]
        sim_subject_type_ids = ["Q"+str(i) for i in sim_subject_type_idxs[0]]
       
        return sim_object_type_ids, sim_subject_type_ids


    def retrieve_properties_for_entity_type(self, target_relation, object_types, subject_types, k=10):
        # dicionaries to preserve constraints of the ranked properties
        prop2obj = {}
        prop2subj = {}
        
        direct_properties = {"subject_properties": set(), 
                             "object_properties": set(),
                             "property_to_subject_types": {},
                             "property_to_object_types": {}
                             }
        
        inverse_properties = {"subject_properties": set(), 
                              "object_properties": set(),
                              "property_to_subject_types": {},
                              "property_to_object_types": {}
                              }

        for object_type in object_types:
            # extending the object type with its supertypes
            object_super_types = [object_type]
            if object_type in self.entity2hierarchy:
                object_super_types.extend(self.entity2hierarchy[object_type])

            for entity in object_super_types:
                if entity in self.obj2prop:
                    properties = self.obj2prop[entity]
                    direct_properties["object_properties"].update(properties)
                    for prop in properties:
                        if prop not in direct_properties["property_to_object_types"]:
                            direct_properties["property_to_object_types"][prop] = set()
                        direct_properties["property_to_object_types"][prop].add(object_type)
                
                if entity in self.subj2prop:
                    properties = self.subj2prop[entity]
                    inverse_properties["subject_properties"].update(properties)
                    for prop in properties:
                        if prop not in inverse_properties["property_to_subject_types"]:
                            inverse_properties["property_to_subject_types"][prop] = set()
                        inverse_properties["property_to_subject_types"][prop].add(object_type)

        
        for subject_type in subject_types:
            # extending the subject type with its supertypes
            subject_super_types = [subject_type]
            if subject_type in self.entity2hierarchy:
                subject_super_types.extend(self.entity2hierarchy[subject_type])

            for entity in subject_super_types:
                if entity in self.subj2prop:
                    properties = self.subj2prop[entity]
                    direct_properties["subject_properties"].update(properties)
                    for prop in properties:
                        if prop not in direct_properties["property_to_subject_types"]:
                            direct_properties["property_to_subject_types"][prop] = set()
                        direct_properties["property_to_subject_types"][prop].add(subject_type)
                
                if entity in self.obj2prop:
                    properties = self.obj2prop[entity]
                    inverse_properties["object_properties"].update(properties)
                    for prop in properties:
                        if prop not in inverse_properties["property_to_object_types"]:
                            inverse_properties["property_to_object_types"][prop] = set()
                        inverse_properties["property_to_object_types"][prop].add(subject_type)
        
        direct_props = list(direct_properties["subject_properties"] & direct_properties["object_properties"])
        inverse_props = list(inverse_properties["subject_properties"] & inverse_properties["object_properties"])

        props = [(prop, self.prop2label[prop], "direct") for prop in direct_props] + \
                [(prop, self.prop2label[prop], "inverse") for prop in inverse_props]
        props = list(set(props))

        ranked_props = []

        if len(props) > 0:

            similarities = self.rank_relations_by_similarity(candidates_labels=[prop[1] for prop in props], \
                                                              target_relation=target_relation)
            ranked_props = sorted(zip(props, similarities), key=lambda x: x[1], reverse=True)
            ranked_props = [ranked_prop[0] for ranked_prop in ranked_props[:k]]
        
        resulted_props = []
        for prop in ranked_props:

            resulted_prop = {"id": prop[0], "label": prop[1], "type": prop[2]}

            if prop[2] == "direct":
                resulted_prop['object_types'] = list(direct_properties["property_to_object_types"][prop[0]])
                resulted_prop['subject_types'] = list(direct_properties["property_to_subject_types"][prop[0]])
            else:
                resulted_prop['object_types'] = list(inverse_properties["property_to_object_types"][prop[0]])
                resulted_prop['subject_types'] = list(inverse_properties["property_to_subject_types"][prop[0]])
            
            resulted_props.append(resulted_prop)

        return resulted_props
    