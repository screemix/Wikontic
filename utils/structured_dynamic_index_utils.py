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


    def rank_strings_by_similarity(self, candidates, target, k=10):
        candidate_embeddings = self.embed_batch(candidates)
        target_embedding = self.embed_batch([target])

        similarities = cosine_similarity(candidate_embeddings, target_embedding).flatten()

        ranked_candidates = sorted(zip(candidates, similarities), key=lambda x: x[1], reverse=True)

        return ranked_candidates[:k]


    def retrieve_similar_entity_types(self, triplet, k=20):
        # retrieve k most similar entity types to the given triplet
        # using the entity type index
        # return the wikidata ids of the most similar entity types
        subject_type = triplet['subject_type']
        object_type = triplet['object_type']

        _, sim_object_type_idxs =  self.entity_type_index.search(self.embed_batch([subject_type]), k)
        _, sim_subject_type_idxs = self.entity_type_index.search(self.embed_batch([object_type]), k)

        sim_object_type_ids = ["Q"+str(i) for i in sim_object_type_idxs[0]]
        sim_subject_type_ids = ["Q"+str(i) for i in sim_subject_type_idxs[0]]
       
        return sim_object_type_ids, sim_subject_type_ids

    def retrieve_properties_for_entity_type(self, target_relation, object_types, subject_types, k=10):
        props = []

        obj_props = []
        subj_props = []

        for entity in object_types:
            if entity in self.obj2prop:
                obj_props.extend(self.obj2prop[entity])
        
        for entity in subject_types:
            if entity in self.subj2prop:
                subj_props.extend(self.subj2prop[entity])
        
        props.extend(list(set(obj_props) & set(subj_props)))

        obj_props = []
        subj_props = []
        for entity in object_types:
            if entity in self.subj2prop:
                subj_props.extend(self.subj2prop[entity])
        
        for entity in subject_types:
            if entity in self.obj2prop:
                obj_props.extend(self.obj2prop[entity])

        props.extend(list(set(obj_props) & set(subj_props)))

        prop_labels = [self.prop2label[prop] for prop in props]
        ranked_props = []

        if len(prop_labels) > 0:

            ranked_props = self.rank_strings_by_similarity(candidates=prop_labels, target=target_relation, k=k)
        
        return [prop[0] for prop in ranked_props]
    