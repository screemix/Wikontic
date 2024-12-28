import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class Aligner:
    # TODO - COMBINE INDICES WITH AND WITHOUT DESCRIPTIONS
    def __init__(self, k=5, device='cuda:0'):

        self.k = k
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)

        self.id2relation = {}
        self.relation2id = {}
        self.id2relation_with_description = {}

        self.id2entity = {}
        self.entity2id = {}
        self.id2entity_with_description = {}

        dim = 768
        metric = faiss.METRIC_INNER_PRODUCT

        self.relation_index = faiss.index_factory(dim, "IDMap,Flat", metric)
        self.entity_index = faiss.index_factory(dim, "IDMap,Flat", metric)
        
        self.relation_with_description_index = faiss.index_factory(dim, "IDMap,Flat", metric)
        self.entity_with_description_index = faiss.index_factory(dim, "IDMap,Flat", metric)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


    def embed_batch(self, names):
        inputs = self.tokenizer(names, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**inputs.to(self.device))
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        return np.array(embeddings.detach().cpu())


    def top_relations_by_llm_output(self, relations, with_descriptions=False):
        
        output = {}
        embeddings = self.embed_batch(relations)
        
        if with_descriptions:
            _, indices = self.relation_with_description_index.search(embeddings, min(self.k, len(self.relation2id)))
            
        else:
            _, indices = self.relation_index.search(embeddings, min(self.k, len(self.relation2id)))


        for i, rel in enumerate(relations):

            top_rels_names = [self.id2relation[idx] for idx in indices[i]]

            if with_descriptions:
                output[rel.split(";")[0]] = top_rels_names
            else:
                output[rel] = top_rels_names

        return output
    

    def top_entities_by_llm_output(self, entities, with_descriptions=False):
        
        output = {}
        embeddings = self.embed_batch(entities)
        
        if with_descriptions:
             _, indices =  self.entity_with_description_index.search(embeddings, min(self.k, len(self.entity2id)))
           
        else:
            _, indices = self.entity_index.search(embeddings, min(self.k, len(self.entity2id)))

        

        for i, entity in enumerate(entities):
            top_entities_names = [self.id2entity[idx] for idx in indices[i]]

            if with_descriptions:
                output[entity.split(";")[0]] = top_entities_names
                
            else:
                output[entity] = top_entities_names

        return output


    def add_entities(self, entities, descriptions):

        ids = np.array([len(self.id2entity) + i for i in range(len(entities))])
        embeddings = self.embed_batch(entities)
        description_embeddings = self.embed_batch(descriptions)

        self.entity_index.add_with_ids(embeddings, ids)
        self.entity_with_description_index.add_with_ids(description_embeddings, ids)

        for (id_, entity, description) in zip(ids, entities, descriptions):
            self.id2entity[id_] = entity
            self.entity2id[entity] = id_
            self.id2entity_with_description[id_] = description


    def add_relations(self, relations, descriptions):

        ids = np.array([len(self.id2relation) + i for i in range(len(relations))])
        embeddings = self.embed_batch(relations)
        description_embeddings = self.embed_batch(descriptions)

        self.relation_index.add_with_ids(embeddings, ids)
        self.relation_with_description_index.add_with_ids(description_embeddings, ids)

        for (id_, relation, description) in zip(ids, relations, descriptions):
            self.id2relation[id_] = relation
            self.relation2id[relation] = id_
            self.id2relation_with_description[id_] = description

