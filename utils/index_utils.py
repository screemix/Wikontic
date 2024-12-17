import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class Aligner:
    def __init__(self, k=5, 
                relation_mapping_filename="utils/index_files/relation_mapping.json",
                relation_with_description_mapping_filename='utils/index_files/relation_mapping_with_descriptions.json', 
                entity_mapping_filename='utils/index_files/entity_mapping.json', 
                entity_with_description_mapping_filename='utils/index_files/entity_mapping_with_descriptions.json',
                relation_index_filename='utils/index_files/relations.index',
                relation_with_description_index_filename='utils/index_files/relations_with_descriptions.index',
                entity_index_filename='utils/index_files/entities.index', 
                entity_with_description_index_filename='utils/index_files/entities_with_descriptions.index',
                device='cuda:0'):

        self.k = k
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)

        with open(relation_mapping_filename, "r") as f:
            self.id2relation = json.load(f)

        with open(relation_with_description_mapping_filename, "r") as f:
            self.id2relation_with_description = json.load(f)

        with open(entity_mapping_filename, "r") as f:
            self.id2entity = json.load(f)
        
        with open(entity_with_description_mapping_filename, "r") as f:
            self.id2entity_with_description = json.load(f)

        self.relation2id = {}
        for id_, rel in self.id2relation.items():
            self.relation2id[rel] = id_

        self.entity2id = {}
        for id_, ent in self.id2entity.items():
            self.entity2id[ent] = id_

        self.relation_index = faiss.read_index(relation_index_filename)
        self.entity_index = faiss.read_index(entity_index_filename)

        self.relation_with_description_index = faiss.read_index(relation_with_description_index_filename)
        self.entity_with_description_index = faiss.read_index(entity_with_description_index_filename)


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
            _, indices = self.relation_with_description_index.search(embeddings, self.k)
            
        else:
            _, indices = self.relation_index.search(embeddings, self.k)


        for i, rel in enumerate(relations):

            top_rels_names = [self.id2relation["P" + str(idx)] for idx in indices[i]]

            if with_descriptions:
                output[rel.split(";")[0]] = top_rels_names
            else:
                output[rel] = top_rels_names

        return output
    

    def top_entities_by_llm_output(self, entities, with_descriptions=False):
        
        output = {}
        embeddings = self.embed_batch(entities)
        
        if with_descriptions:
             _, indices =  self.entity_with_description_index.search(embeddings, self.k)
           
        else:
            _, indices = self.entity_index.search(embeddings, self.k)

        

        for i, entity in enumerate(entities):
            top_entities_names = [self.id2entity["Q" + str(idx)] for idx in indices[i]]

            if with_descriptions:
                output[entity.split(";")[0]] = top_entities_names
                
            else:
                output[entity] = top_entities_names

        return output