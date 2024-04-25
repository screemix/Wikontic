import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class Aligner:
    def __init__(self, k=5, relation_mapping_filename="relation_mapping.json", 
                 entity_mapping_filename='constrained_entity_mapping.json', entity_index_filename='entities.index', 
                 relation_index_filename='relations.index', device='cuda:0'):
        
        self.k = k
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)

        with open(relation_mapping_filename, "r") as f:
            self.id2relation = json.load(f)

        with open(entity_mapping_filename, "r") as f:
            self.id2entity = json.load(f)

        self.relation2id = {}
        for id_, rel in self.id2relation.items():
            self.relation2id[rel] = id_

        self.entity2id = {}
        for id_, ent in self.id2entity.items():
            self.entity2id[ent] = id_

        self.relation_index = faiss.read_index(relation_index_filename)
        self.entity_index = faiss.read_index(entity_index_filename)


    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


    def embed_batch(self, names):
        inputs = self.tokenizer(names, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**inputs.to(self.device))
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        return np.array(embeddings.detach().cpu())


    def top_relations_by_llm_output(self, llm_output):
        # 2do: rewrite it for universal format
        if isinstance(llm_output, str):
            if llm_output.startswith('Output:'):
                llm_output = llm_output.replace('Output:', '')
            llm_output = json.loads(llm_output)

        relations = [out['relation'] for out in llm_output]
        # print(relations)
        embeddings = self.embed_batch(relations)
        _, indices = self.relation_index.search(embeddings, self.k)

        output = {}
        # output = []

        for i, rel in enumerate(relations):
            top_rels_names = [self.id2relation["P" + str(idx)] for idx in indices[i]]
            output[rel] = top_rels_names

        return output
    

    def top_entities_by_llm_output(self, llm_output, entity_type='head'):
        # 2do: rewrite it for universal format
        if isinstance(llm_output, str):
            if llm_output.startswith('Output:'):
                llm_output = llm_output.replace('Output:', '')
            llm_output = json.loads(llm_output)

        entities = [out[entity_type] for out in llm_output]
        # print(relations)
        embeddings = self.embed_batch(entities)
        _, indices = self.entity_index.search(embeddings, self.k)

        output = {}
        # output = []

        for i, entity in enumerate(entities):
            top_entities_names = [self.id2entity["Q" + str(idx)] for idx in indices[i]]
            output[entity] = top_entities_names
        return output

    # def top_similar_relations(self, rel, k=5):

    #     embeddings = self.embed_batch([rel])
    #     _, indices = self.relation_index.search(embeddings, k)

    #     top_rels_names = [self.id2relation["P" + str(idx)] for idx in indices[0]]
    #     top_rels_ids = ["P" + str(idx)for idx in indices[0]]
    #     return {"names": top_rels_names, "ids": top_rels_ids}


    # def top_similar_entities(self, entity, k=5):

    #     embeddings = self.embed_batch(entity)
    #     _, indices = self.entity_index.search(embeddings, k)

    #     top_entities_names = [self.id2entity["Q" + str(idx)] for idx in indices[0]]
    #     top_entities_ids = ["Q" + str(idx) for idx in indices[0]]
    #     return {"names": top_entities_names, "ids": top_entities_ids}
