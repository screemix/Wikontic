import json
import faiss
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def load_prop_mapping(prop_mapping_filename="prop2id.json"):
    with open(prop_mapping_filename, "r") as f:
        prop2id = json.load(f)
    return prop2id


def load_index(index_filename='faiss.index'):
    return  faiss.read_index(index_filename)


def top5_relations(llm_output, faiss_index, prop_names, model=model):
    # 2do: rewrite it for universal format
    relations = [" ".join(out.split("(")[0].split("_")) for out in llm_output]
    # print(relations)
    embeddings = model.encode(relations)
    _, indices = faiss_index.search(embeddings, 5)

    output = ""
    for i, rel in enumerate(relations):
        top5rels = [prop_names[idx] for idx in indices[i]]
        top5_rels = ", ".join(top5rels)
        res = f"{rel}: {top5_rels} \n"
        output += res
    return output

