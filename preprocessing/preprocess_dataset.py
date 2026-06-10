import json
import argparse
import os
from datasets import load_dataset

from wikontic.logging_config import get_logger

logger = get_logger("PreprocessDataset")

def get_json_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        ds = json.load(f)
    return ds


def get_jsonl_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        ds = [json.loads(line) for line in f]
    return ds


def convert_musique_dataset(ds):
    id2texts = {}
    for elem in ds:
        answerable = "Answerable" if elem["answerable"] else "Unanswerable"
        texts = [item["paragraph_text"] for item in elem["paragraphs"]]
        id2texts[elem["id"] + "_" + answerable] = texts
    return id2texts

def convert_edgar_dataset(ds):
    id2texts = {}
    for elem in ds:
        for key in elem:
            id2texts[key] = elem[key]
    return id2texts

def convert_hotpot_dataset(ds):
    id2texts = {}
    for elem in ds:
        texts = [" ".join(item[1]) for item in elem["context"]]
        id2texts[elem["_id"]] = texts
    return id2texts

def convert_mine_dataset():
    ds = load_dataset("kyssen/kg-gen-evaluation-essays")
    ds = ds["train"]
    id2texts = {}
    for i, elem in enumerate(ds):
        texts = elem["content"].split("\n\n")[1:]
        id2texts[i] = texts
    return id2texts


def convert_ru_dataset():
    ds = load_dataset(
        "mxlcw/telegram-financial-sentiment-summarization", split="train"
    )
    id2texts = {}
    for i in range(len(ds)):
        ds_item = ds[i]
        id2texts[i] = ds_item["text"]
    return id2texts
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/data/musique_full_v1.0_train.jsonl")
    parser.add_argument("--preprocessing", type=str, default="musique")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.dataset_path.endswith(".json"):
        ds = get_json_dataset(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl"):
        ds = get_jsonl_dataset(args.dataset_path)
    else:
        raise ValueError(f"Unknown dataset format: {args.dataset_path}. Please use a .json or .jsonl file.")
    logger.info(f"Preprocessing dataset {args.dataset_path} with {args.preprocessing} preprocessing...")
    try:
        if args.preprocessing == "musique":
            ds = convert_musique_dataset(ds)
        elif args.preprocessing == "edgar":
            ds = convert_edgar_dataset(ds)
        elif args.preprocessing == "hotpot":
            ds = convert_hotpot_dataset(ds)
        elif args.preprocessing == "mine":
            ds = convert_mine_dataset()
            args.dataset_path = '../datasets/mine.json'
        elif args.preprocesing == "ru_dataset":
            ds = convert_ru_dataset()
        else:
            raise ValueError(f"Unknown preprocessing function: {args.preprocessing}")
    except Exception as e:
        raise ValueError(f"Error preprocessing dataset: {e}. You may need to check the dataset format and implementation of the preprocessing function.")

    try:
        base, _ = os.path.splitext(args.dataset_path)
        output_path = f"{base}_preprocessed.json"
   
        with open(output_path, "w") as f:
            json.dump(ds, f, indent=4)
    except Exception as e:
        raise ValueError(f"Error saving dataset: {e}. You may need to check the output path and permissions.")