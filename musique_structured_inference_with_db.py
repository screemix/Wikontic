from unidecode import unidecode
import re
from utils.openai_utils import LLMTripletExtractor
from utils.structured_dynamic_index_utils_with_db import Aligner as DBAligner
from pymongo.mongo_client import MongoClient
import json
from tqdm import tqdm
import warnings
import tenacity
from utils.structured_inference_with_db import extract_triplets

warnings.filterwarnings('ignore')

def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client


if __name__ == "__main__":
    with open("musique_200_test.json", "r") as f:
        ds = json.load(f)
    

    mongo_client = get_mongo_client("mongodb://localhost:27018/?directConnection=true")
    db = mongo_client.get_database("wikidata_ontology")

    
    ds = ds['data']

    id2sample = {}
    for elem in ds:
        id2sample[elem['id']] = elem

    sampled_ids = list(id2sample.keys())[:50]

    model_name = 'gpt-4.1-mini'
    current_prompt_token_num, current_completion_token_num = 0, 0

    for i, sample_id in tqdm(enumerate(sampled_ids), total=len(sampled_ids)):
        triplets = []
        faulty_triplets = []
        final_triplets_source_text_ids = []
        filtered_triplets_source_text_ids = []
        prompt_token_nums, completion_token_nums = [], []

        sample = id2sample[sample_id]
        
        aligner = DBAligner(db)
        
        extractor = LLMTripletExtractor(model=model_name)

        texts = [item['paragraph_text'] for item in sample['paragraphs']]
        
        for idx, text in tqdm(enumerate(texts), total=len(texts)):
            try:
                final_triplets, filtered_triplets = extract_triplets(text, sample_id=sample_id, extractor=extractor, aligner=aligner)
            except Exception as e:
                print(f"Failed to extract triplets after 3 retries: {str(e)}")
                final_triplets, filtered_triplets = [], []
            
            prompt_token_num, completion_token_num = extractor.calculate_used_tokens()

            for triple in final_triplets:
                triple["source_text_id"] = idx
                triple["prompt_token_nums"] = prompt_token_num - current_prompt_token_num
                triple["completion_token_num"] = completion_token_num - current_completion_token_num

            for triple in filtered_triplets:
                triple["source_text_id"] = idx
                triple["prompt_token_nums"] = prompt_token_num - current_prompt_token_num
                triple["completion_token_num"] = completion_token_num - current_completion_token_num
            
            if len(final_triplets) > 0:
                aligner.add_triplets(final_triplets, sample_id=sample_id)
            if len(filtered_triplets) > 0:
                aligner.add_filtered_triplets(filtered_triplets, sample_id=sample_id)

            current_prompt_token_num, current_completion_token_num = prompt_token_num, completion_token_num

        print("CURRENT COST: ", extractor.calculate_cost())