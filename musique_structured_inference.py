import time
import numpy as np
from tqdm import tqdm

import json
import pandas as pd

from transformers import AutoTokenizer, AutoModel

from utils.eval_utils import micro_precision, micro_recall, f1_score
from utils.openai_utils import LLMTripletExtractor
from utils.verifier_utils import TripletFilter
from utils.structured_dynamic_index_utils import Aligner

import warnings
import re
from unidecode import unidecode
import tenacity

warnings.filterwarnings('ignore')

@tenacity.retry(stop=tenacity.stop_after_attempt(3), reraise=True)
def extract_triplets(text):
    """Extract and refine knowledge graph triplets from text using LLM.
    
    Args:
        text (str): Input text to extract triplets from
        
    Returns:
        tuple: (final_triplets, filtered_triplets) where:
            - final_triplets: List of validated and refined triplets
            - filtered_triplets: List of triplets that couldn't be validated
    """
    # Extract initial triplets using LLM
    extracted_triplets = extractor.extract_triplets_from_text(text)
    
    final_triplets = []
    filtered_triplets = []

    for triplet in extracted_triplets['triplets']:
        # print("1st step triplet: ", triplet)
        
        # Get candidate entity types
        subj_type_ids, obj_type_ids = aligner.retrieve_similar_entity_types(triplet=triplet)
        
        # Get candidate properties/relations
        properties = aligner.retrieve_properties_for_entity_type(
            target_relation=triplet['relation'],
            object_types=obj_type_ids, 
            subject_types=subj_type_ids,
            k=5
        )

        # Build candidate triplet backbones
        # print(properties)
        candidates = []
        for prop_id, prop_label, prop_direction in properties:
            if prop_direction == 'direct':
                subject_types = set(subj_type_ids) & set(aligner.prop2constraints[prop_id]['Subject type constraint'])
                object_types = set(obj_type_ids) & set(aligner.prop2constraints[prop_id]['Value-type constraint'])
            else:
                object_types = set(subj_type_ids) & set(aligner.prop2constraints[prop_id]['Value-type constraint']) 
                subject_types = set(obj_type_ids) & set(aligner.prop2constraints[prop_id]['Subject type constraint'])

            # Use original type sets if no constraints matched
            subject_types = subj_type_ids if len(subject_types) == 0 else subject_types
            object_types = obj_type_ids if len(object_types) == 0 else object_types

            candidates.append({
                "subject": triplet['subject'] if prop_direction == 'direct' else triplet['object'],
                "relation": prop_label,
                'object': triplet['object'] if prop_direction == 'direct' else triplet['subject'],
                "subject_types": [aligner.entity_type2label[t] for t in subject_types],
                "object_types": [aligner.entity_type2label[t] for t in object_types]
            })

            # print({
            #     "subject": triplet['subject'] if prop_direction == 'direct' else triplet['object'],
            #     "relation": prop_label,
            #     'object': triplet['object'] if prop_direction == 'direct' else triplet['subject'],
            #     "subject_types": [aligner.entity_type2label[t] for t in subject_types],
            #     "object_types": [aligner.entity_type2label[t] for t in object_types]
            # })


        # Refine relation and entity types using LLM - choose among valid backbones for triplet
        backbone_triplet = extractor.refine_relation_and_entity_types(
            text=text, 
            triplet=triplet,
            candidate_triplets=candidates
        )
        backbone_triplet['qualifiers'] = triplet['qualifiers']

        # Refine entity names
        final_triplet = refine_entities(text, backbone_triplet, aligner)
        
        final_triplets.append(final_triplet)
        # print("2nd resulted triplet: ", final_triplet)

    # print("-"*100)
    return final_triplets, filtered_triplets


def refine_entities(text, triplet, aligner):
    """Refine entity names using type constraints."""
    try:
        triplet['subject'] = unidecode(triplet['subject'])
        triplet['object'] = unidecode(triplet['object'])
    except Exception as e:
        print(e, triplet)
    # Handle object refinement
    obj_type = triplet['object_type']
    obj_type_id = aligner.label2entity_type.get(obj_type, '')
    obj_hierarchy = [obj_type_id] + aligner.entity2hierarchy.get(obj_type_id, [])
    updated_obj = 'None'
    
    # do not change time or quantity entities
    if not any(t in ['Q186408', 'Q309314'] for t in obj_hierarchy):
        similar_objects = aligner.retrieve_entity_by_type(
            entity=triplet['object'],
            entity_type=obj_type
        )
        if similar_objects:
            updated_obj = extractor.refine_entity(
                text=text,
                triplet=triplet,
                candidates=similar_objects,
                is_object=True
            )
            updated_obj = unidecode(str(updated_obj))
    
    if re.sub(r'[^\w\s]', '', updated_obj) != 'None':
        triplet['object'] = updated_obj
    else:
        aligner.add_entity(entity=triplet['object'], entity_type=obj_type)

    # Handle subject refinement  
    updated_subj = 'None'
    similar_subjects = aligner.retrieve_entity_by_type(
        entity=triplet['subject'],
        entity_type=triplet['subject_type']
    )
    if similar_subjects:
        updated_subj = extractor.refine_entity(
            text=text,
            triplet=triplet,
            candidates=similar_subjects,
            is_object=False
        )
        updated_subj = unidecode(str(updated_subj))
    
    if re.sub(r'[^\w\s]', '', updated_subj) != 'None':
        triplet['subject'] = updated_subj
    else:
        aligner.add_entity(entity=triplet['subject'], entity_type=triplet['subject_type'])
        
    return triplet

# 127
if __name__ == "__main__":
    with open("musique_200_test.json", "r") as f:
        ds = json.load(f)

    ds = ds['data']

    id2sample = {}
    for elem in ds:
        id2sample[elem['id']] = elem

    sampled_ids = list(id2sample.keys())[109:]
    model_name = 'gpt-4.1'
    extractor = LLMTripletExtractor(model=model_name)
    current_prompt_token_num, current_completion_token_num = 0, 0

    for i, sample_id in tqdm(enumerate(sampled_ids), total=len(sampled_ids)):
        # texts = ["".join(text[1]) for text in ds[i]['context']]
        triplets = []
        faulty_triplets = []
        final_triplets_source_text_ids = []
        filtered_triplets_source_text_ids = []
        prompt_token_nums, completion_token_nums = [], []

        sample = id2sample[sample_id]
        aligner = Aligner()
        texts = [item['paragraph_text'] for item in sample['paragraphs']]
        
        for idx, text in tqdm(enumerate(texts), total=len(texts)):
            try:
                final_triplets, filtered_triplets = extract_triplets(text)
            except Exception as e:
                print(f"Failed to extract triplets after 3 retries: {str(e)}")
                final_triplets, filtered_triplets = [], []
            triplets.extend(final_triplets)
            faulty_triplets.extend(filtered_triplets)

            final_triplets_source_text_ids.extend([idx for _ in range(len(final_triplets))])
            filtered_triplets_source_text_ids.extend([idx for _ in range(len(filtered_triplets))])

            prompt_token_num, completion_token_num = extractor.calculate_used_tokens()
            prompt_token_nums.extend([prompt_token_num - current_prompt_token_num for _ in range(len(final_triplets))])
            completion_token_nums.extend([completion_token_num - current_completion_token_num for _ in range(len(final_triplets))])
            current_prompt_token_num, current_completion_token_num = prompt_token_num, completion_token_num

        print("CURRENT COST: ", extractor.calculate_cost())
        
        df = pd.DataFrame(triplets)
        df['source_text_ids'] = final_triplets_source_text_ids
        df['prompt_token_num'] = prompt_token_nums
        df['completion_token_num'] = completion_token_nums
        df.to_csv(f"musique_res_gpt-4.1/final_triplets_{str(sample_id)}.csv")

        df_filtered = pd.DataFrame(faulty_triplets)
        df_filtered['source_text_ids'] = filtered_triplets_source_text_ids
        df_filtered.to_csv(f"musique_res_gpt-4.1/faulty_triplets_{str(sample_id)}.csv")
