from tqdm import tqdm

import json
import jsonlines

from utils.openai_utils import LLMTripletExtractor
from utils.dynamic_index_utils import Aligner

import pandas as pd
import faiss

import warnings
warnings.filterwarnings('ignore')

########################################

def extract_kg_from_texts(texts, texts_id, extractor, device, logs_filename='logs_musique_50.jsonl'):
    # first_step_triplets = []
    generated_triplets = []
    
    aligner = Aligner(device=device)

    for i, text in tqdm(enumerate(texts), total=len(texts)):
        # messages = []
        second_step_triplets = []
        exceptions = []

        ############## first step prompting ##############
        extracted_triplets = extractor.get_completion_first_query(text)
        # extracted_triplets = parse_output(extracted_triplets)

        # first_step_triplets.append(extracted_triplets)

        # messages.append(extractor.messages.copy())

        ############## second step aligning all entity and relation names ##############
        
        for triplet in extracted_triplets:

            try:
                    
                subject_description = triplet['subject'] + "; " + extractor.generate_description_for_entity(text=text, triplet=triplet, entity=triplet['subject'])[triplet['subject']]
                # messages.append(extractor.messages.copy())
                object_description = triplet['object']  + "; " + extractor.generate_description_for_entity(text=text, triplet=triplet, entity=triplet['object'])[triplet['object']]
                # messages.append(extractor.messages.copy())
                relation_description = triplet['relation'] + "; " + \
                    extractor.generate_description_for_relation(text=text, triplet=triplet,  relation=triplet['relation'])[triplet['relation']]
                # messages.append(extractor.messages.copy())

                if len(aligner.id2entity) > 0 and len(aligner.id2relation) > 0:

                    similar_relations_with_descriptions = aligner.top_relations_by_llm_output(relations=[relation_description], with_descriptions=True)
                    similar_entities_with_descriptions = aligner.top_entities_by_llm_output(entities=[subject_description, object_description], with_descriptions=True)

                    similar_relations = aligner.top_relations_by_llm_output(relations=[triplet['relation']], with_descriptions=False)
                    similar_entities = aligner.top_entities_by_llm_output(entities=[triplet['subject'], triplet['object']], with_descriptions=False)
                    
                    for key in similar_relations:
                        similar_relations[key] = list(set(similar_relations[key] + similar_relations_with_descriptions[key]))
                    
                    for key in similar_entities:
                        similar_entities[key] = list(set(similar_entities[key] + similar_entities_with_descriptions[key]))


                    output = extractor.get_completion_second_query_by_single_triplet(similar_entities=similar_entities, 
                        similar_relations=similar_relations, text=text, triplet=triplet)
                    
                    # messages.append(extractor.messages.copy())

                    if output['subject'] == 'None' or output['subject'] == None:
                        aligner.add_entities([triplet['subject']], [subject_description])
                        output['subject'] = triplet['subject']

                    if output['object'] == 'None' or output['object'] == None:
                        aligner.add_entities([triplet['object']], [object_description])
                        output['object'] = triplet['object']

                    if output['relation'] == 'None' or output['relation'] == None:
                        aligner.add_relations([triplet['relation']], [relation_description])
                        output['relation'] = triplet['relation']
                    
                    second_step_triplets.append(output.copy())
                    generated_triplets.append(output.copy())
                
                else:
                    aligner.add_entities([triplet['subject'], triplet['object']], [subject_description, object_description])
                    aligner.add_relations([triplet['relation']], [relation_description])
                    
                    second_step_triplets.append(triplet)
                    generated_triplets.append(triplet.copy())
            
            except Exception as e:
                exceptions.append(f'Exception {e} on triplet {triplet} extracted triplets:')
                print(str(e))
            
        cost = extractor.calculate_cost()            
        log_output = {"sample_id": texts_id, "text_id": i, "cost": cost, "1-step triplets": extracted_triplets, \
                '2-step triplets': second_step_triplets, 'exceptions': exceptions}        
        with jsonlines.open(f'logs/{logs_filename}', mode='a') as writer:
            writer.write(log_output)

        # generated_triplets.append(second_step_triplets.copy())
            

    with open(f'musique_indices/id2relation_{texts_id}.json', 'w') as f:
        json.dump(aligner.id2relation, f)
    
    with open(f'musique_indices/id2relation_with_description_{texts_id}.json', 'w') as f:
        json.dump(aligner.id2relation_with_description, f)

    with open(f'musique_indices/id2entity_{texts_id}.json', 'w') as f:
        json.dump(aligner.id2entity, f)
    
    with open(f'musique_indices/id2entity_with_description_{texts_id}.json', 'w') as f:
        json.dump(aligner.id2entity_with_description, f)

    # graph_triplets = []
    # for item in generated_triplets:
    #     graph_triplets.extend(item)
        
    df = pd.DataFrame(generated_triplets)
    df = df.drop_duplicates()

    return df

########################################

with open("musique_200_test.json", "r") as f:
    ds = json.load(f)

ds = ds['data']

id2sample = {}
for elem in ds:
    id2sample[elem['id']] = elem


exception_items = []
with open('logs/logs_musique_50.jsonl', 'r') as f:
    for line in f:
        
        messages = json.loads(line)

        if messages['exceptions']:
            exception_items.append(messages)

exception_items = [(elem['sample_id'], elem['text_id']) for elem in exception_items]

device = 'cuda:4'
model_name = 'gpt-4o'
extractor = LLMTripletExtractor(model=model_name,  prompt2_individual_triplets_path='utils/prompts/prompt2_individual_triplets_dynamic.txt')

# N = 50
# ids = [sample['id'] for sample in ds[:N]]

# for i, elem in zip(ids, ds[:N]):
for sample_id, text_id in exception_items:
    # texts = ["".join(text[1]) for text in ds[i]['context']]
    sample = id2sample[sample_id]

    texts = [sample['paragraphs'][text_id]['paragraph_text']]
    new_df = extract_kg_from_texts(texts, text_id, extractor, device)

    old_df = pd.read_csv(f"musique_res/{str(sample_id)}.csv")
    print(f"Previous # of triplets: {len(old_df)}")

    df = pd.concat([new_df, old_df])
    print(f"Updated # of triplets: {len(df)}")

    df.to_csv(f"musique_res/{str(sample_id)}.csv")