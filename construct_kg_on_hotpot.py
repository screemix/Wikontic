from tqdm import tqdm

import json
import jsonlines

from utils.openai_utils import LLMTripletExtractor
from utils.dynamic_index_utils import Aligner

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

########################################

with open("hotpotqa200.json", "r") as f:
    ds = json.load(f)

device = 'cuda:4'
model_name = 'gpt-4o'

extractor = LLMTripletExtractor(model=model_name,  prompt2_individual_triplets_path='utils/prompts/prompt2_individual_triplets_dynamic.txt')

def extract_kg_from_texts(texts):
    first_step_triplets = []
    second_step_triplets = []
    generated_triplets = []
    
    aligner = Aligner(device=device)
    logs_filename = f'logs/logs_hotpot200.jsonl'

    for i, text in tqdm(enumerate(texts), total=len(texts)):
        messages = []

        ############## first step prompting ##############
        extracted_triplets = extractor.get_completion_first_query(text)
        # extracted_triplets = parse_output(extracted_triplets)
        first_step_triplets.append(extracted_triplets)

        messages.append(extractor.messages.copy())

        ############## second step aligning all entity and relation names ##############
        
        for triplet in extracted_triplets:

            try:
                    
                subject_description = triplet['subject'] + "; " + extractor.generate_description_for_entity(text=text, triplet=triplet, entity=triplet['subject'])[triplet['subject']]
                messages.append(extractor.messages.copy())
                object_description = triplet['object']  + "; " + extractor.generate_description_for_entity(text=text, triplet=triplet, entity=triplet['object'])[triplet['object']]
                messages.append(extractor.messages.copy())
                relation_description = triplet['relation'] + "; " + \
                    extractor.generate_description_for_relation(text=text, triplet=triplet,  relation=triplet['relation'])[triplet['relation']]
                messages.append(extractor.messages.copy())

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
                    
                    messages.append(extractor.messages.copy())

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
                
                else:
                    aligner.add_entities([triplet['subject'], triplet['object']], [subject_description, object_description])
                    aligner.add_relations([triplet['relation']], [relation_description])
                    
                    second_step_triplets.append(triplet)
                
                generated_triplets.append(second_step_triplets)
                
                cost = extractor.calculate_cost()
                messages.append({"cost": cost})
                        
                with jsonlines.open(logs_filename, mode='a') as writer:
                    writer.write(messages)
            
            except Exception as e:
                print(str(e))

    
    graph_triplets = []
    for item in generated_triplets:
        graph_triplets.extend(item)
        
    df = pd.DataFrame(graph_triplets)
    df = df.drop_duplicates()

    return df

for i, elem in enumerate(ds[:50]):
    texts = ["".join(text[1]) for text in ds[i]['context']]
    df = extract_kg_from_texts(texts)
    df.to_csv(f"hotpot200_res/{str(i)}.csv")