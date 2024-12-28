import time
import numpy as np
from tqdm import tqdm
import datasets

import json
import jsonlines
import openai
import os
import sys
import logging
import pprint

from utils.eval_utils import micro_precision, micro_recall, f1_score
from utils.openai_utils import LLMTripletExtractor
from utils.index_utils import Aligner
from utils.verifier_utils import TripletFilter

import argparse
import re
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(stream=sys.stderr)

logger = logging.getLogger('Main')
logger.setLevel(logging.INFO)

############## formatting logs ############## 
pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)

def log_pretty(obj):
    pretty_out = f"{pp.pformat(obj)}"

    return f'{pretty_out}\n'

############## transform target triplets into tuples with labels and ids to calculate metrics properly ############## 
def transform_targets_syntie(sample):
    
    targets = {"triplet_texts": [], "triplet_ids": []}
    for t in sample:
        rel = eval(t['predicate'])
        rel_label = rel['surfaceform'].strip()
        rel_id = rel['uri']
        
        head = eval(t['subject'])
        head_label = " ".join(head['surfaceform'].split("_")).strip()
        head_id = head['uri']

        tail =  eval(t['object'])
        tail_label = " ".join(tail['surfaceform'].split("_")).strip()
        tail_id = tail['uri']

        targets['triplet_texts'].append((head_label, rel_label, tail_label))
        targets['triplet_ids'].append((head_id, rel_id, tail_id))

    return targets


############## align extracted triplets to wikidata ids ############## 
def align_outputs(outputs, aligner, triplet_filter, verify=True):
    transformed_output_ids = []
    filtered_triplets = []
    faulty_choice_triplets = []

    for res in outputs:

        rel = res['relation']
        rel = " ".join(rel.split("_")).strip()

        head, tail = res['subject'], res['object']
        head = " ".join(head.split("_")).strip()
        tail = " ".join(tail.split("_")).strip()

        if (rel in aligner.relation2id) and (head in aligner.entity2id) and (tail in aligner.entity2id):
            rel_id = aligner.relation2id[rel].split(";")[0]
            head_id = aligner.entity2id[head].split(";")[0]
            tail_id = aligner.entity2id[tail].split(";")[0]

            if verify:

                if triplet_filter.check_triplet_validity(head_id, rel_id, tail_id):
                    transformed_output_ids.append((head_id, rel_id, tail_id))

                elif triplet_filter.check_triplet_validity(tail_id, rel_id, head_id):
                    transformed_output_ids.append((tail_id, rel_id, head_id))
                
                else:
                    filtered_triplets.append({"subject": head, "relation": rel, "object": tail})

            else:
                transformed_output_ids.append((head_id, rel_id, tail_id))
                
        else:
            faulty_choice_triplets.append({"subject": head, "relation": rel, "object": tail})
        
    return {'transformed_outputs': transformed_output_ids, 'filtered_triplets': filtered_triplets, "faulty_choice_triplets": faulty_choice_triplets}

def calculate_and_represent_metrics(target_ids, first_step_output_triplets, first_step_unverified_output_triplets,
                                    second_step_output_triplets, second_step_unverified_output_triplets, second_step_faulty_choice, second_step_empty_choice, cost):
                                    
    first_step_recall, first_step_precision = micro_recall(first_step_output_triplets, target_ids), \
        micro_precision(first_step_output_triplets, target_ids)

    first_step_unverified_recall, first_step_unverified_precision = micro_recall(first_step_unverified_output_triplets, target_ids), \
        micro_precision(first_step_unverified_output_triplets, target_ids)
            
    second_step_recall, second_step_precision = micro_recall(second_step_output_triplets, target_ids), \
        micro_precision(second_step_output_triplets, target_ids)

    second_step_unverified_recall, second_step_unverified_precision = micro_recall(second_step_unverified_output_triplets, target_ids), \
        micro_precision(second_step_unverified_output_triplets, target_ids)

    first_step_f1 = f1_score(first_step_precision, first_step_recall)
    first_step_unverified_f1 = f1_score(first_step_unverified_precision, first_step_unverified_recall)
                
    second_step_f1 = f1_score(second_step_precision, second_step_recall)
    second_step_unverified_f1 = f1_score(second_step_unverified_precision, second_step_unverified_recall)

    second_step_faulty_choice_ratio = sum(second_step_faulty_choice) / len(second_step_faulty_choice)
    second_step_empty_choice_ratio = sum(second_step_empty_choice) / len(second_step_empty_choice)

    res =  {       
        "First step unverified precision": first_step_unverified_precision,
        "First step unverified recall": first_step_unverified_recall,
        "first step unverified f1": first_step_unverified_f1,

        "First step precision": first_step_precision,
        "First step recall": first_step_recall,
        "First step f1": first_step_f1,

        "Second step unverified precision": second_step_unverified_precision, 
        "Second step unverified recall": second_step_unverified_recall,
        "Second step unverified f1": second_step_unverified_f1,

        "Second step precision": second_step_precision, 
        "Second step recall": second_step_recall,
        "Second step f1": second_step_f1,
        
        "Second step faulty choice ratio": second_step_faulty_choice_ratio,
        "Second step empty choice ratio": second_step_empty_choice_ratio,
        
        "Cost stat": cost,
    }
    
    return res


def test_triplet_extraction(dataset, dataset_name, split_name, model_name, 
                            prompt1_path, prompt2_path, dataset_items, verbose_step, 
                            launch_id, launch_description, device='cuda:2', file_logging=True):

    num_items = len(dataset_items)

    ############## auxilary classes ##############
    aligner = Aligner(device=device)

    triplet_filter = TripletFilter()
    extractor = LLMTripletExtractor(model=model_name, prompt1_path=prompt1_path, prompt2_path=prompt2_path)

    ############## checking that all the entities and relation from test dataset are indexed ##############
    # entities = set()
    # relations = set()

    # for item in tqdm(dataset):

    #     for triplet in item['triplets']:
    #         entities.add(eval(triplet['subject'])['uri'])
    #         entities.add(eval(triplet['object'])['uri'])
    #         relations.add(eval(triplet['predicate'])['uri'])

    # assert len(set(entities) - set(aligner.id2entity.keys())) == 0
    # assert len(set(relations) - set(aligner.id2relation.keys())) == 0

    # del entities
    # del relations

    ############## arrays to keep results and calculate metrics ##############

    target_ids = []
    first_step_output_triplets = []
    first_step_unverified_output_triplets = []
    
    second_step_output_triplets = []
    second_step_unverified_output_triplets = []
    
    second_step_faulty_choice = []
    second_step_empty_choice = []

    ############################ evaluation loop ############################

    for idx, i in tqdm(enumerate(dataset_items), total=len(dataset_items)):
        try:
            metadata = {}

            i = int(i)
            text = dataset[i]['text']
            target = transform_targets_syntie(dataset['triplets'][i])

            metadata['index'] = i
            metadata['text'] = text
            metadata['target'] = target
            
            target_ids.append(target['triplet_ids'])

            ############## first step prompting ##############
            extracted_triplets = extractor.get_completion_first_query(text)
            messages = extractor.messages.copy()


            ############## second step prompting for choosing among aligned entity and relation names, each triplet individually ##############
            empty_choice_num = 0
            second_step_output = []

            entity2desc = []
            relation2desc = []

            for triplet in extracted_triplets:
                
                subject_description = triplet['subject'] + "; " + extractor.generate_description_for_entity(text=text, triplet=triplet, entity=triplet['subject'])[triplet['subject']]
                messages.append(extractor.messages.copy())
                
                object_description = triplet['object']  + "; " + extractor.generate_description_for_entity(text=text, triplet=triplet, entity=triplet['object'])[triplet['object']]
                messages.append(extractor.messages.copy())

                entity2desc.append(subject_description)
                entity2desc.append(object_description)
                
                relation_description = triplet['relation'] + "; " + \
                    extractor.generate_description_for_relation(text=text, triplet=triplet,  relation=triplet['relation'])[triplet['relation']]
                messages.append(extractor.messages.copy())
                
                relation2desc.append(relation_description)

                similar_relations_with_descriptions = aligner.top_relations_by_llm_output(relations=[relation_description], with_descriptions=True)
                similar_entities_with_descriptions = aligner.top_entities_by_llm_output(entities=[subject_description, object_description], with_descriptions=True)

                similar_relations = aligner.top_relations_by_llm_output(relations=[triplet['relation']], with_descriptions=False)
                similar_entities = aligner.top_entities_by_llm_output(entities=[triplet['subject'], triplet['object']], with_descriptions=False)
                
                mappings_filename = f'logs/relation_entity_mappings_{launch_description}_{dataset_name}_{split_name}_{model_name}_{str(num_items)}_launch_{str(launch_id)}.jsonl'
                with jsonlines.open(mappings_filename, mode='a') as writer:
                    mappings = {
                                    "similar_relations_with_descriptions": similar_relations_with_descriptions,
                                    "similar_entities_with_descriptions":  similar_entities_with_descriptions,
                                    "similar_relations": similar_relations,
                                    "similar_entities": similar_entities
                                }
                    writer.write(mappings)
            
                for key in similar_relations:
                    similar_relations[key] = list(set(similar_relations[key] + similar_relations_with_descriptions[key]))
                
                for key in similar_entities:
                    similar_entities[key] = list(set(similar_entities[key] + similar_entities_with_descriptions[key]))

                output = extractor.get_completion_second_query_by_single_triplet(similar_entities=similar_entities, 
                    similar_relations=similar_relations, text=text, triplet=triplet)
                
                if output:
                    second_step_output.append(output)
                else:
                    empty_choice_num += 1

            ############## first step entity linking, with constraints verification ##############
            first_step_output = align_outputs(extracted_triplets, aligner=aligner, triplet_filter=triplet_filter, verify=True)
            first_step_output_ids = first_step_output['transformed_outputs'].copy()

            ############## first step entity linking, without constraints verification ##############
            first_step_unverified_output = align_outputs(extracted_triplets, aligner=aligner, triplet_filter=triplet_filter, verify=False)
            first_step_unverified_output_ids = first_step_unverified_output['transformed_outputs'].copy()

            ############## second step entity linking, with constraints verification ##############
            aligned_second_step = align_outputs(second_step_output, aligner=aligner, triplet_filter=triplet_filter, verify=True)
            second_step_output_ids, faulty_choice_triplets = aligned_second_step['transformed_outputs'], aligned_second_step['faulty_choice_triplets']

            ############## second step entity linking, without constraints verification ##############
            second_step_unverified_output_ids = align_outputs(second_step_output, aligner=aligner, triplet_filter=triplet_filter, verify=False)['transformed_outputs']

            ################ saving verbalized outputs on both steps ################
            first_step_output_triplets.append(first_step_output_ids.copy())
            metadata['first_step_output'] = [(aligner.id2entity[item[0]], aligner.id2relation[item[1]], aligner.id2entity[item[2]]) \
                for item in first_step_output_ids]

            first_step_unverified_output_triplets.append(first_step_unverified_output_ids.copy())
            metadata['first_step_unverified_output'] = [(aligner.id2entity[item[0]], aligner.id2relation[item[1]], aligner.id2entity[item[2]]) \
                for item in first_step_unverified_output_ids]

            second_step_output_triplets.append(second_step_output_ids.copy())

            metadata['entity2description'] = entity2desc.copy()
            metadata['relation2description'] = relation2desc.copy()
            
            metadata['second_step_output'] = [(aligner.id2entity[item[0]], aligner.id2relation[item[1]], aligner.id2entity[item[2]]) \
                for item in second_step_output_ids]

            second_step_unverified_output_triplets.append(second_step_unverified_output_ids.copy())
            metadata['second_step_unverified_output'] = [(aligner.id2entity[item[0]], aligner.id2relation[item[1]], aligner.id2entity[item[2]]) \
                for item in second_step_unverified_output_ids]

            second_step_faulty_choice.append(len(faulty_choice_triplets) / len(second_step_output) if second_step_output else 0)
            second_step_empty_choice.append(empty_choice_num / len(extracted_triplets))

            ############## logging ##############
            if (idx > 0) and (idx % verbose_step == 0):
                
                cost = extractor.calculate_cost()
                metrics_log = calculate_and_represent_metrics(target_ids, first_step_output_triplets, first_step_unverified_output_triplets,
                                    second_step_output_triplets, second_step_unverified_output_triplets, second_step_faulty_choice, second_step_empty_choice, cost)

                logger.info(log_pretty(metrics_log))

                logger.info("\nText: " + text +
                    "\nFirst step output: " + str(metadata['first_step_output']) + 
                    "\nSecond step output: " + str(metadata['second_step_output']) +
                    "\nTargets: " + str(target['triplet_texts'])+'\n')

            messages.append(extractor.messages.copy())
            messages.append(metadata.copy())

            if file_logging:
                logs_filename = f'logs/logs_{launch_description}_{dataset_name}_{split_name}_{model_name}_{str(num_items)}_launch_{str(launch_id)}.jsonl'
                with jsonlines.open(logs_filename, mode='a') as writer:
                    writer.write(messages)
            
        except Exception as e:
            error_logs_filename = f'logs/errors_{launch_description}_{dataset_name}_{split_name}_{model_name}_{str(num_items)}_launch_{str(launch_id)}.jsonl'
            with jsonlines.open(error_logs_filename, mode='a') as writer:
                writer.write(str(e))
            logger.info("ERROR: ", str(e))


    ############## final metrics calculation ##############
    cost = extractor.calculate_cost()
    metrics_log = calculate_and_represent_metrics(target_ids, first_step_output_triplets, first_step_unverified_output_triplets,
                    second_step_output_triplets, second_step_unverified_output_triplets, second_step_faulty_choice, second_step_empty_choice, cost)

    if file_logging:
        stat_filename = f'logs/stat_{launch_description}_{dataset_name}_{split_name}_{model_name}_{str(num_items)}_examples_launch_{launch_id}.json'
        with jsonlines.open(stat_filename, mode='a') as writer:
            writer.write(metrics_log)


def main():
    parser = argparse.ArgumentParser("Test")
    parser.add_argument("--dataset_name", help="synthie_code, rebel, synthie_text_pc, synthie_text, synthie_code_pc, rebel_pc", type=str, default='synthie_text')
    parser.add_argument("--split", help="train, test, test_small", type=str, default='test')
    parser.add_argument("--num_items", type=int, default=100)
    parser.add_argument("--verbose_step", type=int, default=10)
    parser.add_argument("--launch_id", type=int, default=0)
    parser.add_argument("--launch_description", type=str, default='')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--prompt_1_filename", type=str, default='utils/prompts/prompt1.txt')
    parser.add_argument("--prompt_2_filename", type=str, default='utils/prompts/prompt2.txt')
    parser.add_argument("--file_logging", type=bool, default=True)
    args = parser.parse_args()
    
    dataset = datasets.load_dataset(f"martinjosifoski/SynthIE", args.dataset_name, split=args.split)

    # random_items = np.random.choice(list(range(0, len(dataset))), size=args.num_items, replace=False, )

    # random_items = list(range(4162, len(dataset)))

    random_items = []
    with open('eval_ids.json', 'r') as f:
        random_items = json.load(f)

    random_items = random_items[:args.num_items]
    test_triplet_extraction(dataset=dataset, dataset_name=args.dataset_name, split_name=args.split, model_name=args.model_name, prompt1_path=args.prompt_1_filename, \
        prompt2_path=args.prompt_2_filename, dataset_items=random_items, verbose_step=args.verbose_step, launch_id=args.launch_id, launch_description=args.launch_description, 
        device=args.device, file_logging=args.file_logging)


if __name__ == "__main__":
    main()