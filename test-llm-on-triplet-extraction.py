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

from utils.eval_utils import micro_precision, micro_recall
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

############## transform target triplets into tuples with labels and ids to calculate metrics properly ############## 
def transform_targets_syntie(sample):
    
    targets = {"text_triplets": [], "triplet_ids": []}
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

        targets['text_triplets'].append((head_label, rel_label, tail_label))
        targets['triplet_ids'].append((head_id, rel_id, tail_id))

    return targets


############## align extracted triplets to wikidata ids to calculate metrics after the second round of prompting ############## 
def align_outputs(outputs, aligner, triplet_filter, verify=True):
    transformed_output_ids = []
    filtered_triplets = []

    if isinstance(outputs, str):
        if outputs.startswith('Output:'):
            outputs = outputs.replace('Output:', '')
        try:
            outputs = json.loads(outputs)    
        except Exception as e:
            logger.info("Cannot convert input to dictionary: ", outputs)
            return {'transformed_outputs': [], 'filtered_triplets': []}

    for res in outputs:
        rel = res['relation']
        rel = " ".join(rel.split("_")).strip()

        head, tail = res['subject'], res['object']
        head = " ".join(head.split("_")).strip()
        tail = " ".join(tail.split("_")).strip()

        if (rel in aligner.relation2id) and (head in aligner.entity2id) and (tail in aligner.entity2id):
            rel_id = aligner.relation2id[rel]
            head_id = aligner.entity2id[head]
            tail_id = aligner.entity2id[tail]

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
            filtered_triplets.append({"subject": head, "relation": rel, "object": tail})
        
    return {'transformed_outputs': transformed_output_ids, 'filtered_triplets': filtered_triplets}
    


def test_triplet_extraction(dataset, random_items, verbose_step, device='cuda:2'):

    num_items = len(random_items)

    ############## auxilary classes ##############
    aligner = Aligner(device=device)
    triplet_filter = TripletFilter()

    extractor1 = LLMTripletExtractor()
    extractor2 = LLMTripletExtractor()

    target_ids = []



    ############## checking that all the entities and relation from test dataset are indexed ##############
    entities = set()
    relations = set()

    for item in tqdm(dataset):

        for triplet in item['triplets']:
            entities.add(eval(triplet['subject'])['uri'])
            entities.add(eval(triplet['object'])['uri'])
            relations.add(eval(triplet['predicate'])['uri'])

    assert len(set(entities) - set(aligner.id2entity.keys())) == 0
    assert len(set(relations) - set(aligner.id2relation.keys())) == 0

    del entities
    del relations

    ############## arrays to keep results and calculate metrics ##############

    first_step_output_triplets = []
    second_step_output_triplets = []
    first_plus_second_step_output_triplets = []

    for idx, i in tqdm(enumerate(random_items), total=len(random_items)):

        metadata1 = {}
        metadata2 = {}

        i = int(i)
        text = dataset[i]['text']
        target = transform_targets_syntie(dataset['triplets'][i])

        for metadata in [metadata1, metadata2]:
            metadata['index'] = i
            metadata['text'] = text
            metadata['target'] = target
        
        target_ids.append(target['triplet_ids'])

        ############## first step prompting ##############
        extracted_triplets1 = extractor1.get_completion_first_query(text)

        ############## first step entity linking ##############
        first_step_output = align_outputs(extracted_triplets1, aligner=aligner, triplet_filter=triplet_filter, verify=True)
        first_step_output_ids = first_step_output['transformed_outputs'].copy()
        first_step_filtered_triplets = first_step_output['filtered_triplets'].copy()
        
        first_step_output_triplets.append(first_step_output_ids.copy())
        metadata1['first_step_output'] = [(aligner.id2entity[item[0]], aligner.id2relation[item[1]], aligner.id2entity[item[2]]) for item in first_step_output_ids]
        metadata1['first_step_filtered_triplets'] = first_step_filtered_triplets.copy()


        ############## second step aligning all entity and relation names ##############
        try:
            similar_relations = aligner.top_relations_by_llm_output(llm_output=extracted_triplets1)
            similar_entities = aligner.top_entities_by_llm_output(llm_output=extracted_triplets1, entity_type='subject')
            similar_entities.update(aligner.top_entities_by_llm_output(llm_output=extracted_triplets1, entity_type='object'))
        except Exception as e:
            print(e)
            print(extracted_triplets1)

        ############## second step prompting with all aligned entity and relation names ##############

        extractor2.messages = extractor1.messages.copy()
        second_step_output = extractor2.get_completion_second_query(similar_entities=similar_entities, 
            similar_relations=similar_relations, text=text, triplets=extracted_triplets1)

        second_step_output_ids = align_outputs(second_step_output, aligner=aligner, triplet_filter=triplet_filter, verify=True)['transformed_outputs']

        second_step_output_triplets.append(second_step_output_ids.copy())
        metadata2['second_step_output'] = [(aligner.id2entity[item[0]], aligner.id2relation[item[1]], aligner.id2entity[item[2]]) for item in second_step_output_ids]

        assert extractor1.messages[:3] == extractor2.messages[:3]

        ############## second step for first step with filtered triplets aligning ##############

        if len(first_step_filtered_triplets) > 0:

            ############## second step prompting with filtered and aligned entity and relation names ##############

            first_plus_second_step_output = extractor1.get_completion_second_query(similar_entities=similar_entities,
                similar_relations=similar_relations, text=text, triplets=first_step_filtered_triplets)

            first_plus_second_step_output_ids = align_outputs(first_plus_second_step_output, aligner=aligner, \
                 triplet_filter=triplet_filter, verify=True)['transformed_outputs']

            first_plus_second_step_output_full_ids = first_step_output_ids + first_plus_second_step_output_ids

        else:
            first_plus_second_step_output_ids = []
            first_plus_second_step_output_full_ids = first_step_output_ids.copy()
        
        first_plus_second_step_output_triplets.append(first_plus_second_step_output_full_ids.copy())

        metadata1['first_plus_second_step_output'] = [(aligner.id2entity[item[0]], aligner.id2relation[item[1]], aligner.id2entity[item[2]]) \
            for item in first_plus_second_step_output_ids]
        metadata1['first_plus_second_step_output_full'] = [(aligner.id2entity[item[0]], aligner.id2relation[item[1]], aligner.id2entity[item[2]]) \
            for item in first_plus_second_step_output_full_ids]


        ############## logging ##############
        if (idx > 0) and (idx % verbose_step == 0):
            logger.info("First step output precision: " + str(micro_precision(first_step_output_triplets, target_ids)) + 
                "\nFirst step output recall: " + str(micro_recall(first_step_output_triplets, target_ids)) + 
                "\nFirst plus second step output precision: " + str(micro_precision(first_plus_second_step_output_triplets, target_ids)) + 
                "\nFirst plus second  step output recall: " + str(micro_recall(first_plus_second_step_output_triplets, target_ids)) +
                "\nSecond step output precision: " + str(micro_precision(second_step_output_triplets, target_ids)) +
                "\nSecond step output recall: " + str(micro_recall(second_step_output_triplets, target_ids)))

            logger.info("First step output: " + str(metadata1['first_step_output']) + 
                "\nFirst plus second step output: " + str(metadata1['first_plus_second_step_output']) + 
                "\nFirst plus second step output full: " + str(metadata1['first_plus_second_step_output_full']) +
                "\nSecond step output: " + str(metadata2['second_step_output']) +
                "\nTargets: " + str(target['text_triplets']))

        messages1 = extractor1.messages.copy()
        messages2 = extractor2.messages.copy()
        messages1.append(metadata1.copy())
        messages2.append(metadata2.copy())
        
        with jsonlines.open('logs/logs_with_1st_step_filtering_{}.jsonl'.format(str(num_items)), mode='a') as writer:
            writer.write(messages1)

        with jsonlines.open('logs/logs_without_1st_step_filtering_{}.jsonl'.format(str(num_items)), mode='a') as writer:
            writer.write(messages2)
        
        # time.sleep(0.5)

    ############## metrics ##############
    first_step_output_recall, first_step_output_precision = micro_recall(first_step_output_triplets, target_ids), micro_precision(first_step_output_triplets, target_ids)
    second_step_output_recall, second_step_output_precision = micro_recall(second_step_output_triplets, target_ids), micro_precision(second_step_output_triplets, target_ids)
    first_plus_second_step_output_recall, first_plus_second_step_output_precision = micro_recall(first_plus_second_step_output_triplets, target_ids), \
         micro_precision(first_plus_second_step_output_triplets, target_ids)

    cost1 = extractor1.calculate_cost()
    cost2 = extractor2.calculate_cost()


    with jsonlines.open('logs/stat_second_prompt_with_text_and_triplets_wo_first{}_examples.json'.format(str(num_items)), mode='a') as writer:
        writer.write({"second_step_output_recall": second_step_output_recall,
                    "second_step_output_precision": second_step_output_precision, 
                    "first_step_output_recall": first_step_output_recall,
                    "first_step_output_precision": first_step_output_precision,
                    "first_plus_second_step_output_recall": first_plus_second_step_output_recall,
                    "first_plus_second_step_output_precision": first_plus_second_step_output_precision,
                    "cost_stat_1": cost1,
                    "cost_stat_2": cost2})


def main():
    parser = argparse.ArgumentParser("Test")
    parser.add_argument("--dataset_name", help="synthie_code, rebel, synthie_text_pc, synthie_text, synthie_code_pc, rebel_pc", type=str, default='synthie_text')
    parser.add_argument("--split", help="train, test, test_small", type=str, default='test')
    parser.add_argument("--num_items", type=int, default=150)
    parser.add_argument("--verbose_step", type=int, default=15)
    args = parser.parse_args()
    
    dataset = datasets.load_dataset(f"martinjosifoski/SynthIE", args.dataset_name, split=args.split)

    # random_items = np.random.choice(list(range(0, len(dataset))), size=args.num_items, replace=False, )
    idxs = []
    
    with open("logs/old/logs_second_prompt_with_text_and_triplets150_launch2.jsonl", 'r') as f:
        for line in f:
            line = json.loads(line)
            idx = int(line[-1]["index"])
            idxs.append(idx)
    
    random_items = idxs[:150]

    test_triplet_extraction(dataset=dataset, random_items=random_items, verbose_step=args.verbose_step)


if __name__ == "__main__":
    main()