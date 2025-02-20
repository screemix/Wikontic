import openai
import os
from dotenv import load_dotenv, find_dotenv
from tenacity import retry, wait_random_exponential, before_sleep_log
import logging
import sys
import json
import re
from pydantic import BaseModel
from typing import List
from dataclasses import asdict, dataclass

logging.basicConfig(stream=sys.stderr)

logger = logging.getLogger('OpenAIUtils')
logger.setLevel(logging.ERROR)

_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('KEY')
client = openai.OpenAI(api_key=openai.api_key)

@dataclass
class Qualifier(BaseModel):
    relation: str
    object_: str

@dataclass
class Triplet(BaseModel):
    subject: str
    relation: str
    object_: str
    qualifiers: List[Qualifier]
    subject_type: str
    object_type: str


@dataclass
class Triplets(BaseModel):
    triplets: List[Triplet]


class LLMTripletExtractor:
    def __init__(self, prompt1_path='utils/prompts/prompt1.txt', prompt2_path='utils/prompts/prompt2.txt', 
            prompt2_individual_triplets_path='utils/prompts/prompt2_individual_triplets.txt',
            entity_description_prompt_path='utils/prompts/prompt_entity_descriptions.txt', 
            relation_description_prompt_path='utils/prompts/prompt_relation_descriptions.txt', 
            triplet_from_question_extraction_prompt_path='utils/prompts/prompt_triplet_extraction_from_question.txt',
            relation_description_from_question_prompt_path='utils/prompts/prompt_relation_description_from_question.txt',
            entity_description_from_question_prompt_path='utils/prompts/prompt_entity_description_from_question.txt',
            entity_from_question_prompt_path='utils/prompts/prompt_entity_extraction_from_question.txt',
            qa_prompt_path='utils/prompts/qa_prompt.txt',
            model="gpt-3.5-turbo", input_price=5, output_price=15):


        with open(prompt1_path, 'r') as f:
            self.system_prompt_1 = f.read()

        with open(prompt2_path, 'r') as f:
            self.system_prompt_2 = f.read()

        # for individual triplet refinement

        with open(prompt2_individual_triplets_path, 'r') as f:
            self.prompt2_individual_triplets = f.read()

        # for description generation

        with open(entity_description_prompt_path, 'r') as f:
            self.entity_description_prompt = f.read()
        
        with open(relation_description_prompt_path, 'r') as f:
            self.relation_description_prompt = f.read()

        # for questions

        with open(triplet_from_question_extraction_prompt_path, 'r') as f:
            self.triplet_from_question_extraction_prompt = f.read()

        with open(entity_description_from_question_prompt_path, 'r') as f:
            self.entity_description_from_question_prompt = f.read()
        
        with open(relation_description_from_question_prompt_path, 'r') as f:
            self.relation_description_from_question_prompt = f.read()

        with open(entity_from_question_prompt_path, 'r') as f:
            self.entity_from_question_prompt = f.read()

        with open(qa_prompt_path, 'r') as f:
            self.qa_prompt= f.read()


        self.model = model
        self.messages = []

        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0
        self.input_price = input_price
        self.output_price = output_price


    def parse_output(self, output):
        try:
            pattern = r'(\{.*\})'
            match = re.search(pattern, output, re.DOTALL)
            match = match.group(1)
            return json.loads(match)

        except Exception as e:
            return output


    # @retry(wait=wait_random_exponential(multiplier=1, max=60), before_sleep=before_sleep_log(logger, logging.ERROR))
    def get_completion(self, user_prompt, system_prompt, transform_to_json=True):

        messages = [
                        {
                            "role": "system", 
                            "content": system_prompt
                        }, 

                        {
                            "role": "user",
                            "content": user_prompt
                        },
                    ]

        # response = client.beta.chat.completions.parse(
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            # response_format=Triplets
        )

        self.completion_tokens_num += response.usage.completion_tokens
        self.prompt_tokens_num += response.usage.prompt_tokens

        if transform_to_json:
            try:
                output = json.loads(response.choices[0].message.content.strip())
            except Exception as e:
                output = self.parse_output(response.choices[0].message.content.strip())
                # output = response.choices[0].message.content.strip()
        else:
            output = response.choices[0].message.content.strip()

        self.messages = messages
        self.messages.append({'role': 'assistant', 'content': output})

        return output


    def get_completion_first_query(self, text):

        user_prompt = f'''"Text: "{text}"'''
        return self.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt_1)
        

    def generate_description_for_entity(self, text, triplet, entity):
        
        user_prompt = f"""Text: {text}\n"Triplet: "{triplet}\n\nEntity:\n{entity}"""

        return self.get_completion(user_prompt=user_prompt, system_prompt=self.entity_description_prompt)

    
    def generate_description_for_relation(self, text, triplet, relation):
        
        user_prompt = f"""Text: {text}\n"Triplet: "{triplet}\n\nRelation: "{relation}"""

        return self.get_completion(user_prompt=user_prompt, system_prompt=self.relation_description_prompt)


    def get_completion_second_query(self, similar_entities, similar_relations, text, triplets):

        triplet2names = ''
        triplets = triplets

        for triplet in triplets:
            subj_mapping = similar_entities[triplet['subject']]
            subj_mapping = '{"' + triplet['subject'] + '": ' + json.dumps(subj_mapping) + '}'

            obj_mapping = similar_entities[triplet['object']]
            obj_mapping = '{"' + triplet['object'] + '": ' + json.dumps(obj_mapping) + '}'

            rel_mapping = similar_relations[triplet['relation']]
            rel_mapping = '{"' + triplet['relation'] + '": ' + json.dumps(rel_mapping) + '}'

            triplet2name = '\t' + json.dumps(triplet) + '\n\t' + subj_mapping + '\n\t' + rel_mapping + '\n\t' + obj_mapping +'\n\n'

            triplet2names = triplet2names + triplet2name
            
        user_prompt = f"""Text: {text}\n\nTriplets and corresponding entity and relation mappings:\n\n{triplet2names}"""

        return self.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt_2)
    

    def get_completion_second_query_by_single_triplet(self, similar_entities, similar_relations, text, triplet):

        subj_mapping = similar_entities[triplet['subject']]
        subj_mapping = '{"' + triplet['subject'] + '": ' + json.dumps(subj_mapping) + '}'

        obj_mapping = similar_entities[triplet['object']]
        obj_mapping = '{"' + triplet['object'] + '": ' + json.dumps(obj_mapping) + '}'

        rel_mapping = similar_relations[triplet['relation']]
        rel_mapping = '{"' + triplet['relation'] + '": ' + json.dumps(rel_mapping) + '}'

        text_triplet = '\t' + json.dumps(triplet)
        text_mapping = '\t' + subj_mapping + '\n\t' + rel_mapping + '\n\t' + obj_mapping

            
        user_prompt = f"""Text: {text}\n\nTriplet:\n{text_triplet}\n\nEntity and relation mappings:\n{text_mapping}"""
        
        return self.get_completion(user_prompt=user_prompt, system_prompt=self.prompt2_individual_triplets)
    

    def extract_triplets_from_question(self, question):
        
        user_prompt = f"""Question: {question}"""

        return self.get_completion(user_prompt=user_prompt, system_prompt=self.triplet_from_question_extraction_prompt)
    

    def extract_entities_from_question(self, question):
        
        user_prompt = f"""Question: {question}"""

        return self.get_completion(user_prompt=user_prompt, system_prompt=self.entity_from_question_prompt)


    def generate_description_for_entity_from_question(self, text, triplet, entity):
        
        user_prompt = f"""Text: {text}\n"Triplet: "{triplet}\n\nEntity:\n{entity}"""

        return self.get_completion(user_prompt=user_prompt, system_prompt=self.entity_description_from_question_prompt)

    
    def generate_description_for_relation_from_question(self, text, triplet, relation):
        
        user_prompt = f"""Text: {text}\n"Triplet: "{triplet}\n\nRelation: "{relation}"""

        return self.get_completion(user_prompt=user_prompt, system_prompt=self.relation_description_from_question_prompt)

    def answer_question(self, question, triplets):
        user_prompt = f"""Question: {question}\n\n"Triplets: "{triplets}"""
        return self.get_completion(user_prompt=user_prompt, system_prompt=self.qa_prompt, transform_to_json=False)


    def calculate_cost(self):
        return (self.prompt_tokens_num * self.input_price + self.completion_tokens_num * self.output_price) / (10**6)
