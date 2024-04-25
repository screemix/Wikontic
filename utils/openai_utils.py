import openai
import os
from dotenv import load_dotenv, find_dotenv
from tenacity import retry, wait_random_exponential, before_sleep_log
import logging
import sys
import json
import re

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('KEY')
client = openai.OpenAI(api_key=openai.api_key)


class LLMTripletExtractor:
    def __init__(self, prompt1_path='utils/prompt1.txt', prompt2_path='utils/prompt2.txt', input_price=0.5, output_price=1.5):
        with open(prompt1_path, 'r') as f:
            self.system_prompt_1 = f.read()


        with open(prompt2_path, 'r') as f:
            self.system_prompt_2 = f.read()

        self.model = "gpt-3.5-turbo"
        self.messages = []

        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0
        self.input_price = input_price
        self.output_price = output_price

    @retry(wait=wait_random_exponential(multiplier=1, max=60), before_sleep=before_sleep_log(logger, logging.ERROR))
    def get_completion_first_query(self, text):

        user_prompt = f"""Text:\n{text}"""

        self.messages = [{"role": "system", "content": self.system_prompt_1}, 
                                {
                                    "role": "user",
                                    "content": user_prompt
                                }
                            ]   

       
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0
        )

        self.completion_tokens_num += response.usage.completion_tokens
        self.prompt_tokens_num += response.usage.prompt_tokens

        self.messages.append({'role': 'assistant', 'content': response.choices[0].message.content})
        
        return response.choices[0].message.content.strip()


    @retry(wait=wait_random_exponential(multiplier=1, max=60), before_sleep=before_sleep_log(logger, logging.ERROR))
    def get_completion_second_query(self, similar_relations, similar_entities, text, triplets):

        triplet2names = ''

        if isinstance(triplets, str):
            if triplets.startswith('Output:'):
                triplets = triplets.replace('Output:', '')
            triplets = json.loads(triplets)

        for triplet in triplets:
            subj_mapping = similar_entities[triplet['subject']]
            subj_mapping = triplet['subject'] + ": " + "; ".join(subj_mapping)

            obj_mapping = similar_entities[triplet['object']]
            obj_mapping = triplet['object'] + ": " + "; ".join(obj_mapping)

            rel_mapping = similar_relations[triplet['relation']]
            rel_mapping = triplet['relation'] + ": " + "; ".join(rel_mapping)

            triplet2name = str(triplet) + '\n' + subj_mapping + '\n' + rel_mapping + '\n' + obj_mapping +'\n\n'

            triplet2names = triplet2names + triplet2name
            
        # user_prompt = f"""Text:\n{text}\n\nMapping of top-5 similar relations from Wikidata:\n{similar_relations}\n\nMapping of top-5 similar entities from Wikidata:\n{similar_entities}"""

        user_prompt = f"""Text:\n{text}\n\nTriplets and corresponding entity and relation mappings:\n\n{triplet2names}"""

        print(user_prompt)
        
        self.messages.extend([{"role": "system", "content": self.system_prompt_2}, 
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]   
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0
        )


        # print(response.choices[0].message.content.strip())
        
        self.completion_tokens_num += response.usage.completion_tokens
        self.prompt_tokens_num += response.usage.prompt_tokens

        self.messages.append({'role': 'assistant', 'content': response.choices[0].message.content.strip()})
        
        return response.choices[0].message.content.strip()

    def calculate_cost(self):
        price = 0
        price += self.prompt_tokens_num * self.input_price
        price += self.completion_tokens_num * self.output_price

        return price / (10**6)
