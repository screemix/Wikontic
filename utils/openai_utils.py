import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


openai.api_key  = os.getenv('KEY')
client = openai.OpenAI(api_key=openai.api_key)


with open('utils/prompt1.txt', 'r') as f:
    system_prompt_1 = f.read()


with open('utils/prompt2.txt', 'r') as f:
    system_prompt_2 = f.read()


def get_completion_first_query(text, system_prompt=system_prompt_1, model="gpt-3.5-turbo"):
    user_prompt = f"""Text:\n{text}"""

    messages = [{"role": "system", "content": system_prompt}, 
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]   
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip().split("\n")


def get_completion_second_query(text, triplets, similar_relations, system_prompt=system_prompt_2, model="gpt-3.5-turbo"):
    user_prompt = f"""Text:\n{text}\n\n\
    Extracted triplets:\n{triplets}\n\n\
    Top-5 similar relations from Wikidata:\n{similar_relations}"""
    
    messages = [{"role": "system", "content": system_prompt}, 
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]   
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.split("\n")