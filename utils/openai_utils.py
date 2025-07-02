import openai
import os
from dotenv import load_dotenv, find_dotenv
from tenacity import retry, wait_random_exponential, before_sleep_log
import logging
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Union, Optional

import httpx

# proxy_url = "socks5://localhost:1080"

# proxies = {
#     "http://": proxy_url,
#     "https://": proxy_url,
# }

# http_client = httpx.Client(proxy=proxy_url)

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger('OpenAIUtils')
logger.setLevel(logging.ERROR)

# Initialize OpenAI client
_ = load_dotenv(find_dotenv())
client = openai.OpenAI(
    api_key=os.getenv("KEY"),
    # http_client=http_client
)

class LLMTripletExtractor:
    """A class for extracting and processing knowledge graph triplets using OpenAI's LLMs."""
    
    MODEL_PRICES = {
        "gpt-4o": {"input": 2.5, "output": 10},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
        "gpt-4.1": {"input": 2.0, "output": 8.0}
    }

    def __init__(self, 
                 prompt_folder_path: str = 'utils/prompts/',
                 system_prompt_paths: Optional[Dict[str, str]] = None,
                 model: str = "gpt-4o"):
        """
        Initialize the LLMTripletExtractor.
        
        Args:
            prompt_folder_path: Path to folder containing prompt files
            system_prompt_paths: Dictionary mapping prompt types to file paths
            model: Name of the OpenAI model to use
        """
        if system_prompt_paths is None:
            system_prompt_paths = {
                'triplet_extraction': 'propmt_1_types_qualifiers.txt',
                'relation_ranker': 'prompt_choose_relation_and_types.txt', 
                'subject_ranker': 'rank_subject_names.txt',
                'object_ranker': 'rank_object_names.txt',
                'question_entity_extractor': 'prompt_entity_relation_extraction_from_question.txt',
                'question_entity_ranker': 'prompt_choose_relevant_entities_for_question.txt',
                # 'qa': 'qa_prompt_hotpot.txt'
                'qa': 'qa_prompt.txt'
            }

        # Load all prompts
        prompt_folder = Path(prompt_folder_path)
        self.prompts = {}
        for prompt_type, filename in system_prompt_paths.items():
            with open(prompt_folder / filename) as f:
                self.prompts[prompt_type] = f.read()

        self.model = model
        self.messages = []
        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0
        
        # Set pricing
        if model not in self.MODEL_PRICES:
            raise ValueError(f"Unknown model: {model}")
        self.input_price = self.MODEL_PRICES[model]["input"]
        self.output_price = self.MODEL_PRICES[model]["output"]

    def extract_json(self, text: str) -> Union[dict, list, str]:
        """Extract JSON from text, handling both code blocks and inline JSON."""
        patterns = [
            r"```json\s*(\{.*?\}|\[.*?\])\s*```",  # JSON in code blocks
            r"(\{.*?\}|\[.*?\])"  # Inline JSON
        ]

        # print(text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {text}")
                    
        return text

    @retry(wait=wait_random_exponential(multiplier=1, max=60), 
           before_sleep=before_sleep_log(logger, logging.ERROR))
    def get_completion(self, system_prompt: str, user_prompt: str, 
                      transform_to_json: bool = True) -> Union[dict, list, str]:
        """Get completion from OpenAI API with retry logic."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        self.completion_tokens_num += response.usage.completion_tokens
        self.prompt_tokens_num += response.usage.prompt_tokens
        
        content = response.choices[0].message.content.strip()
        output = self.extract_json(content) if transform_to_json else content

        self.messages = messages + [{'role': 'assistant', 'content': output}]
        return output

    def extract_triplets_from_text(self, text: str) -> dict:
        """Extract knowledge graph triplets from text."""
        # return {"system_prompt": self.prompts['triplet_extraction'],
        #     "user_prompt": f'Text: "{text}"'}
        return self.get_completion(
            system_prompt=self.prompts['triplet_extraction'],
            user_prompt=f'Text: "{text}"'
        )

    def refine_relation_and_entity_types(self, text: str, triplet: dict, 
                                       candidate_triplets: List[dict]) -> dict:
        """Refine relations and entity types using candidate backbone triplets."""
        triplet_filtered = {k: triplet[k] for k in 
                          ['subject', 'relation', 'object', 'subject_type', 'object_type']}
        
        candidates_str = "".join(f"{json.dumps(c)}\n" for c in candidate_triplets)

        # return {"system_prompt": self.prompts['relation_ranker'],
        #         "user_prompt": f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\n'}
        #                f'Candidate Triplets: {candidates_str}"'
        # print(f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\nCandidate Triplets: {candidates_str}')
        # print(f'Text: {text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\nCandidate Triplets: {candidates_str}')
        return self.get_completion(
            system_prompt=self.prompts['relation_ranker'],
            user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\n'
                       f'Candidate Triplets: {candidates_str}"'
        )

    def refine_entity(self, text: str, triplet: dict, candidates: List[str], 
                     is_object: bool = False) -> dict:
        """Refine subject/object names using candidate options from pre-built KG."""
        triplet_filtered = {k: triplet[k] for k in ['subject', 'relation', 'object']}
        original_name = triplet_filtered['object' if is_object else 'subject']
        
        prompt_key = 'object_ranker' if is_object else 'subject_ranker'
        entity_type = 'Object' if is_object else 'Subject'

        # return {"system_prompt": self.prompts[prompt_key],
        #     "user_prompt": f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\n'
        #                f'Original {entity_type}: {original_name}\n'
        #                f'Candidate {entity_type}s: {json.dumps(candidates)}"'}
        
        return self.get_completion(
            system_prompt=self.prompts[prompt_key],
            user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\n'
                       f'Original {entity_type}: {original_name}\n'
                       f'Candidate {entity_type}s: {json.dumps(candidates)}"'
        )

    def extract_entities_from_question(self, question: str) -> dict:
        """Extract entities from a question."""
        return self.get_completion(
            system_prompt=self.prompts['question_entity_extractor'],
            user_prompt=f"Question: {question}"
        )

    def identify_relevant_entities(self, question: str, entity_list: List[str]) -> List[str]:
        """Identify entities relevant to a question."""
        return self.get_completion(
            system_prompt=self.prompts['question_entity_ranker'],
            user_prompt=f"Question: {question}\nEntities: {entity_list}"
        )

    def answer_question(self, question: str, triplets: List[dict]) -> str:
        """Answer a question using knowledge graph triplets."""
        return self.get_completion(
            system_prompt=self.prompts['qa'],
            user_prompt=f'Question: {question}\n\nTriplets: "{triplets}"',
            transform_to_json=False
        )

    def calculate_cost(self) -> float:
        """Calculate the total cost of API usage."""
        return (self.prompt_tokens_num * self.input_price + 
                self.completion_tokens_num * self.output_price) / 1e6
    
    def calculate_used_tokens(self) -> int:
        """Calculate the total # of used tokens for generation"""
        return self.prompt_tokens_num, self.completion_tokens_num