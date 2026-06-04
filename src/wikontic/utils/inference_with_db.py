import re
import warnings
from langchain.tools import tool
import logging

from .base_inference_with_db import BaseInferenceWithDB
from wikontic.db.factory import ensure_storage_backend

warnings.filterwarnings("ignore")
logger = logging.getLogger("InferenceWithDB")
logger.setLevel(logging.ERROR)


class InferenceWithDB(BaseInferenceWithDB):
    def __init__(self, extractor, aligner, triplets_db):
        self.extractor = extractor
        self.aligner = aligner
        self.triplets_db = ensure_storage_backend(triplets_db)

        self.extract_triplets_tool = tool(self.extract_triplets)
        self.extract_triplets_and_add_to_db_tool = tool(
            self.extract_triplets_and_add_to_db
        )
        self.retrieve_similar_entity_names_tool = tool(
            self.retrieve_similar_entity_names
        )
        self.identify_relevant_entities_from_question_tool = tool(
            self.identify_relevant_entities_from_question_with_llm
        )
        self.get_1_hop_supporting_triplets_tool = tool(
            self.get_1_hop_supporting_triplets
        )
        self.answer_question_with_llm_tool = tool(self.answer_question_with_llm)

    def sanitize_string(self, s):
        s = str(s).strip().replace('\\"', "")
        if s.startswith(r"\u"):
            s = s.encode().decode("unicode_escape")
        return s.strip()

    def extract_triplets(self, text, sample_id, source_text_id=None):
        """
        Extract and refine knowledge graph triplets from text using LLM.

        Args:
            text (str): Input text to extract triplets from
            sample_id (str): Sample ID - used to distinguish graphs resulted
                in different launches/by different users
            source_text_id (str): Optional, - used to distinguish texts from
                different sources (for example, different paragraphs of the same text)
        Returns:
            tuple:
                (initial_triplets, final_triplets, filtered_triplets)
        """
        self.extractor.reset_tokens()
        self.extractor.reset_messages()
        self.extractor.reset_error_state()

        initial_triplets = []

        extracted_triplets = self.extractor.extract_triplets_from_text(text)
        for triplet in extracted_triplets["triplets"]:
            triplet["prompt_token_num"], triplet["completion_token_num"] = (
                self.extractor.calculate_used_tokens()
            )
            triplet["source_text_id"] = source_text_id
            triplet["sample_id"] = sample_id
            initial_triplets.append(triplet.copy())

        final_triplets = []
        filtered_triplets = []

        for triplet in extracted_triplets["triplets"]:
            self.extractor.reset_tokens()
            try:
                logger.log(logging.DEBUG, "Triplet: %s\n%s" % (str(triplet), "-" * 100))
                refined_subject = self.refine_entity_name(
                    text, triplet, sample_id, is_object=False
                )
                refined_object = self.refine_entity_name(
                    text, triplet, sample_id, is_object=True
                )

                triplet["subject"] = refined_subject
                triplet["object"] = refined_object

                refined_relation = self.refine_relation_name(text, triplet, sample_id)
                triplet["relation"] = refined_relation

                final_triplets.append(triplet)
                logger.log(
                    logging.DEBUG, "Final triplet: %s\n%s" % (str(triplet), "-" * 100)
                )
                logger.log(
                    logging.DEBUG,
                    "Refined subject: %s\n%s" % (str(refined_subject), "-" * 100),
                )
                logger.log(
                    logging.DEBUG,
                    "Refined object: %s\n%s" % (str(refined_object), "-" * 100),
                )
                logger.log(
                    logging.DEBUG,
                    "Refined relation: %s\n%s" % (str(refined_relation), "-" * 100),
                )

            except Exception as e:
                triplet["exception_text"] = str(e)
                triplet["prompt_token_num"], triplet["completion_token_num"] = (
                    self.extractor.calculate_used_tokens()
                )
                triplet["sample_id"] = sample_id
                filtered_triplets.append(triplet)
                logger.log(
                    logging.INFO, "Filtered triplet: %s\n%s" % (str(triplet), "-" * 100)
                )
                logger.log(logging.INFO, "Exception: %s" % (str(e)))

        return initial_triplets, final_triplets, filtered_triplets

    def extract_triplets_and_add_to_db(self, text, source_text_id, sample_id=None):
        """
        Extract and refine knowledge graph triplets from text using LLM, then add them to the database.
        Args:
            text (str): Input text to extract triplets from
            sample_id (str): Sample ID - used to distinguish graphs resulted in different launches/by different users
            source_text_id (str): Optional, - used to distinguish text from different sources (for example, different paragraphs of the same text)
        Returns:
            tuple: (initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets)
        """
        (
            initial_triplets,
            final_triplets,
            filtered_triplets,
        ) = self.extract_triplets(text, sample_id, source_text_id)
        if len(initial_triplets) > 0:
            self.aligner.add_initial_triplets(initial_triplets, sample_id=sample_id)
        if len(final_triplets) > 0:
            self.aligner.add_triplets(final_triplets, sample_id=sample_id)
        if len(filtered_triplets) > 0:
            self.aligner.add_filtered_triplets(filtered_triplets, sample_id=sample_id)
        return (
            initial_triplets,
            final_triplets,
            filtered_triplets,
        )

    def refine_entity_name(self, text, triplet, sample_id, is_object=False):
        """
        Refine entity names using type constraints.
        """
        self.extractor.reset_error_state()
        if is_object:
            entity = triplet["object"]
        else:
            entity = triplet["subject"]
            # entity = unidecode(entity)
        entity = self.sanitize_string(entity)

        similar_entities = self.aligner.retrieve_similar_entity_names(
            entity_name=entity, sample_id=sample_id
        )

        similar_entities = [self.sanitize_string(entity) for entity in similar_entities]

        # if there are similar entities -> refine entity name
        # if no similar entities -> return the original entity
        # if exact match found -> return the exact match
        if len(similar_entities) == 0 or entity in similar_entities:
            updated_entity = entity
        else:
            # if not exact match -> refine entity name
            updated_entity = self.extractor.refine_entity(
                text=text,
                triplet=triplet,
                candidates=similar_entities,
                is_object=is_object,
            )
            # unidecode the updated entity
            # updated_entity = unidecode(updated_entity)
            updated_entity = self.sanitize_string(updated_entity)
            # if the updated entity is None (meaning that LLM didn't find any similar entities)
            # -> return the original entity
            if re.sub(r"[^\w\s]", "", updated_entity) == "None":
                updated_entity = entity

        self.aligner.add_entity(
            entity_name=updated_entity, alias=entity, sample_id=sample_id
        )

        return updated_entity

    def refine_relation_name(self, text, triplet, sample_id):
        """
        Refine relation names using LLM.
        """
        self.extractor.reset_error_state()

        # relation = unidecode(triplet['relation'])
        relation = self.sanitize_string(triplet["relation"])

        similar_relations: List[str] = self.aligner.retrieve_similar_properties(
            target_relation=relation, sample_id=sample_id
        )

        similar_relations = [
            self.sanitize_string(relation) for relation in similar_relations
        ]
        if len(similar_relations) == 0 or relation in similar_relations:
            updated_relation = relation
        else:
            updated_relation = self.extractor.refine_relation_wo_entity_types(
                text=text, triplet=triplet, candidate_relations=similar_relations
            )

            # updated_relation = unidecode(updated_relation)
            updated_relation = self.sanitize_string(updated_relation)

            if re.sub(r"[^\w\s]", "", updated_relation) == "None":
                updated_relation = relation

        self.aligner.add_property(
            property_name=updated_relation, alias=relation, sample_id=sample_id
        )

        return updated_relation
