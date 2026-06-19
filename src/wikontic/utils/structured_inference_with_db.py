from unidecode import unidecode
import re
import warnings
from typing import Dict, List, Tuple
from langchain.tools import tool
import logging

from .base_inference_with_db import BaseInferenceWithDB
from wikontic.db.factory import ensure_storage_backend
from wikontic.logging_config import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("StructuredInferenceWithDB")


class StructuredInferenceWithDB(BaseInferenceWithDB):
    def __init__(self, extractor, aligner, triplets_db):
        self.extractor = extractor
        self.aligner = aligner
        self.triplets_db = ensure_storage_backend(triplets_db)

        self.extract_triplets_with_ontology_filtering_tool = tool(
            self.extract_triplets_with_ontology_filtering
        )
        self.extract_triplets_with_ontology_filtering_and_add_to_db_tool = tool(
            self.extract_triplets_with_ontology_filtering_and_add_to_db
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
        # 1st step extraction without database

    def _refine_entity_types(self, text, triplet):
        """
        Refine entity types using LLM.
        """
        candidate_subj_type_ids, candidate_obj_type_ids = (
            self.aligner.retrieve_similar_entity_types(triplet=triplet)
        )

        candidate_entity_type_id_2_label = self.aligner.retrieve_entity_type_labels(
            candidate_subj_type_ids + candidate_obj_type_ids
        )

        candidate_entity_type_label_2_id = {
            entity_label: entity_id
            for entity_id, entity_label in candidate_entity_type_id_2_label.items()
        }

        candidate_subject_types = [
            candidate_entity_type_id_2_label[t] for t in candidate_subj_type_ids
        ]
        candidate_object_types = [
            candidate_entity_type_id_2_label[t] for t in candidate_obj_type_ids
        ]

        # no need to refine if the triplet's types are in the candidate types
        if (
            triplet["subject_type"] in candidate_subject_types
            and triplet["object_type"] in candidate_object_types
        ):
            refined_subject_type, refined_object_type = (
                triplet["subject_type"],
                triplet["object_type"],
            )
            refined_subject_type_id = candidate_entity_type_label_2_id[
                triplet["subject_type"]
            ]
            refined_object_type_id = candidate_entity_type_label_2_id[
                triplet["object_type"]
            ]

        else:
            # if the triplet's subject type is in the candidate types,
            # then only refine the subject type
            if triplet["subject_type"] in candidate_subject_types:
                candidate_subject_types = [triplet["subject_type"]]
            # if the triplet's object type is in the candidate types,
            # then only refine the object type
            if triplet["object_type"] in candidate_object_types:
                candidate_object_types = [triplet["object_type"]]

            self.extractor.reset_error_state()
            refined_entity_types = self.extractor.refine_entity_types(
                text=text,
                triplet=triplet,
                candidate_subject_types=candidate_subject_types,
                candidate_object_types=candidate_object_types,
            )
            refined_subject_type, refined_object_type = (
                refined_entity_types["subject_type"],
                refined_entity_types["object_type"],
            )

            refined_subject_type_id = (
                candidate_entity_type_label_2_id[refined_subject_type]
                if refined_subject_type in candidate_subject_types
                else None
            )

            refined_object_type_id = (
                candidate_entity_type_label_2_id[refined_object_type]
                if refined_object_type in candidate_object_types
                else None
            )

        return (
            refined_subject_type,
            refined_subject_type_id,
            refined_object_type,
            refined_object_type_id,
        )

    def _get_candidate_entity_properties(
        self, triplet: Dict[str, str], subj_type_ids: List[str], obj_type_ids: List[str]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, dict]]:
        """
        Retrieve candidate properties and their labels/constraints.
        """
        # Get the list of tuples (<property_id>, <property_direction>)
        properties: List[Tuple[str, str]] = (
            self.aligner.retrieve_properties_for_entity_type(
                target_relation=triplet["relation"],
                object_types=obj_type_ids,
                subject_types=subj_type_ids,
                k=10,
            )
        )
        # Get dict {<prop_id>:
        #           {"label": <prop_label>,
        #           "valid_subject_type_ids": <valid_subject_type_ids>,
        #           "valid_object_type_ids": <valid_object_type_ids>}}
        prop_2_label_and_constraint = (
            self.aligner.retrieve_properties_labels_and_constraints(
                property_id_list=[p[0] for p in properties]
            )
        )
        return properties, prop_2_label_and_constraint

    def _refine_relation(
        self, text, triplet, refined_subject_type_id, refined_object_type_id
    ):
        """
        Refine relation using LLM.
        """
        # if refined subject and object types are in the candidate types,
        # then refine the relation
        if refined_subject_type_id and refined_object_type_id:
            relation_direction_candidate_pairs, prop_2_label_and_constraint = (
                self._get_candidate_entity_properties(
                    triplet=triplet,
                    subj_type_ids=[refined_subject_type_id],
                    obj_type_ids=[refined_object_type_id],
                )
            )
            candidate_relations = [
                prop_2_label_and_constraint[p[0]]["label"]
                for p in relation_direction_candidate_pairs
            ]
            # no need to refine
            # if the triplet's relation is in the candidate relations
            if triplet["relation"] in candidate_relations:
                refined_relation = triplet["relation"]
            else:
                self.extractor.reset_error_state()
                refined_relation = self.extractor.refine_relation(
                    text=text, triplet=triplet, candidate_relations=candidate_relations
                )["relation"]
        # if refined subject and object types are not in the candidate types,
        # leave relation as it is
        else:
            refined_relation = triplet["relation"]
            candidate_relations = []

        # if refined relation is in the candidate relations,
        # then identify the relation direction
        if refined_relation in candidate_relations:
            refined_relation_id_candidates = [
                p_id
                for p_id in prop_2_label_and_constraint
                if prop_2_label_and_constraint[p_id]["label"] == refined_relation
            ]
            refined_relation_id = refined_relation_id_candidates[0]
            refined_relation_directions = [
                p[1]
                for p in relation_direction_candidate_pairs
                if p[0] == refined_relation_id
            ]
            refined_relation_direction = (
                "direct" if "direct" in refined_relation_directions else "inverse"
            )

            prop_subject_type_ids = [
                prop_2_label_and_constraint[prop]["valid_subject_type_ids"]
                for prop in prop_2_label_and_constraint
                if prop_2_label_and_constraint[prop]["label"] == refined_relation
            ][0]
            prop_object_type_ids = [
                prop_2_label_and_constraint[prop]["valid_object_type_ids"]
                for prop in prop_2_label_and_constraint
                if prop_2_label_and_constraint[prop]["label"] == refined_relation
            ][0]

        else:
            refined_relation_direction = "direct"
            refined_relation_id = None
            prop_subject_type_ids = []
            prop_object_type_ids = []

        return (
            refined_relation,
            refined_relation_id,
            refined_relation_direction,
            prop_subject_type_ids,
            prop_object_type_ids,
        )

    def _validate_backbone(
        self,
        refined_subject_type: str,
        refined_object_type: str,
        refined_relation: str,
        refined_object_type_id: str,
        refined_subject_type_id: str,
        refined_relation_id: str,
        valid_subject_type_ids: List[str],
        valid_object_type_ids: List[str],
    ):
        """
        Check if the selected backbone_triplet's types and relation are in the valid sets.
        """

        exception_msg = ""
        if not refined_relation_id:
            exception_msg += "Refined relation not in candidate relations\n"
        if not refined_subject_type_id:
            exception_msg += "Refined subject type not in candidate subject types\n"
        if not refined_object_type_id:
            exception_msg += "Refined object type not in candidate object types\n"

        if exception_msg != "":
            return False, exception_msg

        else:

            subject_type_hierarchy = self.aligner.retrieve_entity_type_hierarchy(
                refined_subject_type
            )
            object_type_hierarchy = self.aligner.retrieve_entity_type_hierarchy(
                refined_object_type
            )

            if valid_subject_type_ids == ["ANY"]:
                valid_subject_type_ids = subject_type_hierarchy
            if valid_object_type_ids == ["ANY"]:
                valid_object_type_ids = object_type_hierarchy

            if any(
                [t in subject_type_hierarchy for t in valid_subject_type_ids]
            ) and any([t in object_type_hierarchy for t in valid_object_type_ids]):
                return True, exception_msg
            else:
                exception_msg += "Triplet backbone violates property constraints\n"
                return False, exception_msg

    def _refine_entity_name(self, text, triplet, sample_id, is_object=False, use_unidecode=True):
        """
        Refine entity names using type constraints.
        """
        self.extractor.reset_error_state()
        if is_object:
            entity = unidecode(triplet["object"]) if use_unidecode else triplet["object"]
            entity_type = triplet["object_type"]
            entity_hierarchy = self.aligner.retrieve_entity_type_hierarchy(entity_type)
        else:
            entity = unidecode(triplet["subject"]) if use_unidecode else triplet["subject"]
            entity_type = triplet["subject_type"]
            entity_hierarchy = []

        # do not change time or quantity entities (of objects!)
        if any([t in ["Q186408", "Q309314"] for t in entity_hierarchy]):
            updated_entity = entity
        else:
            # if not time or quantity entities -> retrieve similar entities by type and name similarity
            similar_entities = self.aligner.retrieve_entity_by_type(
                entity_name=entity, entity_type=entity_type, sample_id=sample_id
            )
            # if there are similar entities -> refine entity name
            if len(similar_entities) > 0:
                # if exact match found -> return the exact match
                if entity in similar_entities:
                    updated_entity = similar_entities[entity]
                else:
                    # if not exact match -> refine entity name
                    updated_entity = self.extractor.refine_entity(
                        text=text,
                        triplet=triplet,
                        candidates=list(similar_entities.values()),
                        is_object=is_object,
                    )
                    # unidecode the updated entity
                    updated_entity = unidecode(updated_entity) if use_unidecode else updated_entity
                    # if the updated entity is None (meaning that LLM didn't find any similar entities)
                    # -> return the original entity
                    if re.sub(r"[^\w\s]", "", updated_entity) == "None":
                        updated_entity = entity
            else:
                # if no similar entities -> return the original entity
                updated_entity = entity

        self.aligner.add_entity(
            entity_name=updated_entity,
            alias=entity,
            entity_type=entity_type,
            sample_id=sample_id,
        )

        return updated_entity

    def extract_triplets_with_ontology_filtering(
        self, text, sample_id=None, source_text_id=None, use_unidecode=True
    ):
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
                (initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets)
        """
        self.extractor.reset_tokens()
        self.extractor.reset_messages()
        self.extractor.reset_error_state()

        extracted_triplets = self.extractor.extract_triplets_from_text(text)

        initial_triplets = []
        logger.debug(f"Extracted triplets: {extracted_triplets}")
        for triplet in extracted_triplets["triplets"]:
            triplet["prompt_token_num"], triplet["completion_token_num"] = (
                self.extractor.calculate_used_tokens()
            )
            triplet["source_text_id"] = source_text_id
            triplet["sample_id"] = sample_id
            initial_triplets.append(triplet.copy())

        final_triplets = []
        filtered_triplets = []
        ontology_filtered_triplets = []

        for triplet in extracted_triplets["triplets"]:
            self.extractor.reset_tokens()
            try:
                logger.log(logging.DEBUG, "Triplet: %s\n%s" % (str(triplet), "-" * 100))

                # _____________ Refine entity types  __________

                (
                    refined_subject_type,
                    refined_subject_type_id,
                    refined_object_type,
                    refined_object_type_id,
                ) = self._refine_entity_types(text=text, triplet=triplet)

                # ________________ Refine relation ________________
                (
                    refined_relation,
                    refined_relation_id,
                    refined_relation_direction,
                    prop_subject_type_ids,
                    prop_object_type_ids,
                ) = self._refine_relation(
                    text=text,
                    triplet=triplet,
                    refined_subject_type_id=refined_subject_type_id,
                    refined_object_type_id=refined_object_type_id,
                )

                if refined_relation_direction == "inverse":
                    refined_subject_type_id, refined_object_type_id = (
                        refined_object_type_id,
                        refined_subject_type_id,
                    )

                # __________ Refine entity names ___________
                backbone_triplet = {
                    "subject": (
                        triplet["subject"]
                        if refined_relation_direction == "direct"
                        else triplet["object"]
                    ),
                    "relation": refined_relation,
                    "object": (
                        triplet["object"]
                        if refined_relation_direction == "direct"
                        else triplet["subject"]
                    ),
                    "subject_type": (
                        refined_subject_type
                        if refined_relation_direction == "direct"
                        else refined_object_type
                    ),
                    "object_type": (
                        refined_object_type
                        if refined_relation_direction == "direct"
                        else refined_subject_type
                    ),
                }

                backbone_triplet["qualifiers"] = triplet["qualifiers"]
                if refined_subject_type_id:
                    backbone_triplet["subject"] = self._refine_entity_name(
                        text, backbone_triplet, sample_id, is_object=False, use_unidecode=use_unidecode
                    )

                if refined_object_type_id:
                    backbone_triplet["object"] = self._refine_entity_name(
                        text, backbone_triplet, sample_id, is_object=True, use_unidecode=use_unidecode
                    )

                logger.log(
                    logging.DEBUG,
                    "Original subject name: %s\n%s"
                    % (str(backbone_triplet["subject"]), "-" * 100),
                )
                logger.log(
                    logging.DEBUG,
                    "Original object name: %s\n%s"
                    % (str(backbone_triplet["object"]), "-" * 100),
                )
                logger.log(
                    logging.DEBUG,
                    "Refined subject name: %s\n%s"
                    % (str(backbone_triplet["subject"]), "-" * 100),
                )
                logger.log(
                    logging.DEBUG,
                    "Refined object name: %s\n%s"
                    % (str(backbone_triplet["object"]), "-" * 100),
                )

                (
                    backbone_triplet["prompt_token_num"],
                    backbone_triplet["completion_token_num"],
                ) = self.extractor.calculate_used_tokens()
                backbone_triplet["source_text_id"] = source_text_id
                backbone_triplet["sample_id"] = sample_id

                # ___________________________ Validate backbone triplet ___________________________
                backbone_triplet_valid, backbone_triplet_exception_msg = (
                    self._validate_backbone(
                        backbone_triplet["subject_type"],
                        backbone_triplet["object_type"],
                        backbone_triplet["relation"],
                        refined_object_type_id,
                        refined_subject_type_id,
                        refined_relation_id,
                        prop_subject_type_ids,
                        prop_object_type_ids,
                    )
                )

                if backbone_triplet_valid:
                    final_triplets.append(backbone_triplet.copy())
                    logger.log(
                        logging.DEBUG,
                        "Final triplet: %s\n%s" % (str(backbone_triplet), "-" * 100),
                    )
                else:
                    logger.log(
                        logging.ERROR,
                        "Final triplet is ontology filtered: %s\n%s"
                        % (str(backbone_triplet), "-" * 100),
                    )
                    logger.log(
                        logging.ERROR,
                        "Exception: %s" % (str(backbone_triplet_exception_msg)),
                    )
                    logger.log(
                        logging.ERROR, "Refined relation: %s" % (str(refined_relation))
                    )
                    logger.log(
                        logging.ERROR,
                        "Refined subject type: %s" % (str(refined_subject_type)),
                    )
                    logger.log(
                        logging.ERROR,
                        "Refined object type: %s" % (str(refined_object_type)),
                    )

                    backbone_triplet["exception_text"] = backbone_triplet_exception_msg
                    ontology_filtered_triplets.append(backbone_triplet.copy())

            except Exception as e:
                backbone_triplet = triplet.copy()
                (
                    backbone_triplet["prompt_token_num"],
                    backbone_triplet["completion_token_num"],
                ) = self.extractor.calculate_used_tokens()
                backbone_triplet["source_text_id"] = source_text_id
                backbone_triplet["sample_id"] = sample_id
                backbone_triplet["exception_text"] = str(e)
                filtered_triplets.append(backbone_triplet.copy())
                logger.log(
                    logging.INFO,
                    "Filtered triplet: %s\n%s" % (str(backbone_triplet), "-" * 100),
                )
                logger.log(logging.INFO, "Exception: %s" % (str(e)))

        return (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        )

    def extract_triplets_with_ontology_filtering_and_add_to_db(
        self, text, sample_id=None, source_text_id=None, use_unidecode=True
    ):
        """
        Extract and refine knowledge graph triplets from text using LLM, then add them to the database.
        Args:
            text (str): Input text to extract triplets from
            sample_id (str): Sample ID - used to distinguish graphs resulted in different launches/by different users
            source_text_id (str): Optional, - used to distinguish text from different sources (e.g., different paragraphs of the same text)
        Returns:
            tuple: (initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets)
        """
        (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        ) = self.extract_triplets_with_ontology_filtering(
            text, sample_id=sample_id, source_text_id=source_text_id, use_unidecode=use_unidecode
        )
        if len(initial_triplets) > 0:
            self.aligner.add_initial_triplets(initial_triplets, sample_id=sample_id)
        if len(final_triplets) > 0:
            self.aligner.add_triplets(final_triplets, sample_id=sample_id)
        if len(filtered_triplets) > 0:
            self.aligner.add_filtered_triplets(filtered_triplets, sample_id=sample_id)
        if len(ontology_filtered_triplets) > 0:
            self.aligner.add_ontology_filtered_triplets(
                ontology_filtered_triplets, sample_id=sample_id
            )
        return (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        )
