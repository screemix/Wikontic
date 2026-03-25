from typing import Dict, List, Optional
import logging

logger = logging.getLogger("BaseInferenceWithDB")
logger.setLevel(logging.ERROR)


class BaseInferenceWithDB:
    """
    Base class for inference with database functionality.
    Contains common methods shared by InferenceWithDB and StructuredInferenceWithDB.

    Note: This is an abstract base class. Child classes must define the following
    attributes in their __init__ methods:
        - self.extractor: The extractor instance
        - self.aligner: The aligner instance
        - self.triplets_db: The triplets database instance
    """

    def retrieve_similar_entity_names(
        self, entity_name: str, k: int, sample_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieve similar entity names from the knowledge graph using vector search.
        Useful to link entities from the question to the knowledge graph.
        Args:
            entity_name: The entity name to retrieve similar entity names from.
            k: The number of similar entity names to retrieve.
            sample_id: The sample ID of the subgraph to retrieve similar entity names from. If None, perform the search across all samples.
        Returns:
            A list of dictionaries with the entity name and entity type.
        """

        similar_entity_names = self.aligner.retrieve_similar_entity_names(
            entity_name=entity_name, k=k, sample_id=sample_id
        )
        if isinstance(similar_entity_names, dict):
            similar_entity_names = [e["entity"] for e in similar_entity_names]
        return similar_entity_names

    def identify_relevant_entities_from_question_with_llm(
        self, question, sample_id=None, use_entity_types=True
    ):
        """
        Identify relevant entities from question using LLM.
        Args:
            question: The question to identify relevant entities from.
            sample_id: The sample ID of the subgraph to identify relevant entities from. If None, perform the search across all samples.
        Returns:
            The relevant entities.
        """

        entities = self.extractor.extract_entities_from_question(question)
        identified_entities = []
        linked_entities = []

        if isinstance(entities, dict):
            entities = [entities]

        for ent in entities:
            similar_entities = self.retrieve_similar_entity_names(
                entity_name=ent, k=10, sample_id=sample_id
            )
            logger.log(logging.DEBUG, "Similar entities: %s" % (str(similar_entities)))

            exact_entity_match = [e for e in similar_entities if e == ent]
            if len(exact_entity_match) > 0:
                linked_entities.extend(exact_entity_match)
            else:
                identified_entities.extend(similar_entities)

        logger.log(
            logging.DEBUG,
            "Identified entities from question: %s" % (str(identified_entities)),
        )
        logger.log(
            logging.DEBUG, "Linked entities from question: %s" % (str(linked_entities))
        )
        if use_entity_types:
            linked_identified_entities = self.extractor.identify_relevant_entities(
                question=question, entity_list=identified_entities
            )
        else:
            linked_identified_entities = (
                self.extractor.identify_relevant_entities_wo_types(
                    question=question, entity_list=identified_entities
                )
            )
        linked_entities.extend([e["entity"] for e in linked_identified_entities])

        logger.log(
            logging.DEBUG,
            "Linked entities after refinement: %s" % (str(linked_entities)),
        )
        return linked_entities

    def get_1_hop_supporting_triplets(
        self,
        entities4search: List[str],
        sample_id=None,
        use_qualifiers=False,
        use_filtered_triplets=False,
    ):
        """
        Get the 1-hop supporting triplets for the given entities.
        Useful to answer the question with the given entities.
        Can be invoked multiple times for more than 1-hop support.
        Args:
            entities4search: The entities to get the 1-hop supporting triplets for.
            sample_id: The sample ID of the subgraph to get the 1-hop supporting triplets from. If None, perform the search across all samples.
            use_qualifiers: Whether to use qualifiers.
            use_filtered_triplets: Whether to use the triplets that violate the ontology constraints along with the valid triplets.
        Returns:
            A list of dictionaries with the subject, relation, object, and qualifiers that correspond to the 1-hop supporting triplets for the given entities.
        """
        if len(entities4search) == 0:
            return []
        or_conditions = []
        for ent in entities4search:
            or_conditions.append({"$and": [{"subject": ent}]})
            or_conditions.append({"$and": [{"object": ent}]})
        if sample_id is None:
            pipeline = [{"$match": {"$or": or_conditions}}]
        else:
            pipeline = [{"$match": {"sample_id": sample_id, "$or": or_conditions}}]
        results = list(
            self.triplets_db.get_collection(
                self.aligner.triplets_collection_name
            ).aggregate(pipeline)
        )

        if use_filtered_triplets:
            filtered_results = list(
                self.triplets_db.get_collection(
                    self.aligner.ontology_filtered_triplets_collection_name
                ).aggregate(pipeline)
            )
            results.extend(filtered_results)

        if use_qualifiers:
            supporting_triplets = [
                {
                    "subject": item["subject"],
                    "relation": item["relation"],
                    "object": item["object"],
                    "qualifiers": item["qualifiers"],
                }
                for item in results
            ]
        else:
            supporting_triplets = [
                {
                    "subject": item["subject"],
                    "relation": item["relation"],
                    "object": item["object"],
                }
                for item in results
            ]
        logger.log(
            logging.DEBUG,
            "Supporting triplets: %s\n%s" % (str(supporting_triplets), "-" * 100),
        )
        return supporting_triplets

    def answer_question_with_llm(
        self,
        question,
        linked_entities,
        sample_id=None,
        hop_depth=5,
        use_filtered_triplets=False,
        use_qualifiers=False,
    ):
        """
        "Answer a question with relevant entities."
        Args:
            question: The question to answer.
            linked_entities: The linked entities to answer the question.
            sample_id: The sample ID of the subgraph to answer the question from. If None, perform the search across all samples.
            use_filtered_triplets: Whether to use filtered triplets.
            use_qualifiers: Whether to use qualifiers.
        Returns:
            The answer to the question.
        """
        logger.log(logging.DEBUG, "Linked entities: %s" % (str(linked_entities)))

        entity_set = {e for e in linked_entities}
        entities4search = list(entity_set)
        supporting_triplets = []

        for _ in range(hop_depth):
            new_entities4search = set()
            new_supporting_triplets = self.get_1_hop_supporting_triplets(
                entities4search, sample_id, use_qualifiers, use_filtered_triplets
            )
            for triplet in new_supporting_triplets:
                if triplet not in supporting_triplets:
                    supporting_triplets.append(triplet)

            for doc in supporting_triplets:
                if doc["subject"] not in entities4search:
                    new_entities4search.add(doc["subject"])
                if doc["object"] not in entities4search:
                    new_entities4search.add(doc["object"])
                if use_qualifiers:
                    for q in doc["qualifiers"]:
                        if q["object"] not in entities4search:
                            new_entities4search.add(q["object"])

            entities4search = list(set(new_entities4search))

        if use_qualifiers:
            supporting_triplets = [
                {
                    "subject": item["subject"],
                    "relation": item["relation"],
                    "object": item["object"],
                    "qualifiers": item["qualifiers"],
                }
                for item in supporting_triplets
            ]
        else:
            supporting_triplets = [
                {
                    "subject": item["subject"],
                    "relation": item["relation"],
                    "object": item["object"],
                }
                for item in supporting_triplets
            ]
        logger.log(
            logging.DEBUG,
            "Supporting triplets: %s\n%s" % (str(supporting_triplets), "-" * 100),
        )

        ans = self.extractor.answer_question(
            question=question, triplets=supporting_triplets
        )
        return supporting_triplets, ans

    def answer_with_qa_collapsing(
        self,
        question,
        sample_id=None,
        max_attempts=5,
        use_qualifiers=False,
        use_filtered_triplets=False,
    ):
        """
        "Answer a question with QA collapsing."
        Args:
            question: The question to answer.
            sample_id: The sample ID of the subgraph to answer the question from. If None, perform the search across all samples.
            max_attempts: The maximum number of attempts to answer the question. Useful to handle complex questions that require multiple hops to answer.
            use_qualifiers: Whether to use qualifiers.
            use_filtered_triplets: Whether to use filtered triplets.
        Returns:
            The answer to the question.
        """
        collapsed_question_answer = ""
        collapsed_question_sequence = []
        collapsed_answer_sequence = []

        logger.log(logging.DEBUG, "Question: %s" % (str(question)))
        collapsed_question = self.extractor.decompose_question(question)

        for i in range(max_attempts):
            extracted_entities = self.extractor.extract_entities_from_question(
                collapsed_question
            )
            logger.log(
                logging.DEBUG, "Collapsed question: %s" % (str(collapsed_question))
            )
            logger.log(
                logging.DEBUG, "Extracted entities: %s" % (str(extracted_entities))
            )

            if len(collapsed_question_answer) > 0:
                extracted_entities.append(collapsed_question_answer)

            entities4search = []
            for ent in extracted_entities:
                similar_entities = self.retrieve_similar_entity_names(
                    entity_name=ent, k=10, sample_id=sample_id
                )
                entities4search.extend([e for e in similar_entities])

            entities4search = list(set(entities4search))
            logger.log(logging.DEBUG, "Similar entities: %s" % (str(entities4search)))

            supporting_triplets = self.get_1_hop_supporting_triplets(
                entities4search, sample_id, use_qualifiers, use_filtered_triplets
            )

            logger.log(
                logging.DEBUG,
                "Supporting triplets length: %s" % (str(len(supporting_triplets))),
            )

            collapsed_question_answer = self.extractor.answer_question(
                collapsed_question, supporting_triplets
            )
            collapsed_question_sequence.append(collapsed_question)
            collapsed_answer_sequence.append(collapsed_question_answer)

            logger.log(
                logging.DEBUG, "Collapsed question: %s" % (str(collapsed_question))
            )
            logger.log(
                logging.DEBUG,
                "Collapsed question answer: %s" % (str(collapsed_question_answer)),
            )

            is_answered = self.extractor.check_if_question_is_answered(
                question, collapsed_question_sequence, collapsed_answer_sequence
            )
            question_answer_sequence = list(
                zip(collapsed_question_sequence, collapsed_answer_sequence)
            )

            if is_answered == "NOT FINAL":
                collapsed_question = self.extractor.collapse_question(
                    original_question=question,
                    question=collapsed_question,
                    answer=collapsed_question_answer,
                )
                continue
            else:
                logger.log(logging.DEBUG, "Final answer: %s" % (str(is_answered)))
                return is_answered

        logger.log(logging.DEBUG, "Final answer: %s" % (str(collapsed_question_answer)))
        return collapsed_question_answer
