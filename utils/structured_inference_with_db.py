from unidecode import unidecode
import re
import warnings
import tenacity

warnings.filterwarnings('ignore')


def identify_relevant_entities(question, extractor, aligner, sample_id='0'):
    entities = extractor.extract_entities_from_question(question)

    identified_entities = []
    chosen_entities = []
    print(entities)
    if isinstance(entities, dict):
        entities = [entities]

    for ent in entities:
        similar_entities = aligner.retrive_similar_entity_names(entity_name=ent, k=5, sample_id=sample_id)
        exact_entity_match = [e for e in similar_entities if e['entity']==ent]
        if len(exact_entity_match) > 0:
            chosen_entities.extend(exact_entity_match)
        else:
            identified_entities.extend(similar_entities)
            
    chosen_entities.extend(extractor.identify_relevant_entities(question=question, entity_list=identified_entities))

    return chosen_entities


def answer_question(question, identified_entities, extractor, aligner, db, sample_id='0'):
    
    print("Chosen relevant entities: ", identified_entities)
    entity_set = {(e['entity'], e['entity_type']) for e in identified_entities}

    entities4search = list(entity_set)
    or_conditions = []

    for val, typ in entities4search:
        or_conditions.append({
            '$and': [
                {'subject': val},
                {'subject_type': typ}
            ]
        })
        or_conditions.append({
            '$and': [
                {'object': val},
                {'object_type': typ}
            ]
        })

    pipeline = [
        {
            '$match': {
                'sample_id': sample_id,
                '$or': or_conditions
            }
        }
    ]
    entities4search = [ent[0] for ent in entity_set]

    for i in range(5):
        
        or_conditions = []

        for ent in entities4search:
            or_conditions.append({
                '$and': [
                    {'subject': ent},
                ]
            })
            or_conditions.append({
                '$and': [
                    {'object': ent},
                ]
            })

        pipeline = [
            {
                '$match': {
                    'sample_id': sample_id,
                    '$or': or_conditions
                }
            }
        ]

        results = list(db.get_collection('triplets').aggregate(pipeline))

        for doc in results:
            entities4search.append(doc['subject'])
            entities4search.append(doc['object'])

            for q in doc['qualifiers']:
                entities4search.append(q['object'])

        entities4search = list(set(entities4search))
                        
    print(results)
    supporting_triplets = []
    for item in results:
        supporting_triplets.append({"subject": item['subject'], 'relation': item['relation'], 'object': item['object'], 'qualifiers': item['qualifiers']})
    

    ans = extractor.answer_question(question=question, triplets=supporting_triplets)
    # print(question, ' | ', ans, " | ", id2sample[sample_id]['answer'])
    # sample_id2ans[sample_id] = ans
    return supporting_triplets, ans


@tenacity.retry(stop=tenacity.stop_after_attempt(3), reraise=True)
def extract_triplets(text, sample_id, extractor, aligner):
    """Extract and refine knowledge graph triplets from text using LLM.
    
    Args:
        text (str): Input text to extract triplets from
        
    Returns:
        tuple: (final_triplets, filtered_triplets) where:
            - final_triplets: List of validated and refined triplets
            - filtered_triplets: List of triplets that couldn't be validated
    """
    # Extract initial triplets using LLM
    extracted_triplets = extractor.extract_triplets_from_text(text)

    initial_triplets = extracted_triplets.copy()
    
    final_triplets = []
    filtered_triplets = []

    for triplet in extracted_triplets['triplets']:        
        try:
            # Get candidate entity types ids
            subj_type_ids, obj_type_ids = aligner.retrieve_similar_entity_types(triplet=triplet)
            
            # Get candidate properties/relations ids
            properties = aligner.retrieve_properties_for_entity_type(
                target_relation=triplet['relation'],
                object_types=obj_type_ids, 
                subject_types=subj_type_ids,
                k=5
            )
            prop_2_label_and_constraint = aligner.retrieve_properties_labels_and_constraints(property_id_list=[p[0] for p in properties])
            entity_type_id_2_label = aligner.retrieve_entity_type_labels(subj_type_ids + obj_type_ids)

            # Build candidate triplet backbones
            candidates = []

            for prop_id, prop_direction in properties:
                valid_subject_type_ids = prop_2_label_and_constraint[prop_id]['valid_subject_type_ids']
                valid_object_type_ids = prop_2_label_and_constraint[prop_id]['valid_object_type_ids']
                property_label = prop_2_label_and_constraint[prop_id]['label']

                if prop_direction == 'direct':
                    # Include hierarchy here too??
                    subject_types = set(subj_type_ids) & set(valid_subject_type_ids)
                    object_types = set(obj_type_ids) & set(valid_object_type_ids)

                    # Use original type sets if no constraints matched
                    # meaning that property can be connected with <ANY> entity type
                    subject_types = subj_type_ids if len(subject_types) == 0 else subject_types
                    object_types = obj_type_ids if len(object_types) == 0 else object_types
                else:
                    subject_types = set(obj_type_ids) & set(valid_subject_type_ids)
                    object_types = set(subj_type_ids) & set(valid_object_type_ids) 
                    
                    # Use original type sets if no constraints matched
                    # meaning that property can be connected with <ANY> entity type
                    subject_types = obj_type_ids if len(subject_types) == 0 else subject_types
                    object_types = subj_type_ids if len(object_types) == 0 else object_types
                    
                # subject_types = subj_type_ids if len(subject_types) == 0 else subject_types
                # object_types = obj_type_ids if len(object_types) == 0 else object_types

                candidates.append({
                    "subject": triplet['subject'] if prop_direction == 'direct' else triplet['object'],
                    "relation": property_label,
                    'object': triplet['object'] if prop_direction == 'direct' else triplet['subject'],
                    "subject_types": [entity_type_id_2_label[t]['label'] for t in subject_types],
                    "object_types": [entity_type_id_2_label[t]['label'] for t in object_types]
                })


            # Refine relation and entity types using LLM - choose among valid backbones for triplet
            backbone_triplet = extractor.refine_relation_and_entity_types(
                text=text, 
                triplet=triplet,
                candidate_triplets=candidates,
            )
            # print(backbone_triplet)
            backbone_triplet['qualifiers'] = triplet['qualifiers']

            # Refine entity names
            final_triplet = refine_entities(text, backbone_triplet, aligner, sample_id, extractor=extractor)
            
            final_triplets.append(final_triplet)

            # !! Add validation that triplet was formed from bd types

        except Exception as e:
            filtered_triplets.append(triplet)
        
        for triple in final_triplets:
            triple["source_text_id"] = 0
            triple["prompt_token_nums"] = 0
            triple["completion_token_num"] = 0
        for triple in filtered_triplets:
            triple["source_text_id"] = 0
            triple["prompt_token_nums"] = 0
            triple["completion_token_num"] = 0
        
        if len(final_triplets) > 0:
            aligner.add_triplets(final_triplets, sample_id=sample_id)
        if len(filtered_triplets) > 0:
            aligner.add_filtered_triplets(filtered_triplets, sample_id=sample_id)
        # print("2nd resulted triplet: ", final_triplet)
    return initial_triplets, final_triplets, filtered_triplets


def refine_entities(text, triplet, aligner, sample_id, extractor):
    """Refine entity names using type constraints."""

    triplet['subject'] = unidecode(triplet['subject'])
    triplet['object'] = unidecode(triplet['object'])

    # print(triplet['object_type'], triplet['subject_type'])
    ################################ Handle object refinement ################################
    # get all id in entity type's hierarchy
    obj_hierarchy = aligner.retrieve_entity_type_hirerarchy(triplet['object_type'])
    updated_obj = 'None'
    
    # do not change time or quantity entities
    if not any(t in ['Q186408', 'Q309314'] for t in obj_hierarchy):
        # dict alias: original_entity_label
        similar_objects = aligner.retrieve_entity_by_type(
            entity_name=triplet['object'],
            entity_type=triplet['object_type'],
            sample_id=sample_id
        )
        if len(similar_objects) > 0:
            if triplet['object'] in similar_objects:
                updated_obj = similar_objects[triplet['object']]
            else:
                updated_obj = extractor.refine_entity(
                    text=text,
                    triplet=triplet,
                    candidates=list(similar_objects.values()),
                    is_object=True
                )
                updated_obj = unidecode(updated_obj)
    
    if re.sub(r'[^\w\s]', '', updated_obj) != 'None':
        if triplet['object'] != updated_obj:
            aligner.add_entity(entity_name=updated_obj, alias=triplet['object'], entity_type=triplet['object_type'], sample_id=sample_id)
        triplet['object'] = updated_obj
    else:
        aligner.add_entity(entity_name=triplet['object'], alias=triplet['object'], entity_type=triplet['object_type'], sample_id=sample_id)

    ################################# Handle subject refinement ################################
    updated_subj = 'None'
    similar_subjects = aligner.retrieve_entity_by_type(
        entity_name=triplet['subject'],
        entity_type=triplet['subject_type'],
        sample_id=sample_id
    )
    if len(similar_subjects) > 0:
        if triplet['subject'] in similar_subjects:
            updated_subj = similar_subjects[triplet['subject']]
        else:
            updated_subj = extractor.refine_entity(
                text=text,
                triplet=triplet,
                candidates=list(similar_subjects.values()),
                is_object=False
            )
            updated_subj = unidecode(updated_subj)
    
    if re.sub(r'[^\w\s]', '', updated_subj) != 'None':
        if triplet['subject'] != updated_subj:
            aligner.add_entity(entity_name=updated_subj, alias=triplet['subject'], entity_type=triplet['subject_type'], sample_id=sample_id)
        triplet['subject'] = updated_subj
    else:
        aligner.add_entity(entity_name=triplet['subject'], alias=triplet['subject'], entity_type=triplet['subject_type'], sample_id=sample_id)
        
    return triplet
