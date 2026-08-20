"""Triplet-parallel variant of StructuredInferenceWithDB, for a single document.

The sequential pipeline (structured_inference_with_db.py) refines every
triplet extracted from a text one at a time -- type, then relation, then
subject name, then object name -- even though most of that work is
independent across triplets. This module restructures a single document's
processing into staged phases:

    Stage 1: Triplet extraction (one call, unchanged).
    Stage 2 (parallel across every triplet): Entity type refinement.
        Read-only against the ontology DB (static for the run), so safe
        regardless of triplet.
    Stage 3 (parallel across every triplet): Relation refinement. Same
        read-only guarantee as stage 2, using stage 2's output as input.
    Stage 4: Entity name refinement (subject + object), restructured as:
        4a. Dedup by (entity_name, entity_type) -- the same surface form
            with the same refined type is only resolved once, and the
            result is mapped back onto every triplet slot that shares it.
        4b. Fetch each unique entity's type hierarchy (parallel, read-only).
        4c. Partition the unique entities into groups such that no entity
            in one group can ever be presented as a merge-candidate for an
            entity in another group (see "Why partition by hierarchy"
            below), then resolve each group sequentially, but run
            different groups concurrently.
    Stage 5: Backbone validation + assembly (cheap, sequential).
    Stage 6: One batched write per triplet collection (initial / final /
        filtered / ontology_filtered), not interleaved with the refinement
        stages.

Why stage 4 needs partitioning at all
--------------------------------------
_refine_entity_name reads-then-writes a shared, per-sample_id canonical
store (structured_aligner.Aligner.add_entity): it retrieves existing
similar entities, has the LLM decide merge-or-new, then writes. If two
different entities' resolutions ran fully concurrently and could ever be
candidates for each other, they could each read the store *before* the
other's write landed and both independently conclude "no match, this is
new" -- silently minting two canonical entities that should have been one,
with no later step to catch it.

Why partition by hierarchy is *safe*
-------------------------------------
retrieve_entity_by_type only matches an existing entity whose stored type is
in `{query_type} + ancestors(query_type)` (see structured_aligner.py). So
entity A (type T_A) could only ever retrieve entity B (type T_B) as a
candidate -- in either processing order -- if T_A is in closure(T_B) or
T_B is in closure(T_A), i.e. one type is an ancestor of the other (or they
are the same type). 

This grouping is *conservative* via connected components: if A conflicts
with B and B conflicts with C, all three land in one sequential group even
if A and C don't directly conflict. 

Each worker thread gets and reuses its own LLMTripletExtractor instance
(built via `extractor_factory`), because LLMTripletExtractor is stateful --
token counters and retry/error state live on `self` -- and is not safe to
share across concurrently-running calls.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple
from unidecode import unidecode

from .base_inference_with_db import BaseInferenceWithDB
from .structured_inference_with_db import StructuredInferenceWithDB
from wikontic.db.factory import ensure_storage_backend
from wikontic.logging_config import get_logger

logger = get_logger("StructuredInferenceWithDBTripletParallelized")

EntityKey = Tuple[str, str]  # (entity_name, entity_type)
Slot = str  # "subject" or "object"

# Wikidata "point in time" / "quantity" -- see structured_inference_with_db.py's
# _refine_entity_name: an object-typed entity whose hierarchy includes either
# of these is written verbatim, with no retrieve_entity_by_type call at all.
_TIME_OR_QUANTITY_TYPES = {"Q186408", "Q309314"}


class _UnionFind:
    """Minimal disjoint-set structure for grouping conflicting entity keys
    into connected components."""

    def __init__(self, items):
        self._parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b

    def components(self) -> List[list]:
        groups: Dict[Any, list] = {}
        for item in self._parent:
            groups.setdefault(self.find(item), []).append(item)
        return list(groups.values())


class ParallelStructuredInferenceWithDB(BaseInferenceWithDB):
    def __init__(
        self,
        extractor_factory: Callable[[], Any],
        aligner,
        triplets_db,
        language: str = "en",
        max_workers: int = 8,
    ):
        self.extractor_factory = extractor_factory
        self.aligner = aligner
        self.triplets_db = ensure_storage_backend(triplets_db)
        self._init_language(language)
        self.max_workers = max_workers
        self._thread_local = threading.local()

    # ------------------------------------------------------------------ #
    # Per-thread state
    # ------------------------------------------------------------------ #

    def _thread_extractor(self):
        """This thread's own LLMTripletExtractor, created once and reused
        across every task the thread picks up (not once per task -- loading
        prompt files from disk on every call would be wasteful)."""
        extractor = getattr(self._thread_local, "extractor", None)
        if extractor is None:
            extractor = self.extractor_factory()
            self._thread_local.extractor = extractor
        return extractor

    def _thread_shell(self) -> StructuredInferenceWithDB:
        """A StructuredInferenceWithDB instance built without running its
        __init__ (which loads prompt files and wraps langchain tools -- both
        unnecessary per-thread cost), so this thread can reuse its already
        -correct _refine_entity_types / _refine_relation / _refine_entity_name
        / _validate_backbone methods with *this* thread's own extractor
        instead of a shared one."""
        shell = getattr(self._thread_local, "shell", None)
        if shell is None:
            shell = object.__new__(StructuredInferenceWithDB)
            shell.aligner = self.aligner
            shell.triplets_db = self.triplets_db
            shell.language = self.language
            shell.use_unidecode = self.use_unidecode
            self._thread_local.shell = shell
        shell.extractor = self._thread_extractor()
        return shell

    # ------------------------------------------------------------------ #
    # Stage 2 / 3: type + relation refinement (parallel across triplets)
    # ------------------------------------------------------------------ #

    def _refine_type_for_triplet(self, unit, embedding_cache):
        triplet_idx, text, triplet = unit
        shell = self._thread_shell()
        shell.extractor.reset_tokens()
        result = shell._refine_entity_types(
            text=text, triplet=triplet, embedding_cache=embedding_cache
        )
        prompt_tokens, completion_tokens = shell.extractor.calculate_used_tokens()
        return triplet_idx, result, prompt_tokens, completion_tokens

    def _refine_relation_for_triplet(self, unit, embedding_cache):
        (
            triplet_idx,
            text,
            triplet,
            refined_subject_type_id,
            refined_object_type_id,
        ) = unit
        shell = self._thread_shell()
        shell.extractor.reset_tokens()
        result = shell._refine_relation(
            text=text,
            triplet=triplet,
            refined_subject_type_id=refined_subject_type_id,
            refined_object_type_id=refined_object_type_id,
            embedding_cache=embedding_cache,
        )
        prompt_tokens, completion_tokens = shell.extractor.calculate_used_tokens()
        return triplet_idx, result, prompt_tokens, completion_tokens

    # ------------------------------------------------------------------ #
    # Stage 4: entity name refinement -- dedup, hierarchy partitioning,
    # grouped resolution
    # ------------------------------------------------------------------ #

    def _fetch_hierarchy_for_key(self, key: EntityKey):
        # Read-only against the (static, for this run) ontology DB. No LLM
        # call involved, so this doesn't need a thread-local extractor --
        # just the shared, thread-safe-for-reads aligner.
        _, entity_type = key
        hierarchy = self.aligner.retrieve_entity_type_hierarchy(entity_type)
        return key, hierarchy

    def _partition_entity_keys(
        self, hierarchy_by_key: Dict[EntityKey, List[str]]
    ) -> List[List[EntityKey]]:
        """Group unique entity keys so that keys in different groups can
        never be merge-candidates for each other (see module docstring)."""
        keys = list(hierarchy_by_key.keys())
        own_id = {key: hierarchy_by_key[key][0] for key in keys if hierarchy_by_key[key]}
        closure_set = {key: set(hierarchy_by_key[key]) for key in keys}

        uf = _UnionFind(keys)
        for i, key_i in enumerate(keys):
            for key_j in keys[i + 1:]:
                id_i, id_j = own_id.get(key_i), own_id.get(key_j)
                conflict = (
                    (id_i is not None and id_i in closure_set[key_j])
                    or (id_j is not None and id_j in closure_set[key_i])
                )
                if conflict:
                    uf.union(key_i, key_j)
        return uf.components()

    def _resolve_entity_group(
        self,
        group: List[EntityKey],
        instances_by_key: Dict[EntityKey, List[Tuple[int, Slot]]],
        backbone_triplets: List[dict],
        text: str,
        sample_id,
        use_unidecode: bool,
        embedding_cache,
    ) -> Dict[EntityKey, Tuple[str, int, int, int]]:
        """Sequentially resolve every unique key in one conflict-free group,
        on one thread, so each resolution's DB write is visible to the next
        one in the group -- exactly like the original sequential loop, just
        scoped to one group instead of every entity in the document.

        Returns, per key: (updated_entity, prompt_tokens, completion_tokens,
        representative_triplet_idx). The token counts belong to whichever
        one triplet's instance actually triggered the (single) LLM/DB call
        for this key -- every other triplet sharing the key gets the
        resolved name for free, without being credited or debited for a
        call that didn't happen on its behalf, so summing token counts
        across final triplets still reflects real spend.
        """
        shell = self._thread_shell()
        resolved: Dict[EntityKey, Tuple[str, int, int, int]] = {}

        for key in group:
            instances = instances_by_key[key]
            has_subject_instance = any(slot == "subject" for _, slot in instances)
            # Objects get a "is this a date/quantity, skip lookup" fast path
            # that subjects never get (see module docstring) -- a group
            # touched by any subject instance must go through full
            # treatment, never the object-only shortcut.
            is_object = not has_subject_instance
            required_slot = "object" if is_object else "subject"
            representative_triplet_idx, _ = next(
                inst for inst in instances if inst[1] == required_slot
            )
            representative_triplet = backbone_triplets[representative_triplet_idx]

            shell.extractor.reset_tokens()
            updated_entity = shell._refine_entity_name(
                text, representative_triplet, sample_id, is_object=is_object,
                use_unidecode=use_unidecode, embedding_cache=embedding_cache,
            )
            prompt_tokens, completion_tokens = shell.extractor.calculate_used_tokens()
            resolved[key] = (updated_entity, prompt_tokens, completion_tokens, representative_triplet_idx)

        return resolved

    def _refine_entities(
        self,
        backbone_triplets: List[dict],
        refine_flags: List[Tuple[bool, bool]],  # (needs_subject, needs_object) per triplet
        text: str,
        sample_id,
        use_unidecode: bool,
        embedding_cache,
    ) -> Dict[int, Tuple[int, int]]:
        """Refines every triplet's subject/object in place on
        backbone_triplets, deduped and hierarchy-partitioned. Returns the
        summed (prompt_tokens, completion_tokens) contributed by entity-name
        refinement, per triplet index."""
        instances_by_key: Dict[EntityKey, List[Tuple[int, Slot]]] = {}
        for triplet_idx, (needs_subject, needs_object) in enumerate(refine_flags):
            triplet = backbone_triplets[triplet_idx]
            if needs_subject:
                key = (triplet["subject"], triplet["subject_type"])
                instances_by_key.setdefault(key, []).append((triplet_idx, "subject"))
            if needs_object:
                key = (triplet["object"], triplet["object_type"])
                instances_by_key.setdefault(key, []).append((triplet_idx, "object"))

        token_totals: Dict[int, Tuple[int, int]] = {
            idx: (0, 0) for idx in range(len(backbone_triplets))
        }
        if not instances_by_key:
            return token_totals

        unique_keys = list(instances_by_key.keys())

        # Stage 4b: hierarchy lookup, parallel across unique entities.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            hierarchy_results = list(
                executor.map(self._fetch_hierarchy_for_key, unique_keys)
            )
        hierarchy_by_key = dict(hierarchy_results)

        # Object-only keys whose type is a time/quantity descendant never
        # call retrieve_entity_by_type at all (see module-level docstring
        # on _TIME_OR_QUANTITY_TYPES) -- they always write their own literal
        # name unconditionally, an idempotent blind upsert with no
        # read-then-decide step, so they can never race with anything and
        # don't need to go through conflict partitioning at all.
        def _is_trivial(key: EntityKey) -> bool:
            is_object_only = not any(
                slot == "subject" for _, slot in instances_by_key[key]
            )
            return is_object_only and any(
                t in _TIME_OR_QUANTITY_TYPES for t in hierarchy_by_key[key]
            )

        trivial_keys = [key for key in unique_keys if _is_trivial(key)]
        risky_keys = [key for key in unique_keys if key not in set(trivial_keys)]

        # Stage 4c: partition only the risky keys into conflict-free groups.
        # Each trivial key is its own always-safe, always-parallel singleton
        # "group" -- _resolve_entity_group handles a group of any size.
        groups = self._partition_entity_keys(
            {key: hierarchy_by_key[key] for key in risky_keys}
        )
        groups.extend([[key] for key in trivial_keys])
        logger.debug(
            "Entity resolution: %d unique entities (%d trivial) partitioned into %d group(s)",
            len(unique_keys), len(trivial_keys), len(groups),
        )

        # Stage 4d: resolve each group sequentially, groups in parallel.
        resolved_by_key: Dict[EntityKey, Tuple[str, int, int, int]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._resolve_entity_group,
                    group, instances_by_key, backbone_triplets, text, sample_id,
                    use_unidecode, embedding_cache,
                )
                for group in groups
            ]
            for future in futures:
                resolved_by_key.update(future.result())

        # Stage 4a (finish): map resolved names back onto every triplet slot
        # that shares a key. Token cost is credited only to the one
        # triplet whose instance actually triggered the (single) call for
        # that key -- see _resolve_entity_group's docstring for why.
        for key, instances in instances_by_key.items():
            updated_entity, prompt_tokens, completion_tokens, representative_idx = resolved_by_key[key]
            for triplet_idx, slot in instances:
                backbone_triplets[triplet_idx][slot] = updated_entity
            token_totals[representative_idx] = (
                token_totals[representative_idx][0] + prompt_tokens,
                token_totals[representative_idx][1] + completion_tokens,
            )

        return token_totals

    # ------------------------------------------------------------------ #
    # Full pipeline for one document
    # ------------------------------------------------------------------ #

    def extract_triplets_with_ontology_filtering(
        self, text, sample_id=None, source_text_id=None, use_unidecode=None
    ):
        if use_unidecode is None:
            use_unidecode = self.use_unidecode

        # ---- Stage 1: extraction (single call). ----
        extractor = self._thread_extractor()
        extractor.reset_tokens()
        extractor.reset_messages()
        extractor.reset_error_state()

        extracted_triplets = extractor.extract_triplets_from_text(text)
        triplets = self._parse_extracted_triplets(extracted_triplets, text)
        if triplets is None:
            return [], [], [], []

        initial_triplets: List[dict] = []
        extraction_prompt_tokens, extraction_completion_tokens = extractor.calculate_used_tokens()
        for triplet in triplets:
            triplet["prompt_token_num"] = extraction_prompt_tokens
            triplet["completion_token_num"] = extraction_completion_tokens
            triplet["source_text_id"] = source_text_id
            triplet["sample_id"] = sample_id
            initial_triplets.append(triplet.copy())

        # ---- Shared embedding cache for every unique subject/object
        # (post-unidecode)/relation/entity-type across the document. ----
        texts_to_embed = set()
        for triplet in triplets:
            texts_to_embed.add(unidecode(triplet["subject"]) if use_unidecode else triplet["subject"])
            texts_to_embed.add(unidecode(triplet["object"]) if use_unidecode else triplet["object"])
            texts_to_embed.add(triplet["relation"])
            texts_to_embed.add(triplet["subject_type"])
            texts_to_embed.add(triplet["object_type"])
        texts_to_embed = list(texts_to_embed)
        embedding_cache = (
            dict(zip(texts_to_embed, self.aligner.get_embeddings(texts_to_embed)))
            if texts_to_embed
            else {}
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # ---- Stage 2: type refinement, parallel across every triplet. ----
            type_units = [(idx, text, triplet) for idx, triplet in enumerate(triplets)]
            type_results = list(
                executor.map(
                    lambda unit: self._refine_type_for_triplet(unit, embedding_cache),
                    type_units,
                )
            )
            type_by_idx = {
                idx: (result, prompt_tokens, completion_tokens)
                for idx, result, prompt_tokens, completion_tokens in type_results
            }

            # ---- Stage 3: relation refinement, parallel across every
            # triplet, using stage 2's per-triplet output as input. ----
            relation_units = []
            for idx, _, triplet in type_units:
                (_, refined_subject_type_id, _, refined_object_type_id), _, _ = type_by_idx[idx]
                relation_units.append(
                    (idx, text, triplet, refined_subject_type_id, refined_object_type_id)
                )
            relation_results = list(
                executor.map(
                    lambda unit: self._refine_relation_for_triplet(unit, embedding_cache),
                    relation_units,
                )
            )
            relation_by_idx = {
                idx: (result, prompt_tokens, completion_tokens)
                for idx, result, prompt_tokens, completion_tokens in relation_results
            }

        # ---- Build each triplet's backbone (direction-adjusted subject
        # /object/types), and figure out which slots still need entity-name
        # refinement. Cheap, sequential bookkeeping -- no I/O. ----
        backbone_triplets: List[dict] = []
        refine_flags: List[Tuple[bool, bool]] = []
        relation_meta: List[tuple] = []  # (refined_relation_id, prop_subj_ids, prop_obj_ids)
        type_tokens: List[Tuple[int, int]] = []
        relation_tokens: List[Tuple[int, int]] = []

        for idx, triplet in enumerate(triplets):
            (
                refined_subject_type,
                refined_subject_type_id,
                refined_object_type,
                refined_object_type_id,
            ), type_p, type_c = type_by_idx[idx]
            (
                refined_relation,
                refined_relation_id,
                refined_relation_direction,
                prop_subject_type_ids,
                prop_object_type_ids,
            ), rel_p, rel_c = relation_by_idx[idx]

            if refined_relation_direction == "inverse":
                refined_subject_type_id, refined_object_type_id = (
                    refined_object_type_id,
                    refined_subject_type_id,
                )

            backbone_triplet = {
                "subject": (
                    triplet["subject"] if refined_relation_direction == "direct" else triplet["object"]
                ),
                "relation": refined_relation,
                "object": (
                    triplet["object"] if refined_relation_direction == "direct" else triplet["subject"]
                ),
                "subject_type": (
                    refined_subject_type if refined_relation_direction == "direct" else refined_object_type
                ),
                "object_type": (
                    refined_object_type if refined_relation_direction == "direct" else refined_subject_type
                ),
                "qualifiers": triplet["qualifiers"],
            }
            backbone_triplets.append(backbone_triplet)
            refine_flags.append((bool(refined_subject_type_id), bool(refined_object_type_id)))
            relation_meta.append((refined_relation_id, prop_subject_type_ids, prop_object_type_ids))
            type_tokens.append((type_p, type_c))
            relation_tokens.append((rel_p, rel_c))
            # Stash resolved type ids on the backbone dict for validation later.
            backbone_triplet["_refined_subject_type_id"] = refined_subject_type_id
            backbone_triplet["_refined_object_type_id"] = refined_object_type_id

        # ---- Stage 4: entity name refinement -- deduped, hierarchy
        # -partitioned, parallel across conflict-free groups. ----
        name_tokens = self._refine_entities(
            backbone_triplets, refine_flags, text, sample_id, use_unidecode, embedding_cache,
        )

        # ---- Stage 5: backbone validation + assembly. Sequential, but pure
        # bookkeeping + a handful of read-only ontology lookups -- cheap. ----
        shell = self._thread_shell()
        final_triplets: List[dict] = []
        filtered_triplets: List[dict] = []
        ontology_filtered_triplets: List[dict] = []

        for idx, triplet in enumerate(triplets):
            try:
                backbone_triplet = backbone_triplets[idx]
                refined_subject_type_id = backbone_triplet.pop("_refined_subject_type_id")
                refined_object_type_id = backbone_triplet.pop("_refined_object_type_id")
                refined_relation_id, prop_subject_type_ids, prop_object_type_ids = relation_meta[idx]

                type_p, type_c = type_tokens[idx]
                rel_p, rel_c = relation_tokens[idx]
                name_p, name_c = name_tokens[idx]
                backbone_triplet["prompt_token_num"] = type_p + rel_p + name_p
                backbone_triplet["completion_token_num"] = type_c + rel_c + name_c
                backbone_triplet["source_text_id"] = source_text_id
                backbone_triplet["sample_id"] = sample_id

                backbone_triplet_valid, backbone_triplet_exception_msg = shell._validate_backbone(
                    backbone_triplet["subject_type"],
                    backbone_triplet["object_type"],
                    backbone_triplet["relation"],
                    refined_object_type_id,
                    refined_subject_type_id,
                    refined_relation_id,
                    prop_subject_type_ids,
                    prop_object_type_ids,
                )

                if backbone_triplet_valid:
                    final_triplets.append(backbone_triplet.copy())
                else:
                    logger.error("Final triplet is ontology filtered: %s", backbone_triplet)
                    logger.error("Exception: %s", backbone_triplet_exception_msg)
                    backbone_triplet["exception_text"] = backbone_triplet_exception_msg
                    ontology_filtered_triplets.append(backbone_triplet.copy())

            except Exception as e:
                backbone_triplet = triplet.copy()
                backbone_triplet["prompt_token_num"] = 0
                backbone_triplet["completion_token_num"] = 0
                backbone_triplet["source_text_id"] = source_text_id
                backbone_triplet["sample_id"] = sample_id
                backbone_triplet["exception_text"] = str(e)
                filtered_triplets.append(backbone_triplet.copy())
                logger.info("Filtered triplet: %s", backbone_triplet)
                logger.info("Exception: %s", str(e))

        return initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets

    def extract_triplets_with_ontology_filtering_and_add_to_db(
        self, text, sample_id=None, source_text_id=None, use_unidecode=None
    ):
        """Drop-in parallel replacement for
        StructuredInferenceWithDB.extract_triplets_with_ontology_filtering_and_add_to_db.
        Same signature, same return contract -- refinement work for this
        document's triplets just runs in parallel internally.
        """
        initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets = (
            self.extract_triplets_with_ontology_filtering(
                text, sample_id=sample_id, source_text_id=source_text_id, use_unidecode=use_unidecode
            )
        )

        # ---- Stage 6: one batched write per collection. ----
        if len(initial_triplets) > 0:
            self.aligner.add_initial_triplets(initial_triplets, sample_id=sample_id)
        if len(final_triplets) > 0:
            self.aligner.add_triplets(final_triplets, sample_id=sample_id)
        if len(filtered_triplets) > 0:
            self.aligner.add_filtered_triplets(filtered_triplets, sample_id=sample_id)
        if len(ontology_filtered_triplets) > 0:
            self.aligner.add_ontology_filtered_triplets(ontology_filtered_triplets, sample_id=sample_id)

        return initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets
