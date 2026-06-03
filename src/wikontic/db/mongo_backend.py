from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from pymongo import ASCENDING, UpdateOne
from pymongo.operations import SearchIndexModel

from .interfaces import VectorQuery

# Poll Atlas Search index lifecycle instead of a fixed sleep after create/drop.
SEARCH_INDEX_BUILD_TIMEOUT_SECONDS = 120
SEARCH_INDEX_DROP_TIMEOUT_SECONDS = 60
# Retry $vectorSearch when recent writes may not be indexed yet (Atlas lag).
SEARCH_INDEX_SYNC_MAX_ATTEMPTS = 8
SEARCH_INDEX_SYNC_DELAY_SECONDS = 1.5


def _find_search_index(collection: Any, index_name: str) -> Optional[Dict[str, Any]]:
    for existing in collection.list_search_indexes():
        if existing.get("name") == index_name:
            return existing
    return None


def _wait_search_index_ready(
    collection: Any, index_name: str, timeout: float = SEARCH_INDEX_BUILD_TIMEOUT_SECONDS
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        existing = _find_search_index(collection, index_name)
        if existing is None:
            time.sleep(0.5)
            continue
        status = existing.get("status")
        if status == "READY":
            return
        if status == "FAILED":
            raise RuntimeError(
                f"MongoDB search index {index_name!r} failed to build on "
                f"{collection.name!r}."
            )
        time.sleep(1)
    raise RuntimeError(
        f"MongoDB search index {index_name!r} on {collection.name!r} did not "
        f"become READY within {timeout:.0f}s."
    )


def _wait_search_index_dropped(
    collection: Any, index_name: str, timeout: float = SEARCH_INDEX_DROP_TIMEOUT_SECONDS
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _find_search_index(collection, index_name) is None:
            return
        time.sleep(1)
    raise RuntimeError(
        f"MongoDB search index {index_name!r} on {collection.name!r} was not "
        f"removed within {timeout:.0f}s."
    )


def _drop_search_index_if_exists(collection: Any, index_name: str) -> None:
    if _find_search_index(collection, index_name) is None:
        return
    collection.drop_search_index(index_name)
    # Avoid create while the old index is still in DELETING state.
    _wait_search_index_dropped(collection, index_name)


def _is_compatible_knn_index(
    existing: Dict[str, Any],
    vector_field: str,
    token_fields: Optional[List[str]],
    num_dimensions: int,
) -> bool:
    # This codebase uses Atlas Search knnVector indexes, not the vectorSearch type.
    if existing.get("type") == "vectorSearch":
        return False

    definition = existing.get("latestDefinition") or existing.get("definition") or {}
    fields = definition.get("mappings", {}).get("fields", {})
    if not fields:
        return False

    vector_mapping = fields.get(vector_field, {})
    if vector_mapping.get("type") != "knnVector":
        return False
    if vector_mapping.get("dimensions") not in (None, num_dimensions):
        return False

    for field_name in token_fields or []:
        if fields.get(field_name, {}).get("type") != "token":
            return False
    return True


def _build_knn_index_definition(
    vector_field: str,
    num_dimensions: int,
    similarity: str,
    token_fields: Optional[List[str]],
) -> Dict[str, Any]:
    mapping_fields: Dict[str, Any] = {
        vector_field: {
            "dimensions": num_dimensions,
            "similarity": similarity,
            "type": "knnVector",
        }
    }
    # Token fields enable $vectorSearch pre-filters ($eq / $in on sample_id, etc.).
    for field_name in token_fields or []:
        mapping_fields[field_name] = {"type": "token"}
    return {"mappings": {"dynamic": True, "fields": mapping_fields}}


@dataclass
class MongoCollectionAdapter:
    collection: Any
    # Wired so inserts notify MongoBackend about pending search-index sync.
    backend: Optional["MongoBackend"] = None
    collection_name: Optional[str] = None

    def _mark_changed(self) -> None:
        if self.backend is not None and self.collection_name is not None:
            self.backend._mark_collection_changed(self.collection_name)

    def find_one(
        self, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None
    ) -> Optional[Dict[str, Any]]:
        return self.collection.find_one(query, projection)

    def find(
        self, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        return list(self.collection.find(query, projection))

    def insert_one(self, document: Dict[str, Any]) -> None:
        self.collection.insert_one(document)
        self._mark_changed()

    def insert_many(self, documents: Iterable[Dict[str, Any]]) -> None:
        docs = list(documents)
        if docs:
            self.collection.insert_many(docs)
            self._mark_changed()

    def count_documents(self, query: Dict[str, Any]) -> int:
        return self.collection.count_documents(query)


@dataclass
class MongoBackend:
    db: Any
    # Write batch counter vs last successful vector search (per collection).
    _collection_generation: Dict[str, int] = field(default_factory=dict)
    _indexed_generation: Dict[str, int] = field(default_factory=dict)

    def _mark_collection_changed(self, collection_name: str) -> None:
        self._collection_generation[collection_name] = (
            self._collection_generation.get(collection_name, 0) + 1
        )

    def _needs_search_sync(self, collection_name: str) -> bool:
        """True if docs were written since vector search last returned hits."""
        current = self._collection_generation.get(collection_name, 0)
        indexed = self._indexed_generation.get(collection_name, 0)
        return current > indexed

    def _mark_collection_indexed(self, collection_name: str) -> None:
        self._indexed_generation[collection_name] = self._collection_generation.get(
            collection_name, 0
        )

    def ensure_vector_index(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        num_dimensions: int,
        similarity: str = "cosine",
        token_fields: Optional[List[str]] = None,
        recreate: bool = False,
    ) -> None:
        """Ensure a knnVector Atlas Search index exists and is READY."""
        collection = self.db.get_collection(collection_name)
        index_definition = _build_knn_index_definition(
            vector_field=vector_field,
            num_dimensions=num_dimensions,
            similarity=similarity,
            token_fields=token_fields,
        )

        # Used after drop_collections=True so stale index metadata cannot be reused.
        if recreate:
            _drop_search_index_if_exists(collection, index_name)
            collection.create_search_index(
                model=SearchIndexModel(definition=index_definition, name=index_name)
            )
            _wait_search_index_ready(collection, index_name)
            return

        existing = _find_search_index(collection, index_name)
        if existing is None:
            collection.create_search_index(
                model=SearchIndexModel(definition=index_definition, name=index_name)
            )
            _wait_search_index_ready(collection, index_name)
            return

        status = existing.get("status")
        if status == "BUILDING":
            # Another caller may have started creation; wait rather than recreate.
            _wait_search_index_ready(collection, index_name)
            return

        if status == "READY" and _is_compatible_knn_index(
            existing, vector_field, token_fields, num_dimensions
        ):
            return

        # Wrong type/definition or failed index: replace and wait until searchable.
        _drop_search_index_if_exists(collection, index_name)
        collection.create_search_index(
            model=SearchIndexModel(definition=index_definition, name=index_name)
        )
        _wait_search_index_ready(collection, index_name)

    def create_indexes(self, collection_name: str, indexes: List[List[str]]) -> None:
        collection = self.db.get_collection(collection_name)
        for index_fields in indexes:
            collection.create_index([(field, ASCENDING) for field in index_fields])

    def get_collection(self, collection_name: str) -> MongoCollectionAdapter:
        return MongoCollectionAdapter(
            collection=self.db.get_collection(collection_name),
            backend=self,
            collection_name=collection_name,
        )

    def ensure_collection(self, collection_name: str) -> None:
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)

    def drop_collection(self, collection_name: str) -> None:
        if collection_name in self.db.list_collection_names():
            self.db.drop_collection(collection_name)
        self._collection_generation.pop(collection_name, None)
        self._indexed_generation.pop(collection_name, None)

    def list_collection_names(self) -> List[str]:
        return self.db.list_collection_names()

    def upsert_many(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        unique_fields: List[str],
    ) -> None:
        if not documents:
            return
        collection = self.db.get_collection(collection_name)
        operations = []
        for document in documents:
            filter_query = {field: document.get(field) for field in unique_fields}
            operations.append(
                # Insert only when unique_fields are new; never overwrite existing docs.
                UpdateOne(filter_query, {"$setOnInsert": document}, upsert=True)
            )
        if operations:
            collection.bulk_write(operations)
            self._mark_collection_changed(collection_name)

    def _run_vector_search(self, collection: Any, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return list(collection.aggregate(pipeline))

    def vector_search(self, query: VectorQuery) -> List[Dict[str, Any]]:
        collection = self.db.get_collection(query.collection_name)
        if not query.index_name:
            raise ValueError(
                "MongoBackend.vector_search requires VectorQuery.index_name to be set."
            )
        num_candidates = max(query.limit * 10, 150)
        vector_stage: Dict[str, Any] = {
            "index": query.index_name,
            "queryVector": query.query_vector,
            "path": query.vector_field,
            "numCandidates": num_candidates,
            "limit": query.limit,
        }
        if query.filters:
            # MQL-style pre-filter; token fields must be declared on the search index.
            vector_stage["filter"] = query.filters

        pipeline: List[Dict[str, Any]] = [{"$vectorSearch": vector_stage}]
        if query.projection:
            pipeline.append({"$project": query.projection})

        # After recent writes, Atlas may return [] until new docs are indexed.
        needs_sync = self._needs_search_sync(query.collection_name)
        attempts = SEARCH_INDEX_SYNC_MAX_ATTEMPTS if needs_sync else 1

        for attempt in range(attempts):
            try:
                results = self._run_vector_search(collection, pipeline)
            except Exception as exc:
                if attempt == attempts - 1:
                    raise RuntimeError(
                        "MongoBackend.vector_search requires native MongoDB "
                        "$vectorSearch support. The query failed and fallback "
                        "is disabled."
                    ) from exc
                time.sleep(SEARCH_INDEX_SYNC_DELAY_SECONDS)
                continue

            if results:
                self._mark_collection_indexed(query.collection_name)
                return results

            # Stable data: empty means no matches. After writes: retry before giving up.
            if not needs_sync or attempt == attempts - 1:
                return results

            time.sleep(SEARCH_INDEX_SYNC_DELAY_SECONDS)

        return []

    def match_documents(
        self, collection_name: str, query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        collection = self.get_collection(collection_name)
        return collection.find(query)
