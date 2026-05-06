from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient, models

from .interfaces import VectorQuery


def _deterministic_id(collection_name: str, payload: Dict[str, Any]) -> str:
    key = f"{collection_name}:{payload}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _query_to_filter(query: Dict[str, Any]) -> Optional[models.Filter]:
    if not query:
        return None

    must_conditions: List[Any] = []
    should_conditions: List[Any] = []

    for key, value in query.items():
        if key == "$and":
            if not isinstance(value, list):
                raise ValueError("`$and` expects a list of query objects")
            for item in value:
                sub_filter = _query_to_filter(item)
                if sub_filter is not None:
                    must_conditions.append(sub_filter)
            continue

        if key == "$or":
            if not isinstance(value, list):
                raise ValueError("`$or` expects a list of query objects")
            for item in value:
                sub_filter = _query_to_filter(item)
                if sub_filter is not None:
                    should_conditions.append(sub_filter)
            continue

        if isinstance(value, dict):
            if "$eq" in value:
                must_conditions.append(
                    models.FieldCondition(
                        key=key, match=models.MatchValue(value=value["$eq"])
                    )
                )
                continue
            if "$in" in value:
                must_conditions.append(
                    models.FieldCondition(
                        key=key, match=models.MatchAny(any=value["$in"])
                    )
                )
                continue
            raise ValueError(
                f"Unsupported filter operator for key '{key}'. "
                "Supported operators: $eq, $in, $or, $and."
            )

        must_conditions.append(
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
        )

    if not must_conditions and not should_conditions:
        return None
    return models.Filter(must=must_conditions or None, should=should_conditions or None)


def _payload_selector_from_projection(
    projection: Optional[Dict[str, int]],
) -> Any:
    if not projection:
        return True
    include_fields = [
        key for key, include in projection.items() if include and key != "_id"
    ]
    if not include_fields:
        return True
    return include_fields


def _apply_projection(
    payload: Dict[str, Any], projection: Optional[Dict[str, int]]
) -> Dict[str, Any]:
    if not projection:
        return payload
    include_fields = {
        key for key, include in projection.items() if include and key != "_id"
    }
    if not include_fields:
        return payload
    return {key: payload.get(key) for key in include_fields if key in payload}


@dataclass
class QdrantCollectionAdapter:
    backend: "QdrantBackend"
    collection_name: str

    def find_one(
        self, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None
    ) -> Optional[Dict[str, Any]]:
        docs = self.find(query=query, projection=projection)
        return docs[0] if docs else None

    def find(
        self, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        docs = self.backend.match_documents(self.collection_name, query)
        if not projection:
            return docs
        include_fields = {k for k, v in projection.items() if v}
        if not include_fields:
            return docs
        projected = []
        for doc in docs:
            projected.append({k: doc.get(k) for k in include_fields if k in doc})
        return projected

    def insert_one(self, document: Dict[str, Any]) -> None:
        self.backend._upsert_documents(self.collection_name, [document], unique_fields=None)

    def insert_many(self, documents: Iterable[Dict[str, Any]]) -> None:
        docs = list(documents)
        if docs:
            self.backend._upsert_documents(self.collection_name, docs, unique_fields=None)

    def count_documents(self, query: Dict[str, Any]) -> int:
        return len(self.backend.match_documents(self.collection_name, query))


class QdrantBackend:
    def __init__(self, qdrant_url: str = ":memory:", api_key: Optional[str] = None):
        if qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url, api_key=api_key)
        self._collection_vector_size: Dict[str, int] = {}

    def get_collection(self, collection_name: str) -> QdrantCollectionAdapter:
        self.ensure_collection(collection_name)
        return QdrantCollectionAdapter(self, collection_name)

    def ensure_collection(self, collection_name: str) -> None:
        if collection_name in self.list_collection_names():
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        self._collection_vector_size[collection_name] = 1

    def drop_collection(self, collection_name: str) -> None:
        if collection_name in self.list_collection_names():
            self.client.delete_collection(collection_name)
            self._collection_vector_size.pop(collection_name, None)

    def list_collection_names(self) -> List[str]:
        return [item.name for item in self.client.get_collections().collections]

    def _ensure_vector_size(self, collection_name: str, vector_size: int) -> None:
        current_size = self._collection_vector_size.get(collection_name, 1)
        if collection_name not in self.list_collection_names():
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            self._collection_vector_size[collection_name] = vector_size
            return
        if current_size == vector_size:
            return
        points = self._scroll_payloads(collection_name)
        self.client.delete_collection(collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
        self._collection_vector_size[collection_name] = vector_size
        if points:
            self._upsert_documents(collection_name, points, unique_fields=None)

    def _scroll_payloads(
        self, collection_name: str, query_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        self.ensure_collection(collection_name)
        offset = None
        records: List[Dict[str, Any]] = []
        scroll_filter = _query_to_filter(query_filter or {})
        while True:
            batch, offset = self.client.scroll(
                collection_name=collection_name,
                limit=256,
                scroll_filter=scroll_filter,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend([item.payload for item in batch if item.payload is not None])
            if offset is None:
                break
        return records

    def _upsert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        unique_fields: Optional[List[str]],
    ) -> None:
        if not documents:
            return
        vector_size = 1
        for item in documents:
            candidate = item.get("alias_text_embedding")
            if isinstance(candidate, list) and candidate:
                vector_size = len(candidate)
                break
        self._ensure_vector_size(collection_name, vector_size)
        points: List[models.PointStruct] = []
        for document in documents:
            vector = document.get("alias_text_embedding")
            if not isinstance(vector, list) or len(vector) != vector_size:
                vector = [0.0] * vector_size
            if unique_fields:
                identity = {field: document.get(field) for field in unique_fields}
                point_id = _deterministic_id(collection_name, identity)
            else:
                point_id = _deterministic_id(collection_name, document)
            points.append(models.PointStruct(id=point_id, vector=vector, payload=document))
        self.client.upsert(collection_name=collection_name, points=points)

    def upsert_many(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        unique_fields: List[str],
    ) -> None:
        self._upsert_documents(collection_name, documents, unique_fields=unique_fields)

    def vector_search(self, query: VectorQuery) -> List[Dict[str, Any]]:
        self.ensure_collection(query.collection_name)
        if query.vector_field != "alias_text_embedding":
            raise ValueError(
                "QdrantBackend.vector_search currently supports `alias_text_embedding` "
                "as the logical vector field."
            )
        response = self.client.query_points(
            collection_name=query.collection_name,
            query=query.query_vector,
            using=query.index_name,
            query_filter=_query_to_filter(query.filters or {}),
            limit=query.limit,
            with_payload=_payload_selector_from_projection(query.projection),
            with_vectors=False,
        )
        points = getattr(response, "points", response)
        payloads: List[Dict[str, Any]] = []
        for point in points:
            payload = getattr(point, "payload", None)
            if payload is None:
                continue
            payloads.append(_apply_projection(payload, query.projection))
        return payloads

    def match_documents(
        self, collection_name: str, query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # Exact payload filtering semantics (Mongo-like find query),
        # implemented via Qdrant scroll_filter.
        return self._scroll_payloads(collection_name, query_filter=query)
