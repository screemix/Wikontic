from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol


@dataclass
class VectorQuery:
    """Backend-agnostic vector query parameters.

    Note:
    - `filters` follow a Mongo-like subset used by this codebase and may be
      translated by each backend implementation (`$eq`, `$in`, `$or`, `$and`).
    - `index_name` is required for MongoDB Atlas `$vectorSearch`, and maps to
      Qdrant named vector via `using`.
    """

    collection_name: str
    index_name: Optional[str]
    query_vector: List[float]
    vector_field: str
    limit: int
    filters: Optional[Dict[str, Any]] = None
    projection: Optional[Dict[str, int]] = None


class CollectionLike(Protocol):
    def find_one(
        self, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None
    ) -> Optional[Dict[str, Any]]:
        ...

    def find(
        self, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        ...

    def insert_one(self, document: Dict[str, Any]) -> None:
        ...

    def insert_many(self, documents: Iterable[Dict[str, Any]]) -> None:
        ...

    def count_documents(self, query: Dict[str, Any]) -> int:
        ...


class StorageBackend(Protocol):
    """Storage and retrieval contract used by aligners/inference code."""

    def get_collection(self, collection_name: str) -> CollectionLike:
        ...

    def ensure_collection(self, collection_name: str) -> None:
        ...

    def drop_collection(self, collection_name: str) -> None:
        ...

    def list_collection_names(self) -> List[str]:
        ...

    def create_indexes(self, collection_name: str, indexes: List[List[str]]) -> None:
        ...

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
        ...

    def upsert_many(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        unique_fields: List[str],
    ) -> None:
        ...

    def vector_search(self, query: VectorQuery) -> List[Dict[str, Any]]:
        ...

    def match_documents(
        self, collection_name: str, query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        ...
