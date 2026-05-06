from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional

from pymongo import ASCENDING, UpdateOne
from pymongo.operations import SearchIndexModel

from .interfaces import VectorQuery

@dataclass
class MongoCollectionAdapter:
    collection: Any

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

    def insert_many(self, documents: Iterable[Dict[str, Any]]) -> None:
        docs = list(documents)
        if docs:
            self.collection.insert_many(docs)

    def count_documents(self, query: Dict[str, Any]) -> int:
        return self.collection.count_documents(query)


class MongoBackend:
    def __init__(self, mongo_db):
        self.db = mongo_db

    def ensure_vector_index(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        num_dimensions: int,
        similarity: str = "cosine",
        token_fields: Optional[List[str]] = None,
    ) -> None:
        collection = self.db.get_collection(collection_name)
        mapping_fields: Dict[str, Any] = {
            vector_field: {
                "dimensions": num_dimensions,
                "similarity": similarity,
                "type": "knnVector",
            }
        }
        for field in token_fields or []:
            mapping_fields[field] = {"type": "token"}
        index_definition: Dict[str, Any] = {
            "mappings": {"dynamic": True, "fields": mapping_fields}
        }

        existing_indexes = list(collection.list_search_indexes())
        for existing in existing_indexes:
            if existing.get("name") == index_name:
                return
        model = SearchIndexModel(
            definition=index_definition,
            name=index_name,
        )
        collection.create_search_index(model=model)

    def create_indexes(self, collection_name: str, indexes: List[List[tuple]]) -> None:
        collection = self.db.get_collection(collection_name)
        for index_fields in indexes:
            collection.create_index([(field, ASCENDING) for field in index_fields])

    def get_collection(self, collection_name: str) -> MongoCollectionAdapter:
        return MongoCollectionAdapter(self.db.get_collection(collection_name))

    def ensure_collection(self, collection_name: str) -> None:
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)

    def drop_collection(self, collection_name: str) -> None:
        if collection_name in self.db.list_collection_names():
            self.db.drop_collection(collection_name)

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
                UpdateOne(filter_query, {"$setOnInsert": document}, upsert=True)
            )
        if operations:
            collection.bulk_write(operations)

    def vector_search(self, query: VectorQuery) -> List[Dict[str, Any]]:
        collection = self.db.get_collection(query.collection_name)
        if not query.index_name:
            raise ValueError(
                "MongoBackend.vector_search requires VectorQuery.index_name to be set."
            )
        vector_stage: Dict[str, Any] = {
            "index": query.index_name,
            "queryVector": query.query_vector,
            "path": query.vector_field,
            "numCandidates": 150 if query.limit < 150 else query.limit,
            "limit": query.limit,
        }
        if query.filters:
            vector_stage["filter"] = query.filters

        pipeline = [{"$vectorSearch": vector_stage}]
        if query.projection:
            pipeline.append({"$project": query.projection})
        try:
            return list(collection.aggregate(pipeline))
        except Exception as exc:
            raise RuntimeError(
                "MongoBackend.vector_search requires native MongoDB $vectorSearch support. "
                "The query failed and fallback is disabled."
            ) from exc

    def match_documents(
        self, collection_name: str, query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        collection = self.get_collection(collection_name)
        return collection.find(query)
