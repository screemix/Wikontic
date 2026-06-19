from __future__ import annotations

from typing import Optional, Union

from .mongo_backend import MongoBackend
from .qdrant_backend import QdrantBackend


def ensure_storage_backend(db) -> Union[MongoBackend, QdrantBackend]:
    """Wrap a raw pymongo Database; pass through an existing backend unchanged.

    Atlas/pymongo Database objects expose collection names via __getattr__, so
    hasattr(db, \"vector_search\") / hasattr(db, \"upsert_many\") are true even
    though those are Collection handles, not backend methods.
    """
    if callable(getattr(type(db), "upsert_many", None)):
        return db
    return MongoBackend(db)


def create_backend(
    backend_type: str,
    mongo_db=None,
    qdrant_url: str = ":memory:",
    qdrant_api_key: Optional[str] = None,
):
    normalized = (backend_type or "mongodb").lower()
    if normalized == "mongodb":
        if mongo_db is None:
            raise ValueError("mongo_db must be provided when backend_type='mongodb'")
        return MongoBackend(mongo_db)
    if normalized == "qdrant":
        return QdrantBackend(qdrant_url=qdrant_url, api_key=qdrant_api_key)
    raise ValueError(f"Unsupported backend_type: {backend_type}")
