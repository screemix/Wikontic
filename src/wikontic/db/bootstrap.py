from __future__ import annotations

from typing import Iterable


def ensure_collections(backend, collection_names: Iterable[str], drop_collections: bool) -> None:
    if drop_collections:
        for name in list(backend.list_collection_names()):
            backend.drop_collection(name)
    for name in collection_names:
        backend.ensure_collection(name)
