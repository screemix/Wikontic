from __future__ import annotations

from typing import Any, Iterable, TypedDict

import tiktoken


class TokenStats(TypedDict):
    source_tokens: int
    triplet_tokens: int
    compression_ratio: float
    savings_pct: float
    triplet_text: str

DEFAULT_MODEL = "gpt-4.1"


def get_token_encoder(model: str = DEFAULT_MODEL):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    if not text:
        return 0
    encoder = get_token_encoder(model)
    return len(encoder.encode(text))


def format_qualifiers(qualifiers: Iterable[dict[str, Any]] | None) -> str:
    if not qualifiers:
        return ""
    parts = []
    for qualifier in qualifiers:
        relation = str(qualifier.get("relation", "")).strip()
        obj = str(qualifier.get("object", "")).strip()
        if relation and obj:
            parts.append(f"{relation}: {obj}")
        elif relation:
            parts.append(relation)
        elif obj:
            parts.append(obj)
    return ", ".join(parts)


def verbalize_triplet(triplet: dict[str, Any]) -> str:
    subject = str(triplet.get("subject", "")).strip()
    relation = str(triplet.get("relation", "")).strip()
    obj = str(triplet.get("object", "")).strip()
    qualifiers = format_qualifiers(triplet.get("qualifiers"))
    return f"{subject}, {relation}, {obj} | {qualifiers}"


def verbalize_triplets(triplets: Iterable[dict[str, Any]]) -> str:
    lines = [verbalize_triplet(triplet) for triplet in triplets]
    return "\n".join(lines)


def compare_text_and_triplets(
    text: str,
    triplets: Iterable[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
) -> TokenStats:
    source_tokens = count_tokens(text, model=model)
    triplet_text = verbalize_triplets(triplets)
    triplet_tokens = count_tokens(triplet_text, model=model)
    if source_tokens:
        compression_ratio = triplet_tokens / source_tokens
        savings_pct = (1 - compression_ratio) * 100
    else:
        compression_ratio = 0.0
        savings_pct = 0.0
    return {
        "source_tokens": source_tokens,
        "triplet_tokens": triplet_tokens,
        "compression_ratio": compression_ratio,
        "savings_pct": savings_pct,
        "triplet_text": triplet_text,
    }
