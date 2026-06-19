"""
Chunk EDGAR preprocessed JSONL into per-section sentence groups (3-5 sentences per chunk).

Reads records with section_* keys (numeric Item order, optional letter suffix e.g. 7A); section_1A / section_1B omitted; each has "original" / "replaced" text.
Output: doc_id -> list of {section_id: {chunk_num: chunk_text}} (chunk_num starts at 1).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any
import spacy

from wikontic.logging_config import get_logger

logger = get_logger(__name__)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")


def split_sentences(text: str) -> list[str]:
    return [sent.text.strip() for sent in nlp(text).sents]


def _next_chunk_size(num_remaining: int, rng: random.Random) -> int:
    """Random size in [3, 5] unless fewer sentences remain; avoid 1–3 stragglers."""
    if num_remaining <= 5:
        return num_remaining
    hi = min(5, num_remaining)
    choices = [
        k
        for k in range(3, hi + 1)
        if (num_remaining - k) == 0 or (num_remaining - k) >= 3
    ]
    if not choices:
        return num_remaining
    return rng.choice(choices)


def chunk_sentences(sentences: list[str], rng: random.Random) -> list[str]:
    if not sentences:
        return []
    chunks_sents: list[list[str]] = []
    i = 0
    n = len(sentences)
    while i < n:
        rem = n - i
        if rem < 4:
            tail = sentences[i:]
            if chunks_sents:
                chunks_sents[-1].extend(tail)
            else:
                chunks_sents.append(tail)
            break
        k = _next_chunk_size(rem, rng)
        chunks_sents.append(sentences[i : i + k])
        i += k
    return [" ".join(s) for s in chunks_sents]


def section_text(payload: dict[str, Any], field: str) -> str:
    if field not in ("original", "replaced"):
        raise ValueError("field must be 'original' or 'replaced'")
    if not isinstance(payload, dict):
        return ""
    val = payload.get(field, "")
    return val if isinstance(val, str) else ""


def chunk_record(
    record: dict[str, Any],
    *,
    rng: random.Random,
    text_field: str,
    doc_id_key: str,
) -> tuple[str, list[dict[str, dict[str, str]]]]:
    doc_id = str(record.get(doc_id_key, ""))
    if not doc_id:
        raise ValueError(f"Record missing {doc_id_key!r}")

    for sk in section_keys:
        raw = section_text(record[sk], text_field)
        sents = split_sentences(raw)
        if not sents:
            continue
        blob_list = chunk_sentences(sents, rng)
        chunk_map = {str(j + 1): text for j, text in enumerate(blob_list)}
        out.append({sk: chunk_map})

    return doc_id, out


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()  # Read entire file
        lines = content.splitlines(keepends=True)  # Keep original newlines

        i = 0
        while i < len(lines):
            # Collect all lines until we find a complete JSON object
            buffer = ""
            while i < len(lines):
                buffer += lines[i]
                try:
                    item = json.loads(buffer)
                    data.append(item)
                    i += 1
                    break
                except json.JSONDecodeError:
                    i += 1
            else:
                logger.warning(f"Warning: Incomplete JSON at end of file")
                break
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk EDGAR JSONL by section.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/edgar_preprocessed/edgar_1994_disambiguated.jsonl"),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/edgar_preprocessed/edgar_1994_disambiguated_chunks.json"
        ),
        help="Output JSON path (doc_id -> list of section chunk dicts)",
    )
    parser.add_argument(
        "--text-field",
        choices=("replaced", "original"),
        default="replaced",
        help="Which section subfield to chunk",
    )
    parser.add_argument(
        "--doc-id-key",
        default="filename",
        help="Record field to use as document id (default: filename)",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (optional)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = load_jsonl(args.input)
    mapping: dict[str, list[dict[str, dict[str, str]]]] = {}
    for record in data:
        doc_id, chunks = chunk_record(
            record,
            rng=rng,
            text_field=args.text_field,
            doc_id_key=args.doc_id_key,
        )
        mapping[doc_id] = chunks

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(mapping)} documents to {args.output}")


if __name__ == "__main__":
    main()
