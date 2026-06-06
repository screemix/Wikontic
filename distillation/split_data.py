"""
Split Wikontic distillation data into train / val / test at the ARTICLE level.

This prevents data leakage: when paragraphs from the same article are shuffled
into different splits, entities overlap and validation metrics are inflated.
By splitting on article_id (sample_id), every entity in val/test is truly
unseen during training.

Usage:
    python split_data.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42

Outputs (in ./data/):
    train.jsonl, val.jsonl, test.jsonl   – chat-formatted examples
    split_manifest.json                  – which article_ids went where
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Paths (relative to this script's location in distillation/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
HOTPOT_PATH = _SCRIPT_DIR / ".." / "datasets" / "hotpotqa200.json"
DUMP_PATH = _SCRIPT_DIR / ".." / "datasets" / "kg_dump_hotpot_gpt4_1_onto_triplets.json"
SYSTEM_PROMPT_PATH = (
    _SCRIPT_DIR
    / ".."
    / "src"
    / "wikontic"
    / "utils"
    / "prompts"
    / "triplet_extraction"
    / "propmt_1_types_qualifiers.txt"
)
OUTPUT_DIR = _SCRIPT_DIR / "data"

TRAIN_OUT = OUTPUT_DIR / "train.jsonl"
VAL_OUT = OUTPUT_DIR / "val.jsonl"
TEST_OUT = OUTPUT_DIR / "test.jsonl"
MANIFEST_OUT = OUTPUT_DIR / "split_manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_example(text: str, triplets: list, system_prompt: str) -> dict:
    """Format one text-triplets pair as chat messages for SFT training."""
    assistant_content = json.dumps({"triplets": triplets}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Text: "{text}"'},
        {"role": "assistant", "content": assistant_content},
    ]
    return {"messages": messages}


def get_text(ex: dict) -> str:
    return ex["messages"][1]["content"]


def get_entities(examples: List[dict]) -> Set[str]:
    """Extract all subject and object strings from triplets."""
    entities = set()
    for ex in examples:
        try:
            content = ex["messages"][2]["content"]
            parsed = json.loads(content)
            for t in parsed.get("triplets", []):
                if t.get("subject"):
                    entities.add(t["subject"])
                if t.get("object"):
                    entities.add(t["object"])
        except (json.JSONDecodeError, IndexError, KeyError):
            pass
    return entities


def write_jsonl(path: Path, items: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Build article-level examples
# ---------------------------------------------------------------------------

def build_article_examples(
    hotpot: Dict[str, Any],
    dump: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, List[dict]]:
    """
    Returns a dict mapping article_id -> list of chat-formatted examples.
    """
    article_examples: Dict[str, List[dict]] = {}
    skipped_empty = 0

    for sample_id, source_dict in dump.items():
        if sample_id not in hotpot:
            continue
        sample = hotpot[sample_id]
        context = sample["context"]  # list of [title, [text_segments]]
        examples = []

        for sid_str, entry in source_dict.items():
            sid = int(sid_str)
            if sid >= len(context):
                continue

            title, text_segments = context[sid]
            text = " ".join(text_segments).strip()
            triplets = entry.get("triplets", [])

            if not text or not triplets:
                skipped_empty += 1
                continue

            examples.append(build_example(text, triplets, system_prompt))

        if examples:
            article_examples[sample_id] = examples

    print(f"Articles with examples: {len(article_examples)}")
    print(f"Skipped (empty text or triplets): {skipped_empty}")
    return article_examples


# ---------------------------------------------------------------------------
# Split articles
# ---------------------------------------------------------------------------

def split_articles(
    article_examples: Dict[str, List[dict]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Shuffle article IDs and split into train / val / test.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    article_ids = list(article_examples.keys())
    random.seed(seed)
    random.shuffle(article_ids)

    n = len(article_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Test gets the remainder to avoid rounding issues.
    n_test = n - n_train - n_val

    train_ids = article_ids[:n_train]
    val_ids = article_ids[n_train : n_train + n_val]
    test_ids = article_ids[n_train + n_val :]

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------

def check_leakage(
    train: List[dict],
    val: List[dict],
    test: List[dict],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> None:
    """Print leakage statistics."""
    print("\n" + "=" * 70)
    print("DATA LEAKAGE CHECK")
    print("=" * 70)

    # 1. Article overlap
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    test_id_set = set(test_ids)

    print("\n1. ARTICLE OVERLAP")
    print(f"   Train articles: {len(train_ids)}")
    print(f"   Val articles:   {len(val_ids)}")
    print(f"   Test articles:  {len(test_ids)}")
    print(f"   Train-Val overlap:   {len(train_id_set & val_id_set)}")
    print(f"   Train-Test overlap:  {len(train_id_set & test_id_set)}")
    print(f"   Val-Test overlap:    {len(val_id_set & test_id_set)}")

    # 2. Exact text overlap
    train_texts = set(get_text(ex) for ex in train)
    val_texts = set(get_text(ex) for ex in val)
    test_texts = set(get_text(ex) for ex in test)

    print("\n2. EXACT TEXT OVERLAP")
    print(f"   Train-Val exact duplicates:  {len(train_texts & val_texts)}")
    print(f"   Train-Test exact duplicates: {len(train_texts & test_texts)}")
    print(f"   Val-Test exact duplicates:   {len(val_texts & test_texts)}")

    # 3. Entity overlap
    train_entities = get_entities(train)
    val_entities = get_entities(val)
    test_entities = get_entities(test)

    print("\n3. ENTITY OVERLAP")
    print(f"   Total unique train entities: {len(train_entities)}")
    print(f"   Total unique val   entities: {len(val_entities)}")
    print(f"   Total unique test  entities: {len(test_entities)}")

    if val_entities:
        val_overlap = len(train_entities & val_entities)
        print(f"   Entities in train & val:     {val_overlap} ({val_overlap/len(val_entities)*100:.1f}%)")
    if test_entities:
        test_overlap = len(train_entities & test_entities)
        print(f"   Entities in train & test:    {test_overlap} ({test_overlap/len(test_entities)*100:.1f}%)")
    if test_entities and val_entities:
        vt_overlap = len(val_entities & test_entities)
        print(f"   Entities in val & test:      {vt_overlap} ({vt_overlap/len(test_entities)*100:.1f}%)")

    # 4. Per-example entity leakage (val only)
    print("\n4. PER-EXAMPLE ENTITY LEAKAGE (VAL)")
    fully_leaked = 0
    partially_leaked = 0
    fully_unseen = 0

    for ex in val:
        try:
            content = ex["messages"][2]["content"]
            parsed = json.loads(content)
            triplets = parsed.get("triplets", [])
            ex_entities = set()
            for t in triplets:
                if t.get("subject"):
                    ex_entities.add(t["subject"])
                if t.get("object"):
                    ex_entities.add(t["object"])
            if not ex_entities:
                continue
            overlap_count = len(ex_entities & train_entities)
            if overlap_count == len(ex_entities):
                fully_leaked += 1
            elif overlap_count > 0:
                partially_leaked += 1
            else:
                fully_unseen += 1
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    total = fully_leaked + partially_leaked + fully_unseen
    if total:
        print(f"   All entities in train:   {fully_leaked} ({fully_leaked/total*100:.1f}%)")
        print(f"   Some entities in train:  {partially_leaked} ({partially_leaked/total*100:.1f}%)")
        print(f"   All entities UNSEEN:     {fully_unseen} ({fully_unseen/total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if train_id_set & val_id_set or train_id_set & test_id_set or val_id_set & test_id_set:
        print("CRITICAL: Article overlap detected — split logic is broken!")
    else:
        print("Article-level split is clean: no articles shared between splits.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Wikontic distillation data by article_id."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of articles for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of articles for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of articles for test (default: 0.1). Set to 0 to skip test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.test_ratio == 0.0:
        # Adjust train/val to sum to 1.0 if test is disabled.
        total = args.train_ratio + args.val_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train-ratio + val-ratio must equal 1.0 when test-ratio is 0")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    hotpot = {s["_id"]: s for s in load_json(HOTPOT_PATH)}
    dump = load_json(DUMP_PATH)
    system_prompt = load_system_prompt()

    print(f"HotPotQA samples: {len(hotpot)}")
    print(f"KG dump samples:  {len(dump)}")
    print(f"Overlapping IDs:  {len(set(hotpot) & set(dump))}")
    print(f"System prompt:    {len(system_prompt)} chars")

    # ------------------------------------------------------------------
    # Build examples grouped by article
    # ------------------------------------------------------------------
    article_examples = build_article_examples(hotpot, dump, system_prompt)
    total_examples = sum(len(v) for v in article_examples.values())
    print(f"Total paragraph examples: {total_examples}")

    # ------------------------------------------------------------------
    # Split articles
    # ------------------------------------------------------------------
    train_ids, val_ids, test_ids = split_articles(
        article_examples,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    train_examples = []
    for aid in train_ids:
        train_examples.extend(article_examples[aid])

    val_examples = []
    for aid in val_ids:
        val_examples.extend(article_examples[aid])

    test_examples = []
    for aid in test_ids:
        test_examples.extend(article_examples[aid])

    print(f"\nSplit result:")
    print(f"  Train: {len(train_ids)} articles, {len(train_examples)} examples")
    print(f"  Val:   {len(val_ids)} articles, {len(val_examples)} examples")
    if args.test_ratio > 0:
        print(f"  Test:  {len(test_ids)} articles, {len(test_examples)} examples")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_OUT, train_examples)
    write_jsonl(VAL_OUT, val_examples)
    if args.test_ratio > 0:
        write_jsonl(TEST_OUT, test_examples)
    else:
        # Remove stale test file if it exists from a previous run.
        if TEST_OUT.exists():
            TEST_OUT.unlink()

    manifest = {
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids if args.test_ratio > 0 else [],
        "counts": {
            "train_articles": len(train_ids),
            "val_articles": len(val_ids),
            "test_articles": len(test_ids) if args.test_ratio > 0 else 0,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples) if args.test_ratio > 0 else 0,
        },
    }
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"  {TRAIN_OUT.name}")
    print(f"  {VAL_OUT.name}")
    if args.test_ratio > 0:
        print(f"  {TEST_OUT.name}")
    print(f"  {MANIFEST_OUT.name}")

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------
    check_leakage(
        train_examples,
        val_examples,
        test_examples if args.test_ratio > 0 else [],
        train_ids,
        val_ids,
        test_ids if args.test_ratio > 0 else [],
    )


if __name__ == "__main__":
    main()
