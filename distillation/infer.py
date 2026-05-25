"""
Inference & evaluation script for Wikontic distillation — v2.

Runs the fine-tuned model on validation data, parses predicted triplets,
applies whitelist constraints, and produces a detailed comparison report
including distribution analysis, per-relation P/R/F1, subject/object
hallucination, partial-match metrics, and confused-relation analysis.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def extract_whitelist(train_examples: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Extract unique relations, subject_types, and object_types from training data."""
    relations: Set[str] = set()
    subject_types: Set[str] = set()
    object_types: Set[str] = set()
    for ex in train_examples:
        try:
            parsed = json.loads(ex["messages"][2]["content"])
            for t in parsed.get("triplets", []):
                if t.get("relation"):
                    relations.add(t["relation"])
                if t.get("subject_type"):
                    subject_types.add(t["subject_type"])
                if t.get("object_type"):
                    object_types.add(t["object_type"])
        except (json.JSONDecodeError, IndexError, KeyError):
            pass
    return {
        "relations": relations,
        "subject_types": subject_types,
        "object_types": object_types,
    }


def _similar(a: str, b: str) -> float:
    """Case-insensitive string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def apply_whitelist(
    triplets: List[Dict[str, Any]],
    whitelist: Dict[str, Set[str]],
    soft_mapping: bool = True,
    min_similarity: float = 0.6,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filter triplets against training whitelist.

    * Dropping unknown relations entirely.
    * If ``soft_mapping=True``, map unknown relations to the most similar
      valid relation when similarity >= ``min_similarity``.
    * Type mismatches are mapped when possible but never cause dropping.

    Returns (valid_triplets, stats).
    """
    valid_relations = whitelist.get("relations", set())
    valid_subj_types = whitelist.get("subject_types", set())
    valid_obj_types = whitelist.get("object_types", set())

    valid: List[Dict[str, Any]] = []
    stats: Dict[str, int] = defaultdict(int)
    for t in triplets:
        rel = t.get("relation", "")
        st = t.get("subject_type", "")
        ot = t.get("object_type", "")

        # --- relation ---
        mapped_rel = rel
        if rel not in valid_relations:
            if soft_mapping and valid_relations:
                best = max(valid_relations, key=lambda v: _similar(rel, v))
                if _similar(rel, best) >= min_similarity:
                    mapped_rel = best
                    stats["relation_soft_mapped"] += 1
                else:
                    stats["relation_dropped"] += 1
                    continue
            else:
                stats["relation_dropped"] += 1
                continue

        # --- types (try to map, but never drop) ---
        mapped_st = st
        if st and st not in valid_subj_types:
            if soft_mapping and valid_subj_types:
                best = max(valid_subj_types, key=lambda v: _similar(st, v))
                if _similar(st, best) >= min_similarity:
                    mapped_st = best
                    stats["subject_type_mapped"] += 1

        mapped_ot = ot
        if ot and ot not in valid_obj_types:
            if soft_mapping and valid_obj_types:
                best = max(valid_obj_types, key=lambda v: _similar(ot, v))
                if _similar(ot, best) >= min_similarity:
                    mapped_ot = best
                    stats["object_type_mapped"] += 1

        new_t = dict(t)
        new_t["relation"] = mapped_rel
        new_t["subject_type"] = mapped_st
        new_t["object_type"] = mapped_ot
        valid.append(new_t)

    return valid, dict(stats)


def parse_completion_triplets(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract triplets from the completion field of a chat example."""
    try:
        completion = example["messages"][2]["content"]
        data = json.loads(completion)
        if isinstance(data, dict) and "triplets" in data:
            return data["triplets"]
    except (json.JSONDecodeError, TypeError, IndexError, KeyError):
        pass
    return []


def robust_extract_triplets(text: str) -> Optional[Dict[str, Any]]:
    """Robust triplet extractor with multiple fallback strategies."""
    cleaned = text.rstrip(". \n\t")
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "triplets" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: find all JSON objects via brace-depth matching
    candidates = []
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 1
            for j in range(i + 1, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[i : j + 1])
                        break

    triplets = []
    seen = set()
    for candidate in candidates:
        if '"subject"' not in candidate or '"relation"' not in candidate or '"object"' not in candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and all(k in obj for k in ("subject", "relation", "object")):
                key = (obj.get("subject"), obj.get("relation"), obj.get("object"))
                if key not in seen:
                    seen.add(key)
                    triplets.append(obj)
        except (json.JSONDecodeError, TypeError):
            pass

    if triplets:
        return {"triplets": triplets}

    # Attempt 3: largest top-level dict
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        for candidate in [text[start : end + 1], text[start : end + 1].rstrip(".")]:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "triplets" in data:
                    return data
            except (json.JSONDecodeError, TypeError):
                pass
    return None


def triplet_key_full(t: Dict[str, Any]) -> Tuple:
    return (
        t.get("subject", ""),
        t.get("relation", ""),
        t.get("object", ""),
        t.get("subject_type", ""),
        t.get("object_type", ""),
    )


def triplet_key_sro(t: Dict[str, Any]) -> Tuple:
    """Subject-Relation-Object key (ignores types)."""
    return (
        t.get("subject", ""),
        t.get("relation", ""),
        t.get("object", ""),
    )


def compute_metrics(
    gt_triplets: List[Dict],
    pred_triplets: List[Dict],
    key_fn=triplet_key_full,
) -> Dict[str, Any]:
    gt_set = set(key_fn(t) for t in gt_triplets)
    pred_set = set(key_fn(t) for t in pred_triplets)
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_count": len(gt_set),
        "pred_count": len(pred_set),
    }


def compute_relation_metrics(
    gt_examples: List[List[Dict]],
    pred_examples: List[List[Dict]],
    key_fn=triplet_key_full,
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for gt_triplets, pred_triplets in zip(gt_examples, pred_examples):
        gt_by_rel: Dict[str, set] = {}
        for t in gt_triplets:
            r = t.get("relation", "UNKNOWN")
            gt_by_rel.setdefault(r, set()).add(key_fn(t))
        pred_by_rel: Dict[str, set] = {}
        for t in pred_triplets:
            r = t.get("relation", "UNKNOWN")
            pred_by_rel.setdefault(r, set()).add(key_fn(t))
        all_rels = set(gt_by_rel.keys()) | set(pred_by_rel.keys())
        for r in all_rels:
            if r not in stats:
                stats[r] = {"tp": 0, "gt": 0, "pred": 0}
            gt_set = gt_by_rel.get(r, set())
            pred_set = pred_by_rel.get(r, set())
            stats[r]["tp"] += len(gt_set & pred_set)
            stats[r]["gt"] += len(gt_set)
            stats[r]["pred"] += len(pred_set)
    results = {}
    for r, s in stats.items():
        p = s["tp"] / s["pred"] if s["pred"] else 0.0
        rec = s["tp"] / s["gt"] if s["gt"] else 0.0
        f1 = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0
        results[r] = {"precision": p, "recall": rec, "f1": f1, "support_gt": s["gt"], "support_pred": s["pred"]}
    return results


def extract_text_from_prompt(prompt_content: str) -> str:
    """Strip the 'Text: "..."' wrapper used in the dataset."""
    m = re.search(r'Text:\s*"((?:[^"\\]|\\.)*?)"', prompt_content)
    if m:
        return m.group(1)
    return prompt_content


def compute_hallucination_stats(
    texts: List[str],
    pred_examples: List[List[Dict]],
) -> Dict[str, Any]:
    """
    For each predicted triplet, check whether subject / object strings
    appear (at word level) in the source text.
    """
    text_words = [set(re.findall(r"\b\w+\b", text.lower())) for text in texts]

    subj_hallucinated = 0
    obj_hallucinated = 0
    subj_total = 0
    obj_total = 0

    for text_words_set, pred_triplets in zip(text_words, pred_examples):
        for t in pred_triplets:
            subj = t.get("subject", "").lower()
            obj = t.get("object", "").lower()

            subj_words = set(re.findall(r"\b\w+\b", subj))
            obj_words = set(re.findall(r"\b\w+\b", obj))

            # Consider hallucinated if NONE of the entity's words appear in text
            if subj_words and not subj_words.intersection(text_words_set):
                subj_hallucinated += 1
            subj_total += 1

            if obj_words and not obj_words.intersection(text_words_set):
                obj_hallucinated += 1
            obj_total += 1

    return {
        "subject_hallucination_rate": subj_hallucinated / subj_total if subj_total else 0.0,
        "object_hallucination_rate": obj_hallucinated / obj_total if obj_total else 0.0,
        "subject_hallucinated_count": subj_hallucinated,
        "object_hallucinated_count": obj_hallucinated,
        "total_triplets_checked": subj_total,
    }


def find_confused_relations(
    gt_examples: List[List[Dict]],
    pred_examples: List[List[Dict]],
    top_k: int = 10,
) -> List[Tuple[Tuple[str, str], int]]:
    """
    Find the most common (gt_relation, pred_relation) mismatches.
    Only counts cases where subject & object match but relation differs.
    """
    confusion = Counter()
    for gt_triplets, pred_triplets in zip(gt_examples, pred_examples):
        gt_map = {}
        for t in gt_triplets:
            key = (t.get("subject", ""), t.get("object", ""))
            gt_map[key] = t.get("relation", "UNKNOWN")

        for t in pred_triplets:
            key = (t.get("subject", ""), t.get("object", ""))
            pred_rel = t.get("relation", "UNKNOWN")
            if key in gt_map:
                gt_rel = gt_map[key]
                if gt_rel != pred_rel:
                    confusion[(gt_rel, pred_rel)] += 1

    return confusion.most_common(top_k)


def compute_type_accuracy(
    gt_examples: List[List[Dict]],
    pred_examples: List[List[Dict]],
) -> Dict[str, Any]:
    """Compute per-type accuracy for subject_type and object_type on SRO matches."""
    subj_type_tp = 0
    subj_type_total = 0
    obj_type_tp = 0
    obj_type_total = 0

    for gt_triplets, pred_triplets in zip(gt_examples, pred_examples):
        gt_map = {triplet_key_sro(t): t for t in gt_triplets}
        for t in pred_triplets:
            key = triplet_key_sro(t)
            if key in gt_map:
                gt_t = gt_map[key]
                if t.get("subject_type") == gt_t.get("subject_type"):
                    subj_type_tp += 1
                subj_type_total += 1
                if t.get("object_type") == gt_t.get("object_type"):
                    obj_type_tp += 1
                obj_type_total += 1

    return {
        "subject_type_accuracy": subj_type_tp / subj_type_total if subj_type_total else 0.0,
        "object_type_accuracy": obj_type_tp / obj_type_total if obj_type_total else 0.0,
        "subject_type_checked": subj_type_total,
        "object_type_checked": obj_type_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer.yaml",
        help="Path to YAML inference config (relative to infer.py dir).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Which split to evaluate on. Use 'val' during development, 'test' only for final evaluation.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    script_dir = Path(__file__).resolve().parent

    split = args.split
    data_path_key = f"{split}_path"
    eval_path = script_dir / cfg[data_path_key]
    output_dir = script_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = script_dir / cfg["adapter_path"]

    eval_examples = load_jsonl(eval_path)
    print(f"Loaded {len(eval_examples)} {split} examples from {eval_path}")

    # Tokenizer
    base_model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Quantization
    bnb_config = None
    if cfg.get("load_in_4bit", False):
        compute_dtype = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print(f"Loading base model {base_model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")) if not bnb_config else None,
    )
    print(f"Loading adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    max_new_tokens = cfg.get("max_new_tokens", 2048)
    temperature = cfg.get("temperature", 0.0)
    do_sample = cfg.get("do_sample", False)

    # ── Whitelist from training data ──
    train_path = script_dir / cfg.get("train_path", "./data/train.jsonl")
    train_examples = load_jsonl(train_path)
    whitelist = extract_whitelist(train_examples)
    print(
        f"Whitelist: {len(whitelist['relations'])} relations, "
        f"{len(whitelist['subject_types'])} subj types, "
        f"{len(whitelist['object_types'])} obj types"
    )

    use_whitelist = cfg.get("use_whitelist", True)
    whitelist_soft = cfg.get("whitelist_soft_mapping", True)
    whitelist_min_sim = cfg.get("whitelist_min_similarity", 0.6)

    results = []
    pred_examples_triplets: List[List[Dict]] = []
    gt_examples_triplets: List[List[Dict]] = []
    all_texts: List[str] = []

    for idx, ex in enumerate(eval_examples):
        text = extract_text_from_prompt(ex["messages"][1]["content"])
        prompt_messages = ex["messages"][:]  # system + user
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(output[0], skip_special_tokens=False)

        # Extract assistant answer
        marker = "\nassistant\n"
        marker_pos = full_output.find(marker)
        if marker_pos >= 0:
            answer = full_output[marker_pos + len(marker) :].strip()
        else:
            answer = full_output[len(prompt_text) :].strip()

        for turn_marker in ["<|im_start|>user", "<|im_start|>system", "<|endoftext|>", "</s>"]:
            pos = answer.find(turn_marker)
            if pos >= 0:
                answer = answer[:pos].strip()

        parsed = robust_extract_triplets(answer)
        raw_pred_triplets: List[Dict] = parsed["triplets"] if parsed else []
        gt_triplets = parse_completion_triplets(ex)

        # ── Apply whitelist filtering ──
        if use_whitelist:
            pred_triplets, wl_stats = apply_whitelist(
                raw_pred_triplets,
                whitelist,
                soft_mapping=whitelist_soft,
                min_similarity=whitelist_min_sim,
            )
        else:
            pred_triplets = raw_pred_triplets
            wl_stats = {}

        results.append({
            "index": idx,
            "text": text,
            "raw_output": answer,
            "raw_triplets": raw_pred_triplets,
            "filtered_triplets": pred_triplets,
            "gt_triplets": gt_triplets,
            "whitelist_stats": wl_stats,
        })
        pred_examples_triplets.append(pred_triplets)
        gt_examples_triplets.append(gt_triplets)
        all_texts.append(text)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(eval_examples)} ...")

    # ── Save predictions ──
    preds_filename = f"{split}_predictions.json"
    preds_path = output_dir / preds_filename
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {preds_path}")

    # ── Build report ──
    all_pred_triplets = [t for ex in pred_examples_triplets for t in ex]
    all_gt_triplets = [t for ex in gt_examples_triplets for t in ex]
    all_raw_triplets = [t for ex in results for t in ex["raw_triplets"]]

    pred_relation_counts = Counter(t.get("relation", "UNKNOWN") for t in all_pred_triplets)
    gt_relation_counts = Counter(t.get("relation", "UNKNOWN") for t in all_gt_triplets)

    # Training distribution
    train_triplets = [t for ex in train_examples for t in parse_completion_triplets(ex)]
    train_relation_counts = Counter(t.get("relation", "UNKNOWN") for t in train_triplets)

    lines = []
    lines.append("=" * 70)
    lines.append(f"WIKONTIC DISTILLATION INFERENCE REPORT — {split.upper()} SPLIT")
    lines.append("=" * 70)

    lines.append(f"\n--- Overall counts ---")
    lines.append(f"{split.capitalize()} examples: {len(eval_examples)}")
    lines.append(f"GT triplets total:   {len(all_gt_triplets)}")
    lines.append(f"Raw pred triplets:   {len(all_raw_triplets)}")
    lines.append(f"Filtered pred triplets: {len(all_pred_triplets)}")
    empty = sum(1 for ex in pred_examples_triplets if not ex)
    lines.append(f"Empty predictions:   {empty}")

    # Full-match metrics (subject + relation + object + types)
    total_tp = 0
    total_pred = 0
    total_gt = 0
    for gt, pred in zip(gt_examples_triplets, pred_examples_triplets):
        m = compute_metrics(gt, pred, key_fn=triplet_key_full)
        total_tp += m["tp"]
        total_pred += m["pred_count"]
        total_gt += m["gt_count"]

    overall_p = total_tp / total_pred if total_pred else 0.0
    overall_r = total_tp / total_gt if total_gt else 0.0
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0
    lines.append("\n--- Overall FULL-MATCH metrics (subj+rel+obj+types) ---")
    lines.append(f"Precision: {overall_p:.3f}")
    lines.append(f"Recall:    {overall_r:.3f}")
    lines.append(f"F1:        {overall_f1:.3f}")

    # SRO-only metrics (ignore types)
    total_tp_sro = 0
    total_pred_sro = 0
    total_gt_sro = 0
    for gt, pred in zip(gt_examples_triplets, pred_examples_triplets):
        m = compute_metrics(gt, pred, key_fn=triplet_key_sro)
        total_tp_sro += m["tp"]
        total_pred_sro += m["pred_count"]
        total_gt_sro += m["gt_count"]

    overall_p_sro = total_tp_sro / total_pred_sro if total_pred_sro else 0.0
    overall_r_sro = total_tp_sro / total_gt_sro if total_gt_sro else 0.0
    overall_f1_sro = (2 * overall_p_sro * overall_r_sro / (overall_p_sro + overall_r_sro)) if (overall_p_sro + overall_r_sro) > 0 else 0.0
    lines.append("\n--- Overall SRO-MATCH metrics (subj+rel+obj only) ---")
    lines.append(f"Precision: {overall_p_sro:.3f}")
    lines.append(f"Recall:    {overall_r_sro:.3f}")
    lines.append(f"F1:        {overall_f1_sro:.3f}")
    diff = overall_f1 - overall_f1_sro
    lines.append(f"  (Type labels cost {abs(diff):.3f} F1 {'full>' if diff > 0 else 'SRO>'})")

    # Per-relation metrics (full match)
    rel_metrics = compute_relation_metrics(gt_examples_triplets, pred_examples_triplets, key_fn=triplet_key_full)
    lines.append("\n--- Per-relation metrics FULL-MATCH (top 15 by GT support) ---")
    sorted_rels = sorted(rel_metrics.items(), key=lambda kv: kv[1]["support_gt"], reverse=True)
    for r, m in sorted_rels[:15]:
        lines.append(
            f"  {r:25s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"gt={m['support_gt']:>4}  pred={m['support_pred']:>4}"
        )

    # Per-relation metrics (SRO only)
    rel_metrics_sro = compute_relation_metrics(gt_examples_triplets, pred_examples_triplets, key_fn=triplet_key_sro)
    lines.append("\n--- Per-relation metrics SRO-MATCH (top 15 by GT support) ---")
    sorted_rels_sro = sorted(rel_metrics_sro.items(), key=lambda kv: kv[1]["support_gt"], reverse=True)
    for r, m in sorted_rels_sro[:15]:
        lines.append(
            f"  {r:25s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"gt={m['support_gt']:>4}  pred={m['support_pred']:>4}"
        )

    # Hallucination stats
    hall = compute_hallucination_stats(all_texts, pred_examples_triplets)
    lines.append("\n--- Subject / Object hallucination ---")
    lines.append(
        f"Subject hallucination rate: {hall['subject_hallucination_rate']:.3f} "
        f"({hall['subject_hallucinated_count']}/{hall['total_triplets_checked']})"
    )
    lines.append(
        f"Object  hallucination rate: {hall['object_hallucination_rate']:.3f} "
        f"({hall['object_hallucinated_count']}/{hall['total_triplets_checked']})"
    )

    # Type accuracy
    type_acc = compute_type_accuracy(gt_examples_triplets, pred_examples_triplets)
    lines.append("\n--- Type accuracy (on correctly-matched SRO triplets) ---")
    lines.append(
        f"Subject type accuracy: {type_acc['subject_type_accuracy']:.3f} "
        f"({type_acc['subject_type_checked']} checked)"
    )
    lines.append(
        f"Object  type accuracy: {type_acc['object_type_accuracy']:.3f} "
        f"({type_acc['object_type_checked']} checked)"
    )

    # Confused relations
    confused = find_confused_relations(gt_examples_triplets, pred_examples_triplets)
    lines.append("\n--- Most confused relation pairs (GT relation → predicted relation) ---")
    for (gt_rel, pred_rel), count in confused:
        lines.append(f"  {gt_rel:25s} → {pred_rel:25s}  {count:4d} times")

    # Whitelist stats
    if use_whitelist:
        total_dropped = sum(
            v for ex in results for v in ex.get("whitelist_stats", {}).values() if "dropped" in str(v)
        )
        # Actually stats are per-example dicts
        agg_stats: Dict[str, int] = defaultdict(int)
        for ex in results:
            for k, v in ex.get("whitelist_stats", {}).items():
                agg_stats[k] += v
        lines.append("\n--- Whitelist filtering ---")
        for k, v in sorted(agg_stats.items()):
            lines.append(f"  {k}: {v}")

    # Distribution comparison
    lines.append("\n--- Distribution comparison ---")
    lines.append("TRAINING DATA:")
    lines.append(f"  Total triplets: {len(train_triplets)}")
    lines.append(f"  Unique relations: {len(train_relation_counts)}")
    if train_relation_counts:
        top_r, top_c = train_relation_counts.most_common(1)[0]
        lines.append(f"  Top relation: {top_r} ({top_c} occurrences)")
    top5_train = train_relation_counts.most_common(5)
    top5_cov_train = sum(c for _, c in top5_train) / len(train_triplets) * 100 if train_triplets else 0
    lines.append(f"  Top 5 relations cover: {top5_cov_train:.1f}%")
    top10_cov_train = sum(c for _, c in train_relation_counts.most_common(10)) / len(train_triplets) * 100 if train_triplets else 0
    lines.append(f"  Top 10 relations cover: {top10_cov_train:.1f}%")

    lines.append("\nMODEL PREDICTIONS:")
    lines.append(f"  Total triplets: {len(all_pred_triplets)}")
    lines.append(f"  Unique relations: {len(pred_relation_counts)}")
    parse_success = sum(1 for ex in pred_examples_triplets if ex)
    lines.append(f"  Parse success rate: {parse_success}/{len(eval_examples)} ({parse_success / len(eval_examples) * 100:.1f}%)")
    if pred_relation_counts:
        top_r, top_c = pred_relation_counts.most_common(1)[0]
        lines.append(f"  Top relation: {top_r}")
        lines.append(f"  Top-relation dominance: {top_c / len(all_pred_triplets) * 100:.1f}%")

    lines.append("\nCOMPARISON:")
    train_top5 = {r for r, _ in train_relation_counts.most_common(5)}
    pred_top5 = {r for r, _ in pred_relation_counts.most_common(5)}
    shared = len(train_top5 & pred_top5)
    lines.append(f"  Shared top-5 relations: {shared}/5")
    top_train_share = (train_relation_counts.most_common(1)[0][1] / len(train_triplets) * 100) if train_relation_counts else 0
    top_pred_share = (pred_relation_counts.most_common(1)[0][1] / len(all_pred_triplets) * 100) if pred_relation_counts else 0
    lines.append(f"  Top-relation dominance  train: {top_train_share:.1f}%")
    lines.append(f"  Top-relation dominance  pred:  {top_pred_share:.1f}%")
    if top_pred_share > top_train_share * 1.5:
        lines.append("  WARNING: Model is heavily biased toward the most common relation!")

    report_text = "\n".join(lines)
    print("\n" + report_text)
    report_filename = f"{split}_report.txt"
    report_path = output_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
