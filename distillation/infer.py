"""
Inference & evaluation script for Wikontic distillation.

Runs the fine-tuned model on validation data, parses predicted triplets,
and produces a comparison report including distribution analysis and
per-relation P/R/F1.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def triplet_key(t: Dict[str, Any]) -> Tuple:
    return (
        t.get("subject", ""),
        t.get("relation", ""),
        t.get("object", ""),
        t.get("subject_type", ""),
        t.get("object_type", ""),
    )


def compute_metrics(gt_triplets: List[Dict], pred_triplets: List[Dict]) -> Dict[str, Any]:
    gt_set = set(triplet_key(t) for t in gt_triplets)
    pred_set = set(triplet_key(t) for t in pred_triplets)
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
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for gt_triplets, pred_triplets in zip(gt_examples, pred_examples):
        gt_by_rel: Dict[str, set] = {}
        for t in gt_triplets:
            r = t.get("relation", "UNKNOWN")
            gt_by_rel.setdefault(r, set()).add(triplet_key(t))
        pred_by_rel: Dict[str, set] = {}
        for t in pred_triplets:
            r = t.get("relation", "UNKNOWN")
            pred_by_rel.setdefault(r, set()).add(triplet_key(t))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer.yaml",
        help="Path to YAML inference config (relative to infer.py dir).",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    script_dir = Path(__file__).resolve().parent
    val_path = script_dir / cfg["val_path"]
    output_dir = script_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = script_dir / cfg["adapter_path"]

    val_examples = load_jsonl(val_path)
    print(f"Loaded {len(val_examples)} validation examples.")

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

    results = []
    pred_examples_triplets: List[List[Dict]] = []
    gt_examples_triplets: List[List[Dict]] = []

    for idx, ex in enumerate(val_examples):
        text = extract_text_from_prompt(ex["messages"][1]["content"])
        prompt_messages = ex["messages"][:2]
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
        pred_triplets = parsed["triplets"] if parsed else []
        gt_triplets = parse_completion_triplets(ex)

        results.append({
            "index": idx,
            "text": text,
            "raw_output": answer,
            "parsed_triplets": pred_triplets,
            "gt_triplets": gt_triplets,
        })
        pred_examples_triplets.append(pred_triplets)
        gt_examples_triplets.append(gt_triplets)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(val_examples)} ...")

    # Save predictions
    preds_path = output_dir / "val_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {preds_path}")

    # Build report
    all_pred_triplets = [t for ex in pred_examples_triplets for t in ex]
    all_gt_triplets = [t for ex in gt_examples_triplets for t in ex]

    pred_relation_counts = Counter(t.get("relation", "UNKNOWN") for t in all_pred_triplets)
    gt_relation_counts = Counter(t.get("relation", "UNKNOWN") for t in all_gt_triplets)

    # Load training distribution for comparison
    train_path = script_dir / cfg.get("train_path", "./data/train.jsonl")
    train_examples = load_jsonl(train_path)
    train_triplets = [t for ex in train_examples for t in parse_completion_triplets(ex)]
    train_relation_counts = Counter(t.get("relation", "UNKNOWN") for t in train_triplets)

    lines = []
    lines.append("=" * 70)
    lines.append("WIKONTIC DISTILLATION INFERENCE REPORT")
    lines.append("=" * 70)

    lines.append("\n--- Overall counts ---")
    lines.append(f"Validation examples: {len(val_examples)}")
    lines.append(f"GT triplets total:   {len(all_gt_triplets)}")
    lines.append(f"Pred triplets total: {len(all_pred_triplets)}")
    empty = sum(1 for ex in pred_examples_triplets if not ex)
    lines.append(f"Empty predictions:   {empty}")

    total_tp = 0
    total_pred = 0
    total_gt = 0
    for gt, pred in zip(gt_examples_triplets, pred_examples_triplets):
        m = compute_metrics(gt, pred)
        total_tp += m["tp"]
        total_pred += m["pred_count"]
        total_gt += m["gt_count"]

    overall_p = total_tp / total_pred if total_pred else 0.0
    overall_r = total_tp / total_gt if total_gt else 0.0
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0
    lines.append("\n--- Overall micro metrics ---")
    lines.append(f"Precision: {overall_p:.3f}")
    lines.append(f"Recall:    {overall_r:.3f}")
    lines.append(f"F1:        {overall_f1:.3f}")

    # Per-relation metrics
    rel_metrics = compute_relation_metrics(gt_examples_triplets, pred_examples_triplets)
    lines.append("\n--- Per-relation metrics (top 15 by GT support) ---")
    sorted_rels = sorted(rel_metrics.items(), key=lambda kv: kv[1]["support_gt"], reverse=True)
    for r, m in sorted_rels[:15]:
        lines.append(
            f"  {r:25s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"gt={m['support_gt']:>4}  pred={m['support_pred']:>4}"
        )

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
    lines.append(f"  Parse success rate: {parse_success}/{len(val_examples)} ({parse_success / len(val_examples) * 100:.1f}%)")
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
    report_path = output_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
