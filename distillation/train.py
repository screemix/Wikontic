"""
Training script for Wikontic distillation.

Key features:
  1. Curriculum scheduler on the number of triplets per example.
     Triplets are reordered so "instance of" comes first, then the list is
     truncated to a stage-dependent limit.
  2. Per-token loss down-weighting for tokens that literally spell
     "instance of" (configurable coefficient < 1).
  3. Standard Trainer (instead of SFTTrainer) so we have full control over
     the collator and compute_loss.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def build_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_triplets: Optional[int] = None,
) -> Dataset:
    """
    Build a tokenized dataset from chat-formatted examples.

    Triplets are reordered so "instance of" comes first.
    If ``max_triplets`` is set, only the first N triplets are kept.
    """
    processed = []
    for ex in examples:
        messages = ex["messages"]
        assistant = messages[2]["content"]
        try:
            parsed = json.loads(assistant)
        except json.JSONDecodeError:
            continue

        triplets = parsed.get("triplets", [])
        # 1) Sort: all "instance of" triplets come first.
        triplets.sort(key=lambda t: 0 if t.get("relation") == "instance of" else 1)
        # 2) Apply curriculum limit.
        if max_triplets is not None:
            triplets = triplets[:max_triplets]

        new_assistant = json.dumps({"triplets": triplets}, ensure_ascii=False)
        new_messages = messages[:2] + [{"role": "assistant", "content": new_assistant}]

        prompt_text = tokenizer.apply_chat_template(
            new_messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            new_messages, tokenize=False, add_generation_prompt=False
        )

        # Guarantee EOS at the end so the model learns to stop.
        if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token

        # Tokenize without adding special tokens again (chat template already did).
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        processed.append({"input_ids": full_ids, "prompt_len": len(prompt_ids)})

    return Dataset.from_list(processed)


class PromptMaskingCollator:
    """
    Collator for pre-tokenized prompt/completion data.

    Masks prompt tokens with -100 so that loss is computed only on the
    assistant completion.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len // self.pad_to_multiple_of) + 1
            ) * self.pad_to_multiple_of

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        for f in features:
            ids = f["input_ids"]
            prompt_len = f["prompt_len"]
            # Mask prompt; keep completion token IDs as labels.
            labels = [-100] * prompt_len + ids[prompt_len:]

            pad_len = max_len - len(ids)
            ids = ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len

            batch_input_ids.append(ids)
            batch_labels.append(labels)
            batch_attention_mask.append([1 if tid != pad_id else 0 for tid in ids])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    """
    Custom Trainer with two modifications:
      - Curriculum-aware dataset switching at epoch boundaries.
      - Down-weighted CE loss for the literal token sequence "instance of".
    """

    def __init__(
        self,
        instance_of_weight: float = 1.0,
        curriculum_datasets: Optional[List[Dataset]] = None,
        curriculum_switch_steps: Optional[List[int]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.instance_of_weight = instance_of_weight
        # Precompute token IDs for the literal string "instance of".
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        ids = tok.encode("instance of", add_special_tokens=False)
        self.instance_of_token_ids = torch.tensor(ids, dtype=torch.long)
        self.curriculum_datasets = curriculum_datasets or []
        self.curriculum_switch_steps = curriculum_switch_steps or []

    def get_train_dataloader(self):
        """Select the appropriate curriculum dataset before each epoch."""
        if (
            self.curriculum_datasets
            and hasattr(self, "state")
            and self.state is not None
        ):
            stage = 0
            for limit in self.curriculum_switch_steps:
                if self.state.global_step < limit:
                    break
                stage += 1
            idx = min(stage, len(self.curriculum_datasets) - 1)
            target = self.curriculum_datasets[idx]
            if self.train_dataset is not target:
                meta = getattr(self, "_curriculum_stages_meta", [])
                info = meta[idx] if idx < len(meta) else "?"
                print(
                    f"[Curriculum] Step {self.state.global_step}: "
                    f"switched to stage {idx} (max_triplets={info})"
                )
                self.train_dataset = target
        return super().get_train_dataloader()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Forward pass *without* passing labels; we compute loss manually.
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits
        labels = inputs["labels"]

        # Next-token prediction shift.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        losses = losses.view(shift_labels.size())

        # Mask prompt & padding positions.
        valid_mask = (shift_labels != -100).to(losses.dtype)

        # ------------------------------------------------------------------
        # Apply instance_of down-weighting only while training.
        # ------------------------------------------------------------------
        if (
            model.training
            and self.instance_of_weight != 1.0
            and self.instance_of_token_ids.numel() > 0
        ):
            weights = torch.ones_like(losses)
            input_ids = inputs["input_ids"]
            target_ids = self.instance_of_token_ids.to(input_ids.device)
            L = target_ids.numel()

            for b in range(input_ids.size(0)):
                ids = input_ids[b]
                seq_len = ids.size(0)
                for pos in range(seq_len - L + 1):
                    if torch.equal(ids[pos : pos + L], target_ids):
                        # The model predicts these tokens at shifted positions
                        # [pos-1, pos+L-1).
                        start = pos - 1
                        end = pos + L - 1
                        if start >= 0 and end <= weights.size(1):
                            weights[b, start:end] = self.instance_of_weight
            losses = losses * weights

        loss = (losses * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML training config (relative to train.py dir).",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    # Resolve paths relative to train.py's directory.
    script_dir = Path(__file__).resolve().parent
    train_path = script_dir / cfg["train_path"]
    val_path = script_dir / cfg["val_path"]
    output_dir = script_dir / cfg["output_dir"]

    # Load data.
    train_examples = load_jsonl(train_path)
    val_examples = load_jsonl(val_path)
    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples.")

    # Tokenizer.
    base_model = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Build curriculum datasets (sorted so "instance of" comes first).
    # ------------------------------------------------------------------
    curriculum_cfg = cfg.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    stages = curriculum_cfg.get("stages", [])

    curriculum_datasets: List[Dataset] = []
    curriculum_switch_steps: List[int] = []
    curriculum_meta: List[str] = []

    if curriculum_enabled and stages:
        for stage in stages:
            limit = stage["max_triplets"]
            ds = build_dataset(train_examples, tokenizer, max_triplets=limit)
            curriculum_datasets.append(ds)
            curriculum_switch_steps.append(stage["until_step"])
            curriculum_meta.append(str(limit))
        # Append a final unlimited stage so training always finishes on full data.
        full_ds = build_dataset(train_examples, tokenizer, max_triplets=None)
        curriculum_datasets.append(full_ds)
        curriculum_meta.append("full")
    else:
        full_ds = build_dataset(train_examples, tokenizer, max_triplets=None)
        curriculum_datasets = [full_ds]
        curriculum_meta = ["full"]

    eval_dataset = build_dataset(val_examples, tokenizer, max_triplets=None)

    # ------------------------------------------------------------------
    # Model & QLoRA.
    # ------------------------------------------------------------------
    bnb_config = None
    if cfg.get("load_in_4bit", False):
        compute_dtype = getattr(
            torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print(f"Loading base model {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16"))
        if not bnb_config
        else None,
    )
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Training arguments.
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg.get("warmup_steps", 0),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 500),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 500),
        bf16=cfg.get("bf16", False),
        remove_unused_columns=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        report_to=cfg.get("report_to", "none"),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=False,
    )

    collator = PromptMaskingCollator(
        tokenizer, pad_to_multiple_of=cfg.get("pad_to_multiple_of", 8)
    )

    instance_of_weight = cfg.get("instance_of_weight", 1.0)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=curriculum_datasets[0],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        instance_of_weight=instance_of_weight,
        curriculum_datasets=curriculum_datasets,
        curriculum_switch_steps=curriculum_switch_steps,
    )
    trainer._curriculum_stages_meta = curriculum_meta

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
