"""
Training script for Wikontic distillation — v2.

Key changes vs v1:
  - Uses trl.SFTTrainer with prompt/completion columns and
    completion_only_loss=True (trl 1.1.0+ compatible).
  - instance_of loss down-weighting applied ONLY to completion tokens
    (via completion_mask), fixing the prompt-contamination bug in v1.
  - Softened curriculum: single short stage (max_triplets=8) before
    jumping to full examples.
  - All datasets (including curriculum stages) are pre-tokenized so
    swapping works seamlessly.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Apply TRL UTF-8 fix before importing trl (required on Windows)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trl_utf8_fix  # noqa: E402, F401

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    Build a **pre-tokenized** dataset from chat-formatted examples.

    Triplets are reordered so all "instance of" triplets come first, then the
    list is optionally truncated to *max_triplets*.

    Returns a dataset with columns:
      - ``input_ids``: token IDs for prompt + completion
      - ``completion_mask``: 0 for prompt tokens, 1 for completion tokens
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
        # Sort: all "instance of" triplets first.
        triplets.sort(key=lambda t: 0 if t.get("relation") == "instance of" else 1)
        if max_triplets is not None:
            triplets = triplets[:max_triplets]

        new_assistant = json.dumps({"triplets": triplets}, ensure_ascii=False)
        new_messages = messages[:2] + [{"role": "assistant", "content": new_assistant}]

        # Prompt includes the generation header so the model starts emitting.
        prompt_text = tokenizer.apply_chat_template(
            new_messages[:-1], tokenize=False, add_generation_prompt=True
        )
        # Full conversation without the trailing generation prompt.
        full_text = tokenizer.apply_chat_template(
            new_messages, tokenize=False, add_generation_prompt=False
        )
        completion_text = full_text[len(prompt_text):]
        if tokenizer.eos_token and not completion_text.endswith(tokenizer.eos_token):
            completion_text += tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        # Guarantee EOS token ID at the end.
        if tokenizer.eos_token_id is not None and (
            not full_ids or full_ids[-1] != tokenizer.eos_token_id
        ):
            full_ids.append(tokenizer.eos_token_id)

        completion_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))

        processed.append({
            "input_ids": full_ids,
            "completion_mask": completion_mask,
        })

    return Dataset.from_list(processed)


# ---------------------------------------------------------------------------
# Custom SFTTrainer with instance_of loss down-weighting
# ---------------------------------------------------------------------------

class WeightedSFTTrainer(SFTTrainer):
    """
    SFTTrainer subclass that:
      1. Down-weights per-token CE loss for the literal string "instance of",
         but ONLY for completion tokens (completion_mask == 1).
      2. Switches the training dataset according to a curriculum schedule at
         every epoch boundary.
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
        tok = (
            getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
        )
        ids = tok.encode("instance of", add_special_tokens=False)
        self.instance_of_token_ids = torch.tensor(ids, dtype=torch.long)
        self.curriculum_datasets = curriculum_datasets or []
        self.curriculum_switch_steps = curriculum_switch_steps or []

    # ------------------------------------------------------------------
    # Curriculum: swap dataset at epoch boundaries
    # ------------------------------------------------------------------

    def get_train_dataloader(self):
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

    # ------------------------------------------------------------------
    # Loss: completion-only instance_of down-weighting
    # ------------------------------------------------------------------

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        # completion_mask is 1 for completion tokens, 0 for prompt/padding.
        # Pop it so the base Trainer doesn't see an unexpected key.
        completion_mask = inputs.pop("completion_mask", None)

        # Forward without labels — we compute loss manually.
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits

        # Standard next-token shift.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        losses = losses.view(shift_labels.size())

        # valid_mask: 1 for completion tokens, 0 for prompt / padding.
        valid_mask = (shift_labels != -100).to(losses.dtype)

        # ----------------------------------------------------------------
        # instance_of down-weighting — completion tokens only.
        # We search for the token sequence in shift_labels.  Any position
        # where shift_labels[pos] == -100 is skipped (prompt/padding).
        # ----------------------------------------------------------------
        if (
            model.training
            and self.instance_of_weight != 1.0
            and self.instance_of_token_ids.numel() > 0
        ):
            weights = torch.ones_like(losses)
            target_ids = self.instance_of_token_ids.to(shift_labels.device)
            L = target_ids.numel()

            for b in range(shift_labels.size(0)):
                seq = shift_labels[b]
                seq_len = seq.size(0)
                for pos in range(seq_len - L + 1):
                    chunk = seq[pos : pos + L]
                    # Skip if this range includes any masked (prompt) tokens.
                    if (chunk == -100).any():
                        continue
                    if torch.equal(chunk, target_ids):
                        weights[b, pos : pos + L] = self.instance_of_weight

            losses = losses * weights

        loss = (losses * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    script_dir = Path(__file__).resolve().parent
    train_path = script_dir / cfg["train_path"]
    val_path = script_dir / cfg["val_path"]
    output_dir = script_dir / cfg["output_dir"]

    train_examples = load_jsonl(train_path)
    val_examples = load_jsonl(val_path)
    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples.")

    # Tokenizer
    base_model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Curriculum datasets (all pre-tokenized with input_ids / completion_mask)
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
        # Always end on full data.
        full_ds = build_dataset(train_examples, tokenizer, max_triplets=None)
        curriculum_datasets.append(full_ds)
        curriculum_meta.append("full")
    else:
        full_ds = build_dataset(train_examples, tokenizer, max_triplets=None)
        curriculum_datasets = [full_ds]
        curriculum_meta = ["full"]

    eval_dataset = build_dataset(val_examples, tokenizer, max_triplets=None)

    print(f"Train dataset columns: {curriculum_datasets[0].column_names}")
    print(f"Eval  dataset columns: {eval_dataset.column_names}")

    # ------------------------------------------------------------------
    # Model + QLoRA
    # ------------------------------------------------------------------
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
    # SFTConfig
    # ------------------------------------------------------------------
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg.get("warmup_steps", 0),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 1000),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 1000),
        bf16=cfg.get("bf16", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        report_to=cfg.get("report_to", "none"),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=True,
        max_length=cfg.get("max_seq_length", 2048),
        packing=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    instance_of_weight = cfg.get("instance_of_weight", 1.0)

    trainer = WeightedSFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=curriculum_datasets[0],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
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
