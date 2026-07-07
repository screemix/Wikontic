"""Shared ai-forever/FRIDA loading with local path and Hugging Face fallbacks.

FRIDA is a T5-encoder-based embedding model for Russian/English text.
It uses CLS pooling (hidden dim = 1536) and task-specific prefixes.

Recommended prefixes
--------------------
- ``"search_query: "``     / ``"search_document: "``  — retrieval
- ``"paraphrase: "``                                   — STS / deduplication
- ``"categorize: "``                                   — title ↔ body matching
- ``"categorize_sentiment: "``                         — sentiment-related tasks
- ``"categorize_topic: "``                             — topic grouping
- ``"categorize_entailment: "``                        — NLI
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

FRIDA_HUB_ID = "ai-forever/FRIDA"
DEFAULT_LOCAL_DIR = "ai-forever--FRIDA"

EMBEDDING_DIM = 1536


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_frida_path() -> str:
    """Prefer HF_FRIDA_MODEL_PATH, then repo models/, then the Hugging Face hub id."""
    env_path = os.getenv("HF_FRIDA_MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_dir():
            return str(path)

    local_path = repo_root() / "models" / DEFAULT_LOCAL_DIR
    if local_path.is_dir():
        return str(local_path)

    return FRIDA_HUB_ID


def _hf_auth_kwargs() -> dict:
    token = os.getenv("HF_KEY") or os.getenv("HF_TOKEN")
    return {"token": token} if token else {}


def load_frida(
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, T5EncoderModel, torch.device]:
    """Load FRIDA tokenizer and T5EncoderModel, trying safetensors first."""
    model_path = resolve_frida_path()
    auth = _hf_auth_kwargs()

    tokenizer = AutoTokenizer.from_pretrained(model_path, **auth)
    try:
        model = T5EncoderModel.from_pretrained(
            model_path, use_safetensors=True, **auth
        )
    except (OSError, ValueError):
        model = T5EncoderModel.from_pretrained(model_path, **auth)

    if device is None:
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    return tokenizer, model.to(resolved_device), resolved_device


def pool(
    hidden_state: torch.Tensor,
    mask: torch.Tensor,
    pooling_method: str = "cls",
) -> torch.Tensor:
    """Pool encoder hidden states into a single sentence vector.

    Args:
        hidden_state: Encoder output of shape (batch, seq_len, hidden_dim).
        mask: Attention mask of shape (batch, seq_len).
        pooling_method: ``"cls"`` (default, recommended for FRIDA) or ``"mean"``.

    Returns:
        Normalized sentence embeddings of shape (batch, hidden_dim).
    """
    if pooling_method == "cls":
        embeddings = hidden_state[:, 0]
    elif pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(dim=1, keepdim=True).float()
        embeddings = s / d
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method!r}")

    return F.normalize(embeddings, p=2, dim=1)
