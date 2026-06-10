"""Shared facebook/contriever loading with local path and Hugging Face fallbacks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

CONTRIEVER_HUB_ID = "facebook/contriever"
DEFAULT_LOCAL_DIR = "facebook--contriever"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_contriever_path() -> str:
    """Prefer HF_MODEL_PATH, then repo models/, then the Hugging Face hub id."""
    env_path = os.getenv("HF_MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_dir():
            return str(path)

    local_path = repo_root() / "models" / DEFAULT_LOCAL_DIR
    if local_path.is_dir():
        return str(local_path)

    return CONTRIEVER_HUB_ID


def _hf_auth_kwargs() -> dict:
    token = os.getenv("HF_KEY") or os.getenv("HF_TOKEN")
    return {"token": token} if token else {}


def load_contriever(
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    """Load tokenizer and model, trying safetensors first then PyTorch weights."""
    model_path = resolve_contriever_path()
    auth = _hf_auth_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_path, **auth)
    try:
        model = AutoModel.from_pretrained(
            model_path, use_safetensors=True, **auth
        )
    except (OSError, ValueError):
        model = AutoModel.from_pretrained(model_path, **auth)

    if device is None:
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    return tokenizer, model.to(resolved_device), resolved_device
