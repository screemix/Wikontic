"""Unified embedding-model loader supporting Contriever and FRIDA.

Usage
-----
    from wikontic.utils.embedding_model import load_embedding_model, get_embedding, EMBEDDING_DIMS

    tokenizer, model, device = load_embedding_model("frida")
    vec = get_embedding("some text", tokenizer, model, device, "frida")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from wikontic.utils.contriever_model import load_contriever
from wikontic.utils.frida_model import load_frida

SUPPORTED_MODELS: frozenset = frozenset({"contriever", "frida"})

# Output vector dimensionality for each model.
EMBEDDING_DIMS: dict = {"contriever": 768, "frida": 1536}

# Prefix prepended to every text when using FRIDA.
# "paraphrase:" is the symmetric-similarity prefix — appropriate for
# entity / property name matching (query and document share the same space).
_FRIDA_PREFIX = "paraphrase: "


def load_embedding_model(
    model_name: str,
    device: Optional[str] = None,
) -> Tuple:
    """Load tokenizer, model, and resolved torch.device for *model_name*.

    Args:
        model_name: One of ``"contriever"`` or ``"frida"``.
        device: Optional device string (``"cuda"``, ``"cpu"``).
                Defaults to CUDA when available.

    Returns:
        ``(tokenizer, model, device)`` — same shape as the individual
        ``load_contriever`` / ``load_frida`` helpers.
    """
    if model_name == "contriever":
        return load_contriever(device=device)
    elif model_name == "frida":
        return load_frida(device=device)
    else:
        raise ValueError(
            f"Unknown embedding model {model_name!r}. "
            f"Supported values: {sorted(SUPPORTED_MODELS)}"
        )


def get_embeddings(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    model_name: str,
) -> List[Optional[List[float]]]:
    """Compute sentence embedding vectors for a batch of texts in one forward pass.

    Invalid entries (empty / non-string) are skipped in the actual model call
    and their corresponding output slot is ``None``, so the returned list is
    always the same length as *texts* and index-aligned with it.

    Pooling and normalisation are chosen automatically based on *model_name*:

    * ``"contriever"`` — mean pooling over non-padding tokens (no L2 norm).
    * ``"frida"`` — CLS-token pooling with L2 normalisation;
      text is prefixed with ``"paraphrase: "`` before encoding.

    Args:
        texts: Input texts.
        tokenizer: HuggingFace tokenizer returned by :func:`load_embedding_model`.
        model: HuggingFace model returned by :func:`load_embedding_model`.
        device: torch.device the model lives on.
        model_name: ``"contriever"`` or ``"frida"``.

    Returns:
        List of the same length as *texts*; each element is a python list of
        floats (length == ``EMBEDDING_DIMS[model_name]``), or ``None`` where
        the corresponding input text was invalid.
    """
    valid_indices = [
        i for i, text in enumerate(texts) if text and isinstance(text, str)
    ]
    results: List[Optional[List[float]]] = [None] * len(texts)
    if not valid_indices:
        return results

    valid_texts = [texts[i] for i in valid_indices]
    if model_name == "frida":
        valid_texts = [_FRIDA_PREFIX + text for text in valid_texts]

    inputs = tokenizer(
        valid_texts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    # T5EncoderModel exposes last_hidden_state; AutoModel returns it as outputs[0]
    hidden = (
        outputs.last_hidden_state
        if hasattr(outputs, "last_hidden_state")
        else outputs[0]
    )
    mask = inputs["attention_mask"]

    if model_name == "frida":
        embeddings = F.normalize(hidden[:, 0], p=2, dim=1)
    else:
        # Mean pooling: zero-out padding then average
        hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
        denom = mask.sum(dim=1).clamp(min=1)[..., None]
        embeddings = hidden.sum(dim=1) / denom

    embeddings_list = embeddings.detach().cpu().tolist()
    for i, embedding in zip(valid_indices, embeddings_list):
        results[i] = embedding

    return results


def get_embedding(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    model_name: str,
) -> Optional[List[float]]:
    """Compute a single sentence embedding vector. See :func:`get_embeddings`."""
    return get_embeddings([text], tokenizer, model, device, model_name)[0]
