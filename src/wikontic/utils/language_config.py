"""Language settings for prompts, text normalization, and ontology mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

SUPPORTED_LANGUAGES = frozenset({"en", "ru"})
DEFAULT_LANGUAGE = "en"
_UTILS_DIR = Path(__file__).resolve().parent


def normalize_language(language: Optional[str]) -> str:
    lang = (language or DEFAULT_LANGUAGE).strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. Supported values: {sorted(SUPPORTED_LANGUAGES)}"
        )
    return lang


def prompt_folder_for_language(language: Optional[str]) -> Path:
    lang = normalize_language(language)
    folder_name = "prompts_ru" if lang == "ru" else "prompts"
    return _UTILS_DIR / folder_name


def use_unidecode_for_language(language: Optional[str]) -> bool:
    """Transliterate entity names for English; keep original script for Russian."""
    return normalize_language(language) == "en"


def default_embedding_model_for_language(language: Optional[str]) -> str:
    """Return the recommended embedding model for *language*.

    * ``"ru"``  → ``"frida"``  (Russian-first T5-encoder model)
    * ``"en"``  → ``"contriever"``
    """
    lang = normalize_language(language)
    return "frida" if lang == "ru" else "contriever"


def ontology_mappings_dir_for_language(
    language: Optional[str], fallback_language: str = "en"
) -> Path:
    lang = normalize_language(language)
    fallback = normalize_language(fallback_language)
    return _UTILS_DIR / f"ontology_mappings_{lang}_{fallback}"
