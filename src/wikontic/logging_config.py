"""Central logging configuration for Wikontic."""

from __future__ import annotations

import logging
import os
from typing import Optional

ENV_VAR = "WIKONTIC_LOG_LEVEL"
DEFAULT_LEVEL_NAME = "INFO"
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_configured = False


def resolve_log_level_name(level_name: Optional[str] = None) -> str:
    name = (level_name or os.getenv(ENV_VAR) or DEFAULT_LEVEL_NAME).upper()
    if name not in _VALID_LEVELS:
        return DEFAULT_LEVEL_NAME
    return name


def get_log_level(level_name: Optional[str] = None) -> int:
    return getattr(logging, resolve_log_level_name(level_name))


def configure_logging(level_name: Optional[str] = None, *, force: bool = False) -> None:
    """Apply process-wide logging once (idempotent unless force=True)."""
    global _configured
    if _configured and not force:
        return

    level = get_log_level(level_name)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(
        logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    )
    logging.getLogger("httpcore").setLevel(
        logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
