"""
Centralized logging configuration for the Smart Search Fino pipeline.

Usage in any module:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened")

All logs go to both console and pipeline.log (rotating, 10 MB max, 5 backups).
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
_LOG_DIR = _PROJECT_ROOT / "data" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "pipeline.log"
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_CONFIGURED = False


def _setup_root():
    """One-time setup of the root logger with console + rotating file handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, _LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file handler — 10 MB per file, keep 5 backups
    file_handler = RotatingFileHandler(
        str(_LOG_FILE),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the shared handlers."""
    _setup_root()
    return logging.getLogger(name)
