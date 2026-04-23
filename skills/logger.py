"""
skills/logger.py
────────────────
Shared structured logger for the RAG pipeline.
Produces colour-coded console output and a rotating file log.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag_pipeline.log"

_COLOURS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

_FMT_CONSOLE = "%(log_color)s%(levelname)-8s%(reset)s | %(name)-28s | %(message)s"
_FMT_FILE    = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"


def _make_console_handler() -> logging.Handler:
    h = logging.StreamHandler(sys.stdout)
    if _HAS_COLORLOG:
        h.setFormatter(colorlog.ColoredFormatter(
            _FMT_CONSOLE, datefmt=_DATE_FMT, log_colors=_COLOURS,
        ))
    else:
        h.setFormatter(logging.Formatter(
            "%(levelname)-8s | %(name)-28s | %(message)s", datefmt=_DATE_FMT,
        ))
    return h


def _make_file_handler() -> logging.Handler:
    h = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    h.setFormatter(logging.Formatter(_FMT_FILE, datefmt=_DATE_FMT))
    return h


def _configure_root() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(_make_console_handler())
    root.addHandler(_make_file_handler())
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "langchain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class StepLogger:
    def __init__(self, stage: str):
        self._log = get_logger(f"pipeline.{stage}")
        self._stage = stage

    def start(self, msg: str, **kw) -> None:
        self._log.info(f">>  {msg}", extra=kw)

    def step(self, msg: str, **kw) -> None:
        self._log.info(f"   |-- {msg}", extra=kw)

    def done(self, msg: str, **kw) -> None:
        self._log.info(f"   OK  {msg}", extra=kw)

    def warn(self, msg: str, **kw) -> None:
        self._log.warning(f"   !!  {msg}", extra=kw)

    def error(self, msg: str, **kw) -> None:
        self._log.error(f"   XX  {msg}", extra=kw)
