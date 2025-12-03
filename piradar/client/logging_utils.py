"""
Shared logging helpers for client CLI scripts.
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging using a human-friendly format."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

