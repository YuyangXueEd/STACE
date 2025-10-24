"""Logging configuration for CAUST system.

Provides structured JSON file logging and Rich-enhanced console output with
correlation ID support. All production code must use this logging framework
instead of print() statements (per coding standards).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pythonjsonlogger import jsonlogger
from rich.console import Console
from rich.logging import RichHandler


_correlation_id: Optional[str] = None


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current execution context."""

    global _correlation_id
    _correlation_id = correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""

    return _correlation_id


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that adds correlation ID and timestamp."""

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        super().add_fields(log_record, record, message_dict)

        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()

        if _correlation_id:
            log_record["correlation_id"] = _correlation_id

        log_record["level"] = record.levelname
        log_record["module"] = record.module
        log_record["function"] = record.funcName


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    console_style: str = "rich",
) -> logging.Logger:
    """Set up logging configuration for CAUST."""

    logger = logging.getLogger()
    logger.setLevel(log_level.upper())
    logger.handlers.clear()

    json_formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level.upper())

        normalized_style = (console_style or "").lower()

        if normalized_style == "json":
            console_handler.setFormatter(json_formatter)
        else:
            if normalized_style in {"rich", "color"}:
                no_color = os.getenv("NO_COLOR") is not None
                console = Console(
                    file=sys.stdout,
                    force_terminal=sys.stdout.isatty() and not no_color,
                    no_color=no_color,
                    highlight=False,
                )
                rich_handler = RichHandler(
                    console=console,
                    show_time=True,
                    show_path=False,
                    markup=True,
                    rich_tracebacks=True,
                    enable_link_path=console.is_terminal and not no_color,
                    log_time_format="%Y-%m-%d %H:%M:%S",
                )
                rich_handler.setFormatter(
                    logging.Formatter("%(name)s:%(funcName)s - %(message)s")
                )
                console_handler = rich_handler
                console_handler.setLevel(log_level.upper())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )

        logger.addHandler(console_handler)

    if enable_file:
        if log_dir is None:
            log_dir = Path("logs")

        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"caust_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level.upper())
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

        logger.info("Logging to file: %s", log_file)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""

    return logging.getLogger(name)


__all__ = [
    "setup_logging",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
]
