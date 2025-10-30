"""Logging configuration for CAUST system.

Provides structured JSON file logging and Rich-enhanced console output with
correlation ID support. All production code must use this logging framework
instead of print() statements (per coding standards).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:  # Optional dependency in some environments
    from openai.types.chat.chat_completion import ChatCompletion
except ImportError:  # pragma: no cover
    ChatCompletion = None

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
    camel_filter = _CamelResultFilter()
    logger.addFilter(camel_filter)
    logging.getLogger("camel.base_model").addFilter(camel_filter)

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
            log_dir = Path("./logs")

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


class _CamelResultFilter(logging.Filter):
    """Condense CAMEL model logs to only write assistant content."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - exercised indirectly
        if record.name != "camel.base_model":
            return True

        if not isinstance(record.msg, str) or not record.msg.startswith("Result:"):
            return True

        if not record.args:
            return True

        content = self._extract_content(record.args[0])
        if content is None:
            return True

        record.msg = "Result content: %s"
        record.args = (content,)
        return True

    def _extract_content(self, result: object) -> Optional[str]:
        contents = self._extract_from_chat_completion(result)
        if not contents:
            contents = self._extract_from_dict(result)

        if not contents:
            return None

        if len(contents) == 1:
            return self._stringify(contents[0])

        joined = " | ".join(self._stringify(item) for item in contents)
        return joined or None

    @staticmethod
    def _extract_from_chat_completion(result: object) -> list:
        if ChatCompletion is None or not isinstance(result, ChatCompletion):
            return []

        contents = []
        for choice in getattr(result, "choices", []):
            message = getattr(choice, "message", None)
            if not message:
                continue
            content = getattr(message, "content", None)
            if content is not None:
                contents.append(content)
        return contents

    @staticmethod
    def _extract_from_dict(result: object) -> list:
        if not isinstance(result, dict):
            return []

        choices = result.get("choices")
        if not isinstance(choices, list):
            return []

        contents = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict) and "content" in message:
                contents.append(message["content"])
        return contents

    @staticmethod
    def _stringify(value: object) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)


__all__ = [
    "setup_logging",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
]
