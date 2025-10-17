"""Unit tests for logging configuration module."""

import json
import logging
from pathlib import Path

import pytest

from src.logging_config import (
    CustomJsonFormatter,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
)


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Create temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture(autouse=True)
def reset_correlation_id() -> None:
    """Reset correlation ID before each test."""
    import src.logging_config

    src.logging_config._correlation_id = None


def test_set_correlation_id_sets_value() -> None:
    """Test that set_correlation_id sets the correlation ID."""
    test_id = "test_correlation_123"
    set_correlation_id(test_id)
    assert get_correlation_id() == test_id


def test_get_correlation_id_returns_none_when_not_set() -> None:
    """Test that get_correlation_id returns None when not set."""
    assert get_correlation_id() is None


def test_setup_logging_creates_logger_with_correct_level() -> None:
    """Test that setup_logging creates logger with correct level."""
    logger = setup_logging(log_level="DEBUG", enable_file=False)
    assert logger.level == logging.DEBUG


def test_setup_logging_creates_console_handler() -> None:
    """Test that setup_logging creates console handler."""
    logger = setup_logging(log_level="INFO", enable_console=True, enable_file=False)

    console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(console_handlers) > 0


def test_setup_logging_creates_file_handler(temp_log_dir: Path) -> None:
    """Test that setup_logging creates file handler."""
    logger = setup_logging(
        log_level="INFO", log_dir=temp_log_dir, enable_console=False, enable_file=True
    )

    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) > 0


def test_setup_logging_creates_log_file(temp_log_dir: Path) -> None:
    """Test that setup_logging creates log file."""
    setup_logging(log_level="INFO", log_dir=temp_log_dir, enable_console=False, enable_file=True)

    log_files = list(temp_log_dir.glob("caust_*.log"))
    assert len(log_files) == 1


def test_get_logger_returns_logger_instance() -> None:
    """Test that get_logger returns logger instance."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_custom_json_formatter_adds_timestamp() -> None:
    """Test that CustomJsonFormatter adds timestamp."""
    formatter = CustomJsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    log_dict = json.loads(formatted)

    assert "timestamp" in log_dict
    assert "T" in log_dict["timestamp"]  # ISO8601 format


def test_custom_json_formatter_adds_correlation_id_when_set() -> None:
    """Test that CustomJsonFormatter adds correlation ID when set."""
    set_correlation_id("test_correlation_456")

    formatter = CustomJsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    log_dict = json.loads(formatted)

    assert log_dict["correlation_id"] == "test_correlation_456"


def test_custom_json_formatter_adds_level() -> None:
    """Test that CustomJsonFormatter adds log level."""
    formatter = CustomJsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="test error",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    log_dict = json.loads(formatted)

    assert log_dict["level"] == "ERROR"


def test_custom_json_formatter_adds_module_and_function() -> None:
    """Test that CustomJsonFormatter adds module and function info."""
    formatter = CustomJsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )
    record.module = "test_module"
    record.funcName = "test_function"

    formatted = formatter.format(record)
    log_dict = json.loads(formatted)

    assert log_dict["module"] == "test_module"
    assert log_dict["function"] == "test_function"


def test_logging_writes_to_file(temp_log_dir: Path) -> None:
    """Test that logging writes messages to file."""
    setup_logging(log_level="INFO", log_dir=temp_log_dir, enable_console=False, enable_file=True)

    logger = get_logger("test_logger")
    test_message = "Test log message for file output"
    logger.info(test_message)

    log_files = list(temp_log_dir.glob("caust_*.log"))
    assert len(log_files) == 1

    log_content = log_files[0].read_text()
    assert test_message in log_content
