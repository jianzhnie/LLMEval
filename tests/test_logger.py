"""Tests for llmeval.utils.logger."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from llmeval.utils.logger import get_logger, init_logger, set_log_level


class TestInitLogger:
    def test_returns_logger_with_expected_name(self) -> None:
        logger = init_logger("test_basic")
        assert logger.name == "test_basic"

    def test_console_handler_present_by_default(self) -> None:
        logger = init_logger("test_console")
        assert any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        )

    def test_file_handler_created(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        logger = init_logger("test_file", log_file=str(log_file))
        assert log_file.exists()
        assert any(
            isinstance(h, logging.FileHandler) for h in logger.handlers
        )

    def test_string_level_resolved(self) -> None:
        logger = init_logger("test_level", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers_on_reinit(self) -> None:
        name = "test_no_dup"
        l1 = init_logger(name)
        count1 = len(l1.handlers)
        l2 = init_logger(name)
        assert len(l2.handlers) == count1

    def test_propagate_default_false(self) -> None:
        logger = init_logger("test_propagate")
        assert logger.propagate is False


class TestGetLogger:
    def test_returns_same_logger(self) -> None:
        init_logger("test_get")
        assert get_logger("test_get").name == "test_get"


class TestSetLogLevel:
    def test_updates_logger_and_handlers(self) -> None:
        logger = init_logger("test_setlevel")
        set_log_level(logger, "WARNING")
        assert logger.level == logging.WARNING
        for h in logger.handlers:
            assert h.level == logging.WARNING
