"""Tests for run-log setup."""

import logging
from pathlib import Path

import pytest

from lecture_transcriber.__main__ import _configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Undo the global handler changes `_configure_logging` makes."""
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    yield
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)
    for handler in saved[0]:
        root.addHandler(handler)
    root.setLevel(saved[1])


def _log_text(logs_dir: Path) -> str:
    """
    Read the single log file written into `logs_dir`.

    Args:
        logs_dir: Directory passed to `_configure_logging`.

    Returns:
        The log file's contents.
    """
    files = list(logs_dir.iterdir())
    assert len(files) == 1, f"expected one log file, found {files}"
    return files[0].read_text(encoding="utf-8")


def test_creates_logs_dir_and_writes_records(tmp_path: Path) -> None:
    """The directory is created on demand and receives the run's records."""
    logs_dir = tmp_path / "logs"
    _configure_logging(logs_dir, verbose=False)
    logging.getLogger("lecture_transcriber.test").info("hello")

    assert logs_dir.is_dir()
    assert "hello" in _log_text(logs_dir)


def test_log_records_carry_the_date(tmp_path: Path) -> None:
    """Timestamps are full dates — a run log outlives the day it was written."""
    logs_dir = tmp_path / "logs"
    _configure_logging(logs_dir, verbose=False)
    logging.getLogger("lecture_transcriber.test").info("dated")

    line = next(l for l in _log_text(logs_dir).splitlines() if "dated" in l)
    assert line[:10].count("-") == 2, f"no ISO date at start of {line!r}"


def test_file_keeps_debug_even_when_console_is_quiet(tmp_path: Path) -> None:
    """Detail must survive a long run nobody thought to pass -v to."""
    logs_dir = tmp_path / "logs"
    _configure_logging(logs_dir, verbose=False)
    logging.getLogger("lecture_transcriber.test").debug("quiet-detail")

    assert "quiet-detail" in _log_text(logs_dir)


def test_http_stacks_are_capped_below_debug(tmp_path: Path) -> None:
    """Their DEBUG records dump request headers, which carry bearer tokens."""
    _configure_logging(tmp_path / "logs", verbose=True)

    for noisy in ("urllib3", "httpx", "httpcore", "anthropic", "googleapiclient"):
        assert logging.getLogger(noisy).level == logging.WARNING


def test_cap_reaches_the_logger_names_the_installed_clients_actually_use(
    tmp_path: Path,
) -> None:
    """
    The cap once listed `httpx`/`httpcore` while the installed packages logged
    under `httpx2`/`httpcore2`, so it matched nothing and 50 lines of HTTP DEBUG
    reached the file every run. Import the clients first, then assert against the
    logger names that exist — a rename in either package fails this test.
    """
    import anthropic  # noqa: F401
    import googleapiclient.discovery  # noqa: F401
    import httpx2  # noqa: F401
    import openai  # noqa: F401

    _configure_logging(tmp_path / "logs", verbose=True)

    prefixes = ("httpx", "httpcore", "openai", "anthropic", "urllib3", "google")
    capped = {
        name: logging.getLogger(name).getEffectiveLevel()
        for name in logging.root.manager.loggerDict
        if name.startswith(prefixes)
    }
    assert capped, "no HTTP-stack loggers registered; the import guard above broke"

    leaking = {n: lvl for n, lvl in capped.items() if lvl < logging.WARNING}
    assert not leaking, f"HTTP loggers below WARNING: {sorted(leaking)}"
