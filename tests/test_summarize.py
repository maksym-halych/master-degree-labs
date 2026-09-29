"""Tests for the summarization backends, mostly the Claude Code CLI one."""

import json
import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from lecture_transcriber.config import Settings
from lecture_transcriber.models import Segment, Transcript
from lecture_transcriber.summarize import SummaryError, summarize

# One recorded (argv, kwargs) pair per fake subprocess call.
_Calls = list[tuple[list[str], dict[str, Any]]]

_TRANSCRIPT = Transcript(
    language="uk",
    duration=60.0,
    segments=(Segment(start=0.0, end=60.0, text="Бінарний пошук."),),
)


def _settings(**overrides: Any) -> Settings:
    """
    Build Settings for the CLI backend, ignoring any .env on the machine.

    Args:
        **overrides: Field values to set.

    Returns:
        Settings selecting the claude-cli backend.
    """
    return Settings(_env_file=None, summary_backend="claude-cli", **overrides)


def _fake_run(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> tuple[Callable[..., subprocess.CompletedProcess[str]], _Calls]:
    """
    Build a `subprocess.run` replacement that records the call it received.

    Args:
        stdout: Standard output the fake process should return.
        returncode: Exit status the fake process should report.
        stderr: Standard error the fake process should return.

    Returns:
        The replacement callable, and the list its calls accumulate in.
    """
    calls: _Calls = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    return run, calls


def _envelope(**fields: Any) -> str:
    """
    Render a `--output-format json` envelope.

    Args:
        **fields: Envelope keys to set, overriding the success defaults.

    Returns:
        The envelope as JSON text.
    """
    return json.dumps({"type": "result", "is_error": False, **fields})


def test_summary_is_read_from_the_json_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backend returns the envelope's result, not the raw stdout."""
    run, _ = _fake_run(_envelope(result="# Лекція\n## TL;DR\nПро пошук.\n"))
    monkeypatch.setattr(subprocess, "run", run)

    assert summarize(_TRANSCRIPT, "lecture.mp4", _settings()).startswith("# Лекція")


def test_transcript_goes_on_stdin_not_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux caps one argument at 128 KiB; a 90-minute lecture is near 100 KB."""
    run, calls = _fake_run(_envelope(result="summary"))
    monkeypatch.setattr(subprocess, "run", run)

    summarize(_TRANSCRIPT, "lecture.mp4", _settings())
    argv, kwargs = calls[0]

    assert "Бінарний пошук." in kwargs["input"]
    assert not any("Бінарний пошук." in arg for arg in argv)


def test_run_is_isolated_from_the_repo_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLAUDE.md is about writing Python and must not reach the summarizer."""
    run, calls = _fake_run(_envelope(result="summary"))
    monkeypatch.setattr(subprocess, "run", run)

    summarize(_TRANSCRIPT, "lecture.mp4", _settings())
    argv = calls[0][0]

    assert "--safe-mode" in argv
    # --bare would also strip CLAUDE.md, but reads credentials strictly from
    # ANTHROPIC_API_KEY — never the login this backend exists to reuse.
    assert "--bare" not in argv
    # The summarizer prompt must replace Claude Code's own, not append to it.
    assert "--system-prompt" in argv
    assert "--append-system-prompt" not in argv


def test_configured_binary_is_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-PATH install must be reachable without editing the source."""
    run, calls = _fake_run(_envelope(result="summary"))
    monkeypatch.setattr(subprocess, "run", run)

    summarize(_TRANSCRIPT, "lecture.mp4", _settings(claude_bin="/opt/claude"))

    assert calls[0][0][0] == "/opt/claude"


def test_expired_login_is_reported_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real failure mode: exit 1, empty stderr, reason only in the envelope."""
    reason = "Failed to authenticate: OAuth session expired and could not be refreshed"
    run, _ = _fake_run(
        _envelope(is_error=True, subtype="success", result=reason), returncode=1
    )
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(SummaryError, match="OAuth session expired"):
        summarize(_TRANSCRIPT, "lecture.mp4", _settings())


def test_error_envelope_on_a_zero_exit_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error must never be written to docs/ as though it were a summary."""
    run, _ = _fake_run(_envelope(is_error=True, result="Overloaded"))
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(SummaryError, match="Overloaded"):
        summarize(_TRANSCRIPT, "lecture.mp4", _settings())


def test_unparseable_stdout_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CLI upgrade that changes the output shape must fail loudly."""
    monkeypatch.setattr(subprocess, "run", _fake_run("not json at all")[0])

    with pytest.raises(SummaryError, match="no JSON envelope"):
        summarize(_TRANSCRIPT, "lecture.mp4", _settings())


def test_empty_result_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank summary would overwrite the cached artifact with nothing."""
    monkeypatch.setattr(subprocess, "run", _fake_run(_envelope(result="   "))[0])

    with pytest.raises(SummaryError, match="empty summary"):
        summarize(_TRANSCRIPT, "lecture.mp4", _settings())


def test_missing_binary_points_at_the_other_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Someone cloning the repo without Claude Code needs to know the way out."""

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(SummaryError, match="SUMMARY_BACKEND=api"):
        summarize(_TRANSCRIPT, "lecture.mp4", _settings())


def test_timeout_is_reported_as_a_summary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged subprocess must surface as an actionable error, not a traceback."""

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 900.0)

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(SummaryError, match="did not finish"):
        summarize(_TRANSCRIPT, "lecture.mp4", _settings())


def test_api_backend_without_credentials_points_at_the_other_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK signals this with a bare TypeError; the pipeline must not leak it."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, summary_backend="api", anthropic_api_key="")

    with pytest.raises(SummaryError, match="SUMMARY_BACKEND=claude-cli"):
        summarize(_TRANSCRIPT, "lecture.mp4", settings)
