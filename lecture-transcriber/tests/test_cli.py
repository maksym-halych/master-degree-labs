"""Tests for how the CLI decides which recording to process."""

import click
import pytest

from lecture_transcriber.__main__ import resolve_file_id


def test_configured_id_is_used_when_no_argument_given() -> None:
    """The usual workflow: FILE_ID in .env, `make transcribe-lecture` takes no argument."""
    assert resolve_file_id(None, "1fromEnv") == "1fromEnv"


def test_argument_overrides_the_configured_id() -> None:
    """A one-off run must not require editing .env."""
    assert resolve_file_id("1fromArg", "1fromEnv") == "1fromArg"


def test_argument_works_with_no_configured_id() -> None:
    """The CLI stays usable before .env is filled in."""
    assert resolve_file_id("1fromArg", "") == "1fromArg"


@pytest.mark.parametrize("configured", ["", "   ", "\n"])
def test_missing_id_is_a_usage_error(configured: str) -> None:
    """An unset or whitespace-only FILE_ID must fail loudly, not run on an empty ID."""
    with pytest.raises(click.UsageError, match="no recording specified"):
        resolve_file_id(None, configured)


def test_surrounding_whitespace_is_stripped() -> None:
    """A stray newline from .env would otherwise become part of the Drive request."""
    assert resolve_file_id(None, " 1fromEnv\n") == "1fromEnv"
