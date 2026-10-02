"""Tests for the startup settings report."""

from lecture_transcriber.config import Settings


def _settings(**overrides: object) -> Settings:
    """
    Build Settings from explicit values, ignoring any .env on the machine.

    Args:
        **overrides: Field values to set.

    Returns:
        Settings carrying exactly the given overrides over the defaults.
    """
    return Settings(_env_file=None, **overrides)


def test_describe_never_echoes_secrets() -> None:
    """Credentials are reported as set/unset — these lines land in CI logs."""
    secret = "sk-ant-do-not-log-me"
    described = _settings(anthropic_api_key=secret, parakeet_api_key=secret).describe()

    assert secret not in "\n".join(described.values())
    assert described["anthropic_api_key"] == "<set>"
    assert described["parakeet_api_key"] == "<set>"


def test_describe_distinguishes_unset_secrets() -> None:
    """An empty key must read as unset, not as a configured credential."""
    described = _settings(anthropic_api_key="", parakeet_api_key="   ").describe()

    assert described["anthropic_api_key"] == "<unset>"
    assert described["parakeet_api_key"] == "<unset>"


def test_describe_omits_file_id() -> None:
    """The CLI argument can override file_id, so the caller logs the resolved one."""
    assert "file_id" not in _settings(file_id="from-dotenv").describe()


def test_describe_reports_non_secret_values() -> None:
    """Everything that changes pipeline behaviour is visible at a glance."""
    described = _settings(
        anthropic_model="claude-sonnet-5", language="en", chunk_seconds=600.0
    ).describe()

    assert described["anthropic_model"] == "claude-sonnet-5"
    assert described["language"] == "en"
    assert described["chunk_seconds"] == "600"
