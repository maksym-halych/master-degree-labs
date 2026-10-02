"""Runtime configuration, loaded from the environment and `.env`."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _secret_state(value: str) -> str:
    """
    Describe a credential without disclosing it.

    Args:
        value: The raw secret, possibly empty.

    Returns:
        A placeholder indicating only whether the secret was supplied.
    """
    return "<set>" if value.strip() else "<unset>"


class Settings(BaseSettings):
    """Process-wide settings for the transcription pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The recording to process. Set in .env for the usual one-lecture-at-a-time
    # workflow; the CLI argument overrides it for a one-off.
    file_id: str = Field(default="")

    # Everything derived or downloaded lives inside the project, not $HOME, so a
    # `rm -rf .cache/` fully resets the tool. Dotted to match .gitignore: this
    # holds gigabyte media and the service account key, none of it committable.
    cache_dir: Path = Field(default=Path(".cache"))

    # Committed text artifacts, mirroring the Drive folder hierarchy.
    docs_dir: Path = Field(default=Path("docs"))

    # One file per run, kept outside .cache/ so clearing the cache to force a
    # re-transcription never destroys the record of why that was necessary.
    logs_dir: Path = Field(default=Path("logs"))

    google_service_account: Path = Field(default=Path(".cache/service-account.json"))

    parakeet_base_url: str = Field(default="http://127.0.0.1:5092/v1")
    parakeet_api_key: str = Field(default="not-needed")

    # Who pays for summarization. "api" calls the Messages API and bills an
    # Anthropic API organization. "claude-cli" shells out to the Claude Code
    # executable, reusing whatever interactive login it already holds — which for
    # a Pro/Max account means the subscription rather than API credits. A Literal
    # so a typo fails at startup instead of silently falling back to "api".
    summary_backend: Literal["api", "claude-cli"] = Field(default="api")

    # Resolved on PATH unless given an absolute path. Only the "claude-cli"
    # backend reads this.
    claude_bin: str = Field(default="claude")

    # Left empty by default so the SDK resolves credentials itself — an API key,
    # ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile. Unused by the
    # "claude-cli" backend, which carries its own credentials.
    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-opus-4-8")

    language: str = Field(default="uk")
    summary_language: str = Field(default="uk")

    # The encoder handles ~400 s per pass; past that the server splits internally
    # and de-duplicates the seams itself, which loses text. Measured on a 1206 s
    # chunk: one request yields 10.7 chars/s, the same audio cut into 390 s pieces
    # here yields 15.2 — a 42% larger transcript. Keep this just under the cap.
    chunk_seconds: float = Field(default=390.0)
    # Narrow, because the window is now a third of the chunk rather than a tenth:
    # a wide search would let a chunk drift past the encoder's single-pass limit.
    chunk_search_window: float = Field(default=30.0)

    def describe(self) -> dict[str, str]:
        """
        Render the effective settings for logging at startup.

        Excludes `file_id`, which the CLI argument may override — the caller logs
        the resolved value instead. Secrets are reported as set/unset only, never
        echoed, because these lines end up in terminal scrollback and CI logs.

        Returns:
            Ordered display name to value, safe to write to a log.
        """
        return {
            "cache_dir": str(self.cache_dir),
            "docs_dir": str(self.docs_dir),
            "logs_dir": str(self.logs_dir),
            "google_service_account": str(self.google_service_account),
            "parakeet_base_url": self.parakeet_base_url,
            "parakeet_api_key": _secret_state(self.parakeet_api_key),
            "summary_backend": self.summary_backend,
            "claude_bin": self.claude_bin,
            "anthropic_model": self.anthropic_model,
            "anthropic_api_key": _secret_state(self.anthropic_api_key),
            "language": self.language,
            "summary_language": self.summary_language,
            "chunk_seconds": f"{self.chunk_seconds:g}",
            "chunk_search_window": f"{self.chunk_search_window:g}",
        }

    @property
    def models_dir(self) -> Path:
        """Directory holding the Parakeet ONNX weights."""
        return self.cache_dir / "models"

    def job_dir(self, file_id: str) -> Path:
        """
        Per-lecture working directory.

        Args:
            file_id: The Google Drive file ID being processed.

        Returns:
            The directory holding every intermediate artifact for that file.
        """
        return self.cache_dir / "jobs" / file_id
