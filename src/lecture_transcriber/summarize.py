"""Lecture summarization with Claude."""

import json
import logging
import subprocess

from anthropic import Anthropic, APIError

from lecture_transcriber.config import Settings
from lecture_transcriber.models import Transcript

log = logging.getLogger(__name__)

# A 90-minute lecture is roughly 25K input tokens, so the whole transcript fits
# in one request — no map-reduce needed.
_MAX_TOKENS = 16000

_EFFORT = "high"

# Generous: a full lecture at high effort takes minutes, and the only thing worse
# than a slow summary is killing one that was nearly done. Bounded all the same,
# so a wedged subprocess cannot hang the pipeline forever.
_CLI_TIMEOUT_SECONDS = 900.0

_SYSTEM = """\
You summarize university lecture transcripts into durable study artifacts.

The transcript comes from automatic speech recognition, so expect misheard terms,
missing punctuation, and mangled proper nouns. Infer the intended term from context
and use the correct spelling; never quote an obvious transcription error as fact.

Write the summary in {summary_language}. Keep technical terms, library names, and
code identifiers in their original language — do not translate them.

Return Markdown using exactly these headings, in this order:

# <lecture title>
## TL;DR
## Outline
## Key concepts
## Formulas, code and examples
## Open questions
## Glossary

Rules:
- `Outline` is a bulleted list of topics, each prefixed with its `[HH:MM]` start time.
- `Key concepts` defines each important idea in two or three sentences.
- `Formulas, code and examples` reproduces anything the lecturer worked through. Omit the
  section if there was none.
- `Open questions` lists what was left unresolved or deferred to later lectures. Omit if empty.
- `Glossary` maps each domain term to a one-line definition.
- Never invent content that is not in the transcript. If something is inaudible or
  incoherent, say so rather than guessing.
"""


class SummaryError(RuntimeError):
    """Raised when the summary could not be generated."""


def _timestamped_transcript(transcript: Transcript) -> str:
    """
    Render the transcript with timestamps for the model to cite.

    Args:
        transcript: The transcript to render.

    Returns:
        One `[HH:MM] text` line per segment.
    """
    lines = []
    for segment in transcript.segments:
        minutes, seconds = divmod(int(segment.start), 60)
        hours, minutes = divmod(minutes, 60)
        lines.append(
            f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {segment.text.strip()}"
        )
    return "\n".join(lines)


def summarize(transcript: Transcript, title: str, settings: Settings) -> str:
    """
    Produce a structured Markdown summary of a lecture.

    Dispatches to whichever backend `settings.summary_backend` selects; both
    return the same Markdown and raise the same error.

    Args:
        transcript: The full lecture transcript.
        title: Name of the source recording, used as a title hint.
        settings: Runtime configuration.

    Returns:
        The summary as Markdown.

    Raises:
        SummaryError: If the request fails or Claude declines it.
    """
    system = _SYSTEM.format(summary_language=settings.summary_language)
    prompt = (
        f"Recording: {title}\n\nTranscript:\n\n{_timestamped_transcript(transcript)}"
    )
    log.info(
        "summarizing %d segments with %s via the %s backend",
        len(transcript.segments),
        settings.anthropic_model,
        settings.summary_backend,
    )

    if settings.summary_backend == "claude-cli":
        return _summarize_via_cli(prompt, system, settings)
    return _summarize_via_api(prompt, system, settings)


def _summarize_via_api(prompt: str, system: str, settings: Settings) -> str:
    """
    Summarize by calling the Messages API directly.

    Args:
        prompt: The user turn carrying the recording name and transcript.
        system: The rendered system prompt.
        settings: Runtime configuration.

    Returns:
        The summary as Markdown.

    Raises:
        SummaryError: If the API call fails or Claude declines the request.
    """
    client = Anthropic(api_key=settings.anthropic_api_key or None)
    # Checked before the request is built: the SDK only notices missing
    # credentials while assembling headers, and signals it with a bare TypeError
    # that escapes the APIError handler below as an unhandled traceback.
    if not (client.api_key or client.auth_token):
        raise SummaryError(
            "no Anthropic credentials — set ANTHROPIC_API_KEY, or set "
            "SUMMARY_BACKEND=claude-cli to reuse the Claude Code login instead"
        )

    try:
        # Streaming keeps a long summary from tripping the SDK's request timeout.
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": _EFFORT},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()
    except APIError as exc:
        raise SummaryError(f"Claude request failed: {exc}") from exc

    if message.stop_reason == "refusal":
        raise SummaryError("Claude declined to summarize this transcript")

    text = "\n".join(
        block.text for block in message.content if block.type == "text"
    ).strip()
    if not text:
        raise SummaryError("Claude returned an empty summary")

    return text


def _cli_argv(system: str, settings: Settings) -> list[str]:
    """
    Build the Claude Code command line for a single non-interactive summary.

    Args:
        system: The rendered system prompt.
        settings: Runtime configuration.

    Returns:
        The argv to execute; the prompt itself goes on stdin, not here.
    """
    return [
        settings.claude_bin,
        "--print",
        # Disables CLAUDE.md, skills, hooks, plugins and MCP servers while leaving
        # auth alone. This repo's CLAUDE.md is about writing Python and would only
        # pollute the context. Not `--bare`, which strips the same things but then
        # reads credentials strictly from ANTHROPIC_API_KEY — never the
        # interactive login, which is the entire point of this backend.
        "--safe-mode",
        # Replaces Claude Code's coding-agent prompt rather than appending to it,
        # so the model is only ever a lecture summarizer here.
        "--system-prompt",
        system,
        "--model",
        settings.anthropic_model,
        "--effort",
        _EFFORT,
        # Nothing to read or run: the transcript arrives on stdin. Disabling tools
        # also means no permission prompt can stall a run with no terminal.
        "--tools",
        "",
        # A parseable envelope, so a failure is detectable rather than being
        # written to docs/ as if it were a summary.
        "--output-format",
        "json",
        # One-shot work; leaving transcripts in Claude Code's session history
        # would clutter the user's own `claude --resume` picker.
        "--no-session-persistence",
    ]


def _summarize_via_cli(prompt: str, system: str, settings: Settings) -> str:
    """
    Summarize by shelling out to the Claude Code CLI.

    Reuses the credentials Claude Code already holds, which bills a Pro/Max
    subscription instead of API credits. Requires `claude` to be installed and
    logged in on this machine — unlike the API backend, this does not travel to
    another host or to CI.

    Args:
        prompt: The user turn carrying the recording name and transcript.
        system: The rendered system prompt.
        settings: Runtime configuration.

    Returns:
        The summary as Markdown.

    Raises:
        SummaryError: If the CLI is missing, times out, exits non-zero, or
            reports an error instead of a result.
    """
    argv = _cli_argv(system, settings)
    # Elide the 40-line system prompt so the record of what was invoked stays
    # readable. Index 0 compares against the last element, which is a flag.
    log.debug(
        "running %s",
        " ".join(
            "<system prompt>" if argv[i - 1] == "--system-prompt" else arg
            for i, arg in enumerate(argv)
        ),
    )

    try:
        completed = subprocess.run(
            argv,
            # The transcript goes on stdin, never in argv: Linux caps one argument
            # at 128 KiB and a 90-minute lecture is around 100 KB of text.
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_CLI_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SummaryError(
            f"{settings.claude_bin} not found — install Claude Code, or set "
            "SUMMARY_BACKEND=api to use an API key instead"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SummaryError(
            f"{settings.claude_bin} did not finish within {_CLI_TIMEOUT_SECONDS:g}s"
        ) from exc

    stderr = (completed.stderr or "").strip()
    if stderr:
        log.debug("%s stderr: %s", settings.claude_bin, stderr)

    if completed.returncode != 0:
        # A failed run still prints its envelope on stdout, and that is where the
        # reason lives — an expired login reports itself in `result` and writes
        # nothing at all to stderr.
        raise SummaryError(
            f"{settings.claude_bin} exited {completed.returncode}: "
            f"{_cli_failure_reason(completed.stdout, stderr)}"
        )

    return _cli_result(completed.stdout, settings)


def _cli_failure_reason(stdout: str, stderr: str) -> str:
    """
    Find the most informative explanation for a failed CLI run.

    Args:
        stdout: The subprocess's captured standard output.
        stderr: The subprocess's captured standard error, already stripped.

    Returns:
        A one-line reason, never empty.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        # `result` carries the human-readable message; `terminal_reason` and
        # `subtype` are the coarser classifications behind it.
        for key in ("result", "terminal_reason", "subtype"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return stderr or stdout.strip()[:200] or "no diagnostics"


def _cli_result(stdout: str, settings: Settings) -> str:
    """
    Extract the summary from the CLI's `--output-format json` envelope.

    Args:
        stdout: The subprocess's captured standard output.
        settings: Runtime configuration, for naming the binary in errors.

    Returns:
        The summary as Markdown.

    Raises:
        SummaryError: If the envelope is unparseable, reports an error, or
            carries no result text.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise SummaryError(
            f"{settings.claude_bin} returned no JSON envelope: "
            f"{stdout.strip()[:200] or '<empty stdout>'}"
        ) from exc

    if not isinstance(payload, dict):
        raise SummaryError(f"{settings.claude_bin} returned {type(payload).__name__}")

    # `is_error` is the authoritative flag: an auth failure sets it while leaving
    # `subtype` as "success", so reporting the subtype alone would be misleading.
    if payload.get("is_error"):
        raise SummaryError(
            f"{settings.claude_bin} failed: {_cli_failure_reason(stdout, '')}"
        )

    result = payload.get("result")
    text = result.strip() if isinstance(result, str) else ""
    if not text:
        raise SummaryError(f"{settings.claude_bin} returned an empty summary")

    return text
