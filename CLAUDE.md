# Project: Lecture Transcriber

Turn the video recording of lecture into text artifacts for future LLM analysis and usage. Core artifacts are transcription and summary.

## Role

You are a senior Python backend engineer specializing in typed Python, Google Workspace APIs, and LLM/ASR pipelines.
Prefer explicit over clever. Keep functions small and single-purpose. Avoid premature abstraction.
Before finalizing code, check for: type safety, error handling, edge cases.

## Quick Context

- Language: Python 3.14
- Package Manager: uv
- CLI framework: click
- ASR: NVIDIA Parakeet TDT 0.6B v3, self-hosted via `docker-compose.yaml` behind an OpenAI-compatible API
- Summarization: Claude
- Google Drive access is **read-only**, via a service account. The folder must be shared with the account's `client_email`. Nothing is ever written back to Drive — a service account has no storage quota and cannot create Drive files
- `make pre-commit` runs every hook over the tree; `make download-models` fetches the ONNX weights into `./.cache/models/`
- Use the **fp32** Parakeet weights, never the `.int8` ones. The int8 encoder is broken on this audio: it emits the blank token at nearly every timestep and yields ~1 transcribed character per second against fp32's 14, while running no faster on CPU. fp32 costs 2.5 GB on disk and ~3 GB RSS per worker
- Audio goes to the server as 16 kHz mono WAV, not Opus. The server's internal ffmpeg conversion reports a bogus `duration` for compressed input — 6.11 s for a 60 s Opus file — which lands in the segment end times and silently corrupts `transcript_coverage`
- Keep `chunk_seconds` under ~400 s, the encoder's single-pass limit. Past it the server splits internally and de-duplicates the seams, losing text: one 1206 s request yields 10.7 chars/s where the same audio in 390 s pieces yields 15.2
- Three output trees, and the distinction matters: `./.cache/` holds media, weights, credentials and intermediates and is gitignored; `./docs/` holds the committed Markdown artifacts; `./logs/` holds one gitignored log file per run, named `YYYY-MM-DD_HH-MM-SS.log`
- Every run logs its effective settings and the resolved Drive path up front. Secrets are reported as `<set>`/`<unset>`, and the HTTP client loggers are capped at WARNING — their DEBUG records dump request headers, which would write bearer tokens to disk. Keep it that way when adding logging
- `./docs/` mirrors the Drive folder hierarchy, one directory per lecture holding `summary.md` and `transcript.md`, both carrying YAML frontmatter for provenance

## Coding Conventions

- Absolute imports by package name only — `from lecture_transcriber.audio import extract_audio`, never `from .audio import ...` or `from . import audio`
- Import the names you use, not the module — `from lecture_transcriber.drive import download`, then call `download(...)`. Applies to third-party too (`from anthropic import Anthropic`). The stdlib is the exception: `import logging`, `import json`, `import re` stay as they are
- Type hints on all function signatures — parameters and return types
- Docstrings on public functions using Google style (see template below)
- Use the `logging` module — `log = logging.getLogger(__name__)`; never `print()`
- Comments explain **why**, not what
- Each pipeline stage caches its output under `./.cache/jobs/<drive-file-id>/`; a failure in a later stage must never force a re-run of transcription
- Record `transcript_coverage` in artifact frontmatter — a run where the ASR server silently dropped audio must be detectable after the fact
- f-strings for string formatting (no `.format()` or `%` formatting)
- Prefer `pathlib.Path` for new file-path code
- Commit messages follow Commitizen / Conventional Commits: `type(scope): subject`, imperative and lowercase, no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`. Scope is the module or area (`drive`, `asr`, `audio`, `summarize`, `cli`, `docker`, `docs`)

**Service/utility functions — Google style:**

```python
def get_item(item_id: int, db: Session) -> Item:
    """
    Fetch a single item by ID.

    Args:
        item_id: The item's primary key.
        db: Database session.

    Returns:
        The matching Item.

    Raises:
        NotFoundError: If no item matches the ID.
    """
```

## Do NOT

- Do not call `pip install` directly — use `uv add <name>` to manage dependencies
- Do not manually activate `.venv` — always use `uv run <command>`
- Do not edit `uv.lock` by hand — it is auto-managed by uv
- Do not commit anything under `./.cache/` — it holds the service account key, gigabyte media, and the 670 MB ONNX weights
- Do not add Google Drive write scopes — the pipeline is read-only by design
- Do not hard-wrap prose in Markdown — never break a sentence across lines. One paragraph (or one list item) per line, however long. Applies to every `.md` file, including this one
- Do not use `import *`
- Do not use mutable default arguments
- Do not add `print()` statements for debugging — use the `logging` module
- Do not use `from __future__ import annotations` at the top of a module
