# Workspace: Master's degree labs

Two projects share this repository. **Lecture Transcriber** (`./lecture-transcriber/`) turns the video recording of a lecture into text artifacts for future LLM analysis and usage; its core artifacts are transcription and summary. **Lab reports** (`./src/<term>/<subject>/`) are the coursework: one Python package per subject plus one LaTeX report per lab. The Quick Context below covers the transcriber; the Lab Reports section covers the coursework; Role, Coding Conventions and Do NOT apply to both.

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
- The repository root is a workspace holding one project per directory. This tool is `./lecture-transcriber/`, with its own `pyproject.toml`, `uv.lock`, `.python-version` and `src/lecture_transcriber/` plus `tests/` — each project pins its own interpreter. The shared data trees — `./docs/`, `./.cache/`, `./logs/` — and `.env`, `Makefile`, `docker-compose.yaml` and `.pre-commit-config.yaml` stay at the root
- The pipeline's path settings are working-directory-relative, so **run it from the root**: `uv run --project lecture-transcriber`, which points uv at the subdir without a `cd`. Never use `--directory` (or a bare `uv run` from inside the subdir) for the pipeline — it silently writes a second, empty `docs/` tree there instead of failing. Dev tools are the opposite: `pre-commit`, `pytest`, `ty` and `pip-audit` need `--directory lecture-transcriber` so their own config discovery works
- Use the **fp32** Parakeet weights, never the `.int8` ones. The int8 encoder is broken on this audio: it emits the blank token at nearly every timestep and yields ~1 transcribed character per second against fp32's 14, while running no faster on CPU. fp32 costs 2.5 GB on disk and ~3 GB RSS per worker
- Audio goes to the server as 16 kHz mono WAV, not Opus. The server's internal ffmpeg conversion reports a bogus `duration` for compressed input — 6.11 s for a 60 s Opus file — which lands in the segment end times and silently corrupts `transcript_coverage`
- Keep `chunk_seconds` under ~400 s, the encoder's single-pass limit. Past it the server splits internally and de-duplicates the seams, losing text: one 1206 s request yields 10.7 chars/s where the same audio in 390 s pieces yields 15.2
- Three output trees, and the distinction matters: `./.cache/` holds media, weights, credentials and intermediates and is gitignored; `./docs/` holds the committed Markdown artifacts; `./logs/` holds one gitignored log file per run, named `YYYY-MM-DD_HH-MM-SS.log` — at the top level for the transcriber, in a per-lab subdirectory for the coursework
- Every run logs its effective settings and the resolved Drive path up front. Secrets are reported as `<set>`/`<unset>`, and the HTTP client loggers are capped at WARNING — their DEBUG records dump request headers, which would write bearer tokens to disk. Keep it that way when adding logging
- `./docs/` mirrors the Drive folder hierarchy, one directory per lecture holding `summary.md` and `transcript.md`, both carrying YAML frontmatter for provenance

## Lab Reports

- One project per subject at `./src/<term>/<subject>/`, holding its own `pyproject.toml` and `src/<package>/`, plus one `lab<N>/` directory per lab containing `main.tex` and `run.py`. Each subject pins its own interpreter, which is not the transcriber's 3.14 — the AI subject is held at `>=3.12,<3.13` by its TensorFlow pin. The package holds all the computation; `run.py` is orchestration only — the list of stages and the check that every declared artifact really appeared
- One make target for the coursework: `make render-report` renders a lab's PDF into `./docs/Reports/`, driven by `REPORT` (the path to a `lab<N>/` directory, read from `.env` so a bare invocation works). It names no subject — term, subject and lab all come from where `REPORT` points, so a new subject or lab needs no change to the `Makefile`. Producing a lab's artifacts has no target: launch its `run.py` directly, from the workspace root, with `uv run --project "<subject dir>" python "<lab dir>/run.py"`
- Three trees, all derived from the lab's own location, never duplicated as strings: figures and metrics to `./.cache/reports-artifacts/<term>/<subject>/`, run logs to `./logs/<term>/<subject>/<lab>/`, the finished PDF to `./docs/Reports/<term>/<subject>/`. `config.SUBJECT_PATH` computes the shared `<term>/<subject>` part from `__file__`
- Reports embed their sources with `\inputminted[firstline=,lastline=]`, so **editing a lab module shifts the line ranges in `main.tex`** — a stale range silently prints the wrong lines, or cuts a listing mid-expression. After changing a module, re-derive every range that points at it, rebuild the PDF, and read the rendered listing to confirm it starts and ends on a complete statement
- **Render a report only through `make render-report`** — never by calling `latexmk` in a lab's directory, and turn off any build-on-save in the editor's LaTeX plugin. The target carries three arguments that exist nowhere else: `-lualatex`, because under pdfLaTeX `fontspec` is a fatal error and the listings package cannot read the Cyrillic in the sources; `-outdir=build`, because `build/` is the one gitignored place for LaTeX scratch; and `-jobname=lab<N>`, which names the PDF after the lab. A bare `latexmk main.tex` loses all three at once: it fails on `fontspec`, and still leaves `main.aux`, `main.log`, `main.fls` and `main.fdb_latexmk` beside `main.tex`, outside `build/` and tracked. There is no `.latexmkrc` anywhere in the repository to fall back on, by design — the `Makefile` is the single place these flags live

## Coding Conventions

- Absolute imports by package name only — `from lecture_transcriber.audio import extract_audio`, never `from .audio import ...` or `from . import audio`
- Import the names you use, not the module — `from lecture_transcriber.drive import download`, then call `download(...)`. Applies to third-party too (`from anthropic import Anthropic`). The stdlib is the exception: `import logging`, `import json`, `import re` stay as they are
- Type hints on all function signatures — parameters and return types
- Docstrings on public functions using Google style (see template below)
- Use the `logging` module — `log = logging.getLogger(__name__)`; never `print()`
- Comments explain **why**, not what
- **Write Python in English** — comments, docstrings, log records and exception messages, with no exceptions for a subject whose report is in Ukrainian. The one case that stays in another language is a string that is itself a deliverable rather than code: a matplotlib title, axis label or legend rendered into a figure the Ukrainian report displays. Say so in the module docstring when a module holds such strings, so the next reader does not "fix" them
- **Every entry point runs from the workspace root.** Derive paths from `__file__` (`Path(__file__).resolve().parent...`), never from the working directory, and document only the root-relative command — a lab's `run.py` is launched by its full path from the root (`uv run --project "<subject dir>" python "<lab dir>/run.py"`), never by `cd`-ing into its directory first. The transcriber is the one exception to `__file__`-derivation — its output paths come from settings and so are working-directory-relative, which is exactly why it too must be run from the root; see the `--project` rule above
- **Every entry point writes a log file.** Console at INFO, file at DEBUG, one file per run named `YYYY-MM-DD_HH-MM-SS.log` in local time, under this project's own subdirectory of `./logs/`. A long run is expensive to repeat, so the detail needed to diagnose it has to survive a run nobody was watching. Pass `force=True` to `basicConfig`: it is a no-op once anything else has touched the root logger, and TensorFlow, matplotlib and the Google client all do so on import — one of them getting there first would silently cost the log file. Cap the noisy third-party loggers while you are there (matplotlib and PIL at WARNING, the HTTP stacks per the secrets rule above)
- Each pipeline stage caches its output under `./.cache/jobs/<drive-file-id>/`; a failure in a later stage must never force a re-run of transcription
- Record `transcript_coverage` in artifact frontmatter — a run where the ASR server silently dropped audio must be detectable after the fact
- f-strings for string formatting (no `.format()` or `%` formatting)
- Prefer `pathlib.Path` for new file-path code
- Commit messages follow Commitizen / Conventional Commits: `type(scope): subject`, imperative and lowercase, no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`. Scope is the module or area — `drive`, `asr`, `audio`, `summarize`, `cli`, `docker`, `docs` for the transcriber; the subject's package name (`aiit`) or `reports` for the coursework

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
- Do not write a comment, docstring or log message in Ukrainian, and do not document a command that only works after a `cd` into a subdirectory
- Do not leave a script logging to the console alone — no bare `basicConfig` with a `StreamHandler` and no file
