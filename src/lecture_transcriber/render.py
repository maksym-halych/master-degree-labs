"""Markdown artifacts written into a mirror of the Drive folder hierarchy."""

import json
import logging
import re
import subprocess
import unicodedata
from functools import cache
from pathlib import Path

from lecture_transcriber.models import LectureSource, Transcript

log = logging.getLogger(__name__)

# ext4 caps a single path component at 255 bytes; Cyrillic costs two bytes a
# character, so budget in bytes rather than characters.
_MAX_COMPONENT_BYTES = 200

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _safe_component(name: str) -> str:
    """
    Convert a Drive item name into one safe path component.

    Drive permits characters a filesystem does not — most importantly `/`, which
    would silently create an extra directory level and break the mirroring.

    Args:
        name: The raw Drive name.

    Returns:
        A non-empty, single-level path component.
    """
    cleaned = unicodedata.normalize("NFC", name)
    cleaned = _CONTROL.sub("", cleaned).replace("/", "-").replace("\\", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Leading dots hide the directory; trailing dots and spaces are legal on Linux
    # but confuse archives and other platforms.
    cleaned = cleaned.strip(". ")

    encoded = cleaned.encode("utf-8")[:_MAX_COMPONENT_BYTES]
    # Truncation can split a multi-byte character, so drop any partial tail.
    cleaned = encoded.decode("utf-8", errors="ignore").strip()

    return cleaned or "untitled"


@cache
def pipeline_revision() -> str | None:
    """
    Identify the code revision that produced an artifact.

    A `-dirty` suffix marks uncommitted changes in the working tree, so an
    artifact can never be attributed to a commit that does not describe it.

    Returns:
        The commit hash, or None when the pipeline runs outside a git checkout
        or git is unavailable.
    """
    repo = Path(__file__).resolve().parent

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", "-C", str(repo), *args),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except OSError, subprocess.SubprocessError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = _git("rev-parse", "HEAD")
    if revision is None:
        log.warning("no git revision available; artifact provenance will omit it")
        return None

    # An empty diff means the checkout matches HEAD; None means the check itself
    # failed, which is not evidence of cleanliness.
    dirty = _git("status", "--porcelain")
    return revision if dirty == "" else f"{revision}-dirty"


def artifact_dir(source: LectureSource, docs_dir: Path) -> Path:
    """
    Locate a recording's artifact directory, mirroring its Drive path.

    Args:
        source: Metadata of the Drive recording.
        docs_dir: Root of the committed artifact tree.

    Returns:
        The directory that should hold this lecture's Markdown files.
    """
    parts = [_safe_component(folder) for folder in source.folders]
    parts.append(_safe_component(source.stem))
    return docs_dir.joinpath(*parts)


def _format_timestamp(seconds: float) -> str:
    """
    Render a position in the recording as HH:MM:SS.

    Args:
        seconds: Offset from the start of the recording.

    Returns:
        The zero-padded timestamp.
    """
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _frontmatter(fields: dict[str, object]) -> str:
    """
    Build a YAML frontmatter block.

    Values are emitted as JSON, which YAML 1.2 accepts verbatim — this quotes and
    escapes Cyrillic titles containing colons or quotes without a YAML dependency.

    Args:
        fields: Mapping of frontmatter keys to values; None values are skipped.

    Returns:
        The frontmatter block, including delimiters and a trailing blank line.
    """
    lines = ["---"]
    for key, value in fields.items():
        if value is None:
            continue
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("---\n")
    return "\n".join(lines)


def _common_fields(
    source: LectureSource, transcript: Transcript, settings_model: str
) -> dict[str, object]:
    """
    Assemble provenance shared by both artifacts.

    Args:
        source: Metadata of the Drive recording.
        transcript: The lecture transcript.
        settings_model: The summarization model identifier.

    Returns:
        Frontmatter fields describing where the artifact came from.
    """
    return {
        "title": source.stem,
        "drive_file_id": source.file_id,
        "drive_path": "/".join((*source.folders, source.name)),
        "drive_path_complete": source.path_complete,
        "recorded": source.recorded_date,
        "duration": _format_timestamp(transcript.duration),
        "language": transcript.language,
        "asr_model": "parakeet-tdt-0.6b-v3",
        "summary_model": settings_model,
        "pipeline_revision": pipeline_revision(),
        # Below 1.0 means the ASR server returned no speech for part of the audio.
        "transcript_coverage": round(
            transcript.covered_seconds / transcript.duration
            if transcript.duration
            else 0.0,
            3,
        ),
    }


def write_artifacts(
    summary_md: str,
    transcript: Transcript,
    source: LectureSource,
    docs_dir: Path,
    summary_model: str,
) -> tuple[Path, Path]:
    """
    Write the summary and transcript Markdown files.

    Args:
        summary_md: The Markdown summary from `summarize`.
        transcript: The full lecture transcript.
        source: Metadata of the Drive recording.
        docs_dir: Root of the committed artifact tree.
        summary_model: The model that produced the summary, recorded for provenance.

    Returns:
        Paths to the written summary and transcript.
    """
    target = artifact_dir(source, docs_dir)
    target.mkdir(parents=True, exist_ok=True)
    fields = _common_fields(source, transcript, summary_model)

    summary_path = target / "summary.md"
    summary_path.write_text(
        _frontmatter({**fields, "kind": "summary"}) + "\n" + summary_md.strip() + "\n",
        encoding="utf-8",
    )

    # One segment per paragraph keeps git diffs line-oriented when a re-run changes
    # only part of the transcription.
    body = "\n\n".join(
        f"**[{_format_timestamp(segment.start)}]** {segment.text.strip()}"
        for segment in transcript.segments
        if segment.text.strip()
    )
    transcript_path = target / "transcript.md"
    transcript_path.write_text(
        _frontmatter({**fields, "kind": "transcript"})
        + f"\n# {source.stem} — транскрипт\n\n"
        + body
        + "\n",
        encoding="utf-8",
    )

    log.info("wrote artifacts to %s", target)
    return summary_path, transcript_path
