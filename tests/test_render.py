"""Tests for artifact layout, path sanitisation and frontmatter."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from lecture_transcriber.models import LectureSource, Segment, Transcript
from lecture_transcriber.render import (
    _safe_component,
    artifact_dir,
    pipeline_revision,
    write_artifacts,
)

_FOLDERS = (
    "Lecture-Recordings-mKNSSH-2026",
    "1th term 2026 Autumn",
    "Архітектура технічних систем та програмних рішень у розробці продуктів",
)


def _source(
    name: str = "2026-09-09 - Lecture 1.mkv",
    folders: tuple[str, ...] = _FOLDERS,
    path_complete: bool = True,
) -> LectureSource:
    """
    Build a LectureSource mirroring the real Drive layout.

    Args:
        name: Recording filename as it appears in Drive.
        folders: Ancestor folder names, outermost first.
        path_complete: Whether the folder walk reached a Drive root.

    Returns:
        The constructed LectureSource.
    """
    return LectureSource(
        file_id="1AbC_xyz",
        name=name,
        mime_type="video/x-matroska",
        size_bytes=1_500_000_000,
        created_time="2026-09-10T08:12:00.000Z",
        folders=folders,
        path_complete=path_complete,
    )


def _transcript() -> Transcript:
    """
    Build a small two-segment transcript.

    Returns:
        The constructed Transcript.
    """
    return Transcript(
        language="uk",
        duration=100.0,
        segments=(
            Segment(0.0, 40.0, "Доброго дня."),
            Segment(3612.4, 3660.0, 'Приклад: "REST vs gRPC" — компроміс.'),
        ),
    )


def test_artifact_dir_mirrors_drive_hierarchy(tmp_path: Path) -> None:
    """The on-disk path reproduces the Drive folder chain plus the lecture name."""
    result = artifact_dir(_source(), tmp_path / "docs")
    assert result.relative_to(tmp_path / "docs").parts == (
        "Lecture-Recordings-mKNSSH-2026",
        "1th term 2026 Autumn",
        "Архітектура технічних систем та програмних рішень у розробці продуктів",
        "2026-09-09 - Lecture 1",
    )


def test_partial_drive_path_still_produces_a_directory(tmp_path: Path) -> None:
    """An unshared ancestor truncates the path rather than failing the run."""
    source = _source(folders=("Архітектура",), path_complete=False)
    result = artifact_dir(source, tmp_path / "docs")
    assert result.relative_to(tmp_path / "docs").parts == (
        "Архітектура",
        "2026-09-09 - Lecture 1",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Lecture 1/2", "Lecture 1-2"),  # Drive allows `/`; a path component does not
        ("  spaced  out  ", "spaced out"),
        ("..", "untitled"),  # would escape the docs tree
        (".", "untitled"),
        (".hidden", "hidden"),
        ("trailing...", "trailing"),
        ("", "untitled"),
        ("with\x00null\x1fctrl", "withnullctrl"),
        ("back\\slash", "back-slash"),
    ],
)
def test_safe_component_neutralises_hostile_names(raw: str, expected: str) -> None:
    """Names that are legal in Drive but unsafe on disk are normalised."""
    assert _safe_component(raw) == expected


def test_safe_component_truncates_without_splitting_characters() -> None:
    """Byte-budget truncation never leaves a partial UTF-8 sequence."""
    result = _safe_component("я" * 300)
    assert len(result.encode("utf-8")) <= 200
    assert result == "я" * (200 // 2)  # decodes cleanly, so no replacement chars


def test_frontmatter_is_valid_yaml_with_cyrillic_and_punctuation(
    tmp_path: Path,
) -> None:
    """Colons, quotes and Cyrillic in Drive names must not corrupt the frontmatter."""
    source = _source(name='2026-09-09 - Lecture: "Вступ".mkv')
    summary_path, transcript_path = write_artifacts(
        "# Заголовок\n", _transcript(), source, tmp_path / "docs", "claude-opus-4-8"
    )

    for path in (summary_path, transcript_path):
        text = path.read_text(encoding="utf-8")
        _, _, rest = text.partition("---\n")
        block, _, _ = rest.partition("\n---")
        meta = yaml.safe_load(block)
        assert meta["title"] == '2026-09-09 - Lecture: "Вступ"'
        assert meta["drive_file_id"] == "1AbC_xyz"
        assert meta["recorded"] == "2026-09-09"


def test_recorded_date_prefers_filename_over_drive_timestamp() -> None:
    """A re-upload changes createdTime but must not change the lecture date."""
    assert _source().recorded_date == "2026-09-09"


def test_recorded_date_falls_back_to_drive_timestamp() -> None:
    """Without a date in the name, Drive's createdTime is the only signal."""
    assert _source(name="Lecture 1.mkv").recorded_date == "2026-09-10"


def test_coverage_is_recorded_so_bad_runs_are_visible(tmp_path: Path) -> None:
    """A partial transcription must be detectable from the committed artifact."""
    summary_path, _ = write_artifacts(
        "# x\n", _transcript(), _source(), tmp_path / "docs", "claude-opus-4-8"
    )
    block = summary_path.read_text(encoding="utf-8").split("---")[1]
    # 40 s + 47.6 s of speech across a 100 s recording.
    assert yaml.safe_load(block)["transcript_coverage"] == pytest.approx(0.876)


def test_transcript_segments_are_timestamped_and_separated(tmp_path: Path) -> None:
    """Each segment is its own paragraph so re-runs produce line-oriented diffs."""
    _, transcript_path = write_artifacts(
        "# x\n", _transcript(), _source(), tmp_path / "docs", "claude-opus-4-8"
    )
    body = transcript_path.read_text(encoding="utf-8")
    assert "**[00:00:00]** Доброго дня." in body
    assert "**[01:00:12]** Приклад:" in body


def test_pipeline_revision_is_recorded(tmp_path: Path) -> None:
    """The artifact names the code revision that produced it."""
    summary_path, transcript_path = write_artifacts(
        "# x\n", _transcript(), _source(), tmp_path / "docs", "claude-opus-4-8"
    )
    for path in (summary_path, transcript_path):
        block = path.read_text(encoding="utf-8").split("---")[1]
        assert yaml.safe_load(block)["pipeline_revision"] == pipeline_revision()


def test_pipeline_revision_flags_an_uncommitted_tree() -> None:
    """A hash alone would misattribute artifacts built from unstaged edits."""
    revision = pipeline_revision()
    assert revision is not None  # the test suite runs from a git checkout
    base = revision.removesuffix("-dirty")
    assert re.fullmatch(r"[0-9a-f]{40}", base)

    dirty = (
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        != ""
    )
    assert revision.endswith("-dirty") is dirty


def test_drive_path_joins_folders_and_name() -> None:
    """A fully-walked path reads as the Drive hierarchy, outermost folder first."""
    assert _source().drive_path == "/".join((*_FOLDERS, "2026-09-09 - Lecture 1.mkv"))


def test_drive_path_marks_truncation() -> None:
    """A permission-truncated path is flagged, so an odd docs/ target is traceable."""
    source = _source(folders=("1th term 2026 Autumn",), path_complete=False)
    assert source.drive_path == ".../1th term 2026 Autumn/2026-09-09 - Lecture 1.mkv"
