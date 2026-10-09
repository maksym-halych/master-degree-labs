"""Tests for the Transcript model: coverage and the cached JSON form."""

import json
from pathlib import Path

import pytest

from lecture_transcriber.models import Segment, Transcript


def test_skipped_silence_does_not_lower_coverage() -> None:
    """Silence never sent to the server cannot have been dropped by it."""
    transcript = Transcript(
        language="uk",
        duration=5016.0,
        segments=(Segment(0.0, 2.7, "Добрий"), Segment(186.3, 5016.0, "ранок")),
        skipped_seconds=183.6,
    )
    assert transcript.coverage == pytest.approx(1.0)


def test_coverage_without_skipped_silence_is_against_the_whole_recording() -> None:
    """A transcript cached before silence was skipped keeps its old meaning."""
    transcript = Transcript(
        language="uk", duration=1000.0, segments=(Segment(0.0, 800.0, "a"),)
    )
    assert transcript.coverage == pytest.approx(0.8)


def test_round_trip_keeps_the_quality_metrics(tmp_path: Path) -> None:
    """A cached transcript must render the same frontmatter as a fresh one."""
    original = Transcript(
        language="uk",
        duration=600.0,
        segments=(Segment(0.0, 400.0, "текст"),),
        skipped_seconds=200.0,
        min_chunk_density=4.2,
    )
    path = tmp_path / "transcript.json"
    original.to_json(path)
    assert Transcript.from_json(path) == original


def test_a_transcript_cached_before_the_metrics_still_loads(tmp_path: Path) -> None:
    """Older cache files lack the new keys; their values are unknown, not zero."""
    path = tmp_path / "transcript.json"
    path.write_text(
        json.dumps(
            {
                "language": "uk",
                "duration": 60.0,
                "segments": [{"start": 0.0, "end": 60.0, "text": "a"}],
            }
        ),
        encoding="utf-8",
    )
    loaded = Transcript.from_json(path)
    assert loaded.skipped_seconds is None
    assert loaded.min_chunk_density is None


def test_a_non_numeric_metric_is_rejected(tmp_path: Path) -> None:
    """A corrupted cache must fail loudly rather than render nonsense."""
    path = tmp_path / "transcript.json"
    path.write_text(
        json.dumps(
            {
                "language": "uk",
                "duration": 60.0,
                "skipped_seconds": "lots",
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a valid transcript"):
        Transcript.from_json(path)
