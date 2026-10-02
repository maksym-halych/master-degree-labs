"""Tests for silence-aware chunk boundary selection."""

from lecture_transcriber.audio import _cut_points
from lecture_transcriber.models import Segment, Transcript


def test_short_recording_needs_no_interior_cuts() -> None:
    """A recording under the target length is left whole."""
    assert _cut_points(duration=100.0, silences=[50.0], target=300.0, window=60.0) == []


def test_cuts_snap_to_the_nearest_silence() -> None:
    """Boundaries prefer a quiet moment over an arbitrary offset."""
    silences = [17.0, 37.0, 57.0, 77.0, 97.0]
    assert _cut_points(duration=100.0, silences=silences, target=30.0, window=10.0) == [
        37.0,
        77.0,
    ]


def test_cuts_fall_back_to_hard_boundaries_when_no_silence_is_near() -> None:
    """Continuous speech must still be split rather than stalling the run."""
    assert _cut_points(duration=100.0, silences=[], target=40.0, window=5.0) == [
        40.0,
        80.0,
    ]


def test_cut_points_are_strictly_increasing() -> None:
    """A repeated or backwards boundary would produce empty or overlapping chunks."""
    silences = [10.0, 10.5, 11.0, 60.0]
    points = _cut_points(duration=200.0, silences=silences, target=50.0, window=45.0)
    assert points == sorted(set(points))
    assert all(second > first for first, second in zip(points, points[1:]))


def test_no_cut_lands_on_or_past_the_end() -> None:
    """A boundary at the duration would create a zero-length trailing chunk."""
    points = _cut_points(duration=100.0, silences=[99.5], target=95.0, window=10.0)
    assert all(point < 100.0 for point in points)


def test_boundaries_tile_the_recording_without_gaps() -> None:
    """Chunk offsets derived from the cuts must cover the audio exactly once."""
    duration = 1000.0
    points = _cut_points(
        duration, silences=[290.0, 610.0, 880.0], target=300.0, window=60.0
    )
    edges = [0.0, *points, duration]
    spans = [(b - a) for a, b in zip(edges, edges[1:])]
    assert sum(spans) == duration
    assert all(span > 0 for span in spans)


def test_segment_shift_moves_a_chunk_onto_the_full_timeline() -> None:
    """Chunk-local timings must be rebased by the chunk's offset."""
    shifted = Segment(0.0, 5.0, "текст").shifted(1200.0)
    assert (shifted.start, shifted.end) == (1200.0, 1205.0)


def test_coverage_detects_dropped_spans() -> None:
    """Coverage is the signal that the ASR server silently skipped audio."""
    partial = Transcript(
        language="uk",
        duration=1000.0,
        segments=(Segment(0.0, 100.0, "a"), Segment(900.0, 1000.0, "b")),
    )
    assert partial.covered_seconds == 200.0
    assert partial.covered_seconds / partial.duration == 0.2
