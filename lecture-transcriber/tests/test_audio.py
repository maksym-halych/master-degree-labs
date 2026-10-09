"""Tests for silence-aware chunking and audio measurement."""

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from lecture_transcriber.audio import (
    AudioError,
    _chunk_spans,
    _cut_points,
    _parse_silences,
    _voiced_regions,
    mean_volume,
    probe_duration,
    split_audio,
)
from lecture_transcriber.models import Segment, Transcript

_RATE = 16_000

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)


def _write_wav(path: Path, parts: list[tuple[str, float]]) -> Path:
    """
    Write a 16 kHz mono WAV from tone and digital-silence sections.

    Args:
        path: Destination file.
        parts: (kind, seconds) pairs in order, kind being "tone" or "zeros".

    Returns:
        The written path.
    """
    samples: list[int] = []
    for kind, seconds in parts:
        count = int(seconds * _RATE)
        if kind == "tone":
            samples.extend(
                int(10_000 * math.sin(2 * math.pi * 440 * i / _RATE))
                for i in range(count)
            )
        else:
            samples.extend([0] * count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_RATE)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


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


def test_silence_report_is_read_into_spans() -> None:
    """Each silence_start pairs with the silence_end that follows it."""
    report = (
        "[silencedetect @ 0x1] silence_start: 2.22\n"
        "[silencedetect @ 0x1] silence_end: 186.75 | silence_duration: 184.53\n"
        "[silencedetect @ 0x1] silence_start: 300\n"
        "[silencedetect @ 0x1] silence_end: 312.5 | silence_duration: 12.5\n"
    )
    assert _parse_silences(report, duration=400.0) == [(2.22, 186.75), (300.0, 312.5)]


def test_silence_starting_before_zero_keeps_the_pairs_aligned() -> None:
    """A negative start must not be skipped, or every later span would be misread."""
    report = (
        "silence_start: -0.0213\n"
        "silence_end: 47.56 | silence_duration: 47.58\n"
        "silence_start: 100\n"
        "silence_end: 120 | silence_duration: 20\n"
    )
    assert _parse_silences(report, duration=200.0) == [(0.0, 47.56), (100.0, 120.0)]


def test_silence_running_into_the_end_is_closed_at_the_duration() -> None:
    """A trailing silence ffmpeg never closed is still a span to skip."""
    assert _parse_silences("silence_start: 5000\n", duration=5016.0) == [
        (5000.0, 5016.0)
    ]


def test_voiced_regions_exclude_every_gap() -> None:
    """Silence at the start, in the middle and at the end is all left out."""
    gaps = [(0.0, 47.0), (2000.0, 2030.0), (5700.0, 5763.0)]
    assert _voiced_regions(5763.0, gaps) == [(47.0, 2000.0), (2030.0, 5700.0)]


def test_voiced_regions_without_gaps_cover_everything() -> None:
    """A recording with no long silence is sent whole."""
    assert _voiced_regions(100.0, []) == [(0.0, 100.0)]


def test_voiced_regions_drop_a_blip_between_two_gaps() -> None:
    """A click between two long silences is not worth a request of its own."""
    gaps = [(10.0, 30.0), (30.4, 60.0)]
    assert _voiced_regions(100.0, gaps) == [(0.0, 10.0), (60.0, 100.0)]


def test_voiced_regions_of_a_silent_recording_are_empty() -> None:
    """Nothing is left to transcribe when the whole file is silence."""
    assert _voiced_regions(100.0, [(0.0, 100.0)]) == []


def test_no_chunk_holds_a_long_silence() -> None:
    """
    The shape that broke the ASR server: a long silence inside one chunk.

    A chunk opening on 184 s of silence came back empty, so no span may overlap
    a gap, whatever the target length would otherwise allow.
    """
    gaps = [(2.7, 186.3)]
    spans = _chunk_spans(5016.0, gaps, silences=[], target=390.0, window=30.0)
    assert spans[0] == (0.0, 2.7)
    assert spans[1][0] == 186.3
    assert all(end <= 2.7 or start >= 186.3 for start, end in spans)


def test_chunks_after_a_gap_carry_offsets_on_the_full_timeline() -> None:
    """Segment timestamps are shifted by the chunk offset, so it must be absolute."""
    gaps = [(100.0, 200.0)]
    spans = _chunk_spans(900.0, gaps, silences=[590.0], target=390.0, window=30.0)
    # The cut lands on the silence at 590 s: 390 s into the region, not the file.
    assert spans == [(0.0, 100.0), (200.0, 590.0), (590.0, 900.0)]


def test_chunk_spans_stay_under_the_target_plus_window() -> None:
    """The encoder's single-pass limit still holds inside every region."""
    gaps = [(50.0, 80.0), (3000.0, 3100.0)]
    spans = _chunk_spans(5000.0, gaps, silences=[], target=390.0, window=30.0)
    assert all(end - start <= 390.0 + 30.0 for start, end in spans)
    assert all(end > start for start, end in spans)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_chunk_spans_tile_everything_but_the_gaps() -> None:
    """Skipping silence must never also skip speech."""
    gaps = [(50.0, 80.0), (3000.0, 3100.0)]
    spans = _chunk_spans(5000.0, gaps, silences=[], target=390.0, window=30.0)
    sent = sum(end - start for start, end in spans)
    assert sent == pytest.approx(5000.0 - 30.0 - 100.0)


@needs_ffmpeg
def test_split_audio_leaves_a_long_silence_out(tmp_path: Path) -> None:
    """Real ffmpeg output: the silence is skipped and the offsets point past it."""
    audio = _write_wav(
        tmp_path / "audio.wav", [("tone", 20.0), ("zeros", 30.0), ("tone", 20.0)]
    )
    chunks = split_audio(audio, tmp_path / "chunks", 390.0, 30.0)

    assert [chunk.offset for chunk in chunks] == pytest.approx([0.0, 49.5], abs=0.1)
    assert [chunk.duration for chunk in chunks] == pytest.approx([20.5, 20.5], abs=0.1)
    for chunk in chunks:
        assert probe_duration(chunk.path) == pytest.approx(chunk.duration, abs=0.1)


@needs_ffmpeg
def test_split_audio_sends_a_short_recording_without_silence_whole(
    tmp_path: Path,
) -> None:
    """Nothing to skip and nothing to cut: the file itself is the only chunk."""
    audio = _write_wav(tmp_path / "audio.wav", [("tone", 5.0), ("zeros", 3.0)])
    chunks = split_audio(audio, tmp_path / "chunks", 390.0, 30.0)
    assert [chunk.path for chunk in chunks] == [audio]


@needs_ffmpeg
def test_split_audio_replaces_a_stale_chunk(tmp_path: Path) -> None:
    """A chunk left by an earlier run with other bounds must not be reused."""
    audio = _write_wav(
        tmp_path / "audio.wav", [("tone", 20.0), ("zeros", 30.0), ("tone", 20.0)]
    )
    stale = _write_wav(tmp_path / "chunks" / "chunk_000.wav", [("tone", 60.0)])

    split_audio(audio, tmp_path / "chunks", 390.0, 30.0)
    assert probe_duration(stale) == pytest.approx(20.5, abs=0.1)


@needs_ffmpeg
def test_split_audio_rejects_a_silent_recording(tmp_path: Path) -> None:
    """A file with no sound at all is an error, not an empty transcript."""
    audio = _write_wav(tmp_path / "audio.wav", [("zeros", 30.0)])
    with pytest.raises(AudioError, match="no sound"):
        split_audio(audio, tmp_path / "chunks", 390.0, 30.0)


@needs_ffmpeg
def test_mean_volume_tells_speech_level_from_silence(tmp_path: Path) -> None:
    """The loudness gate rests on this reading."""
    tone = _write_wav(tmp_path / "tone.wav", [("tone", 2.0)])
    zeros = _write_wav(tmp_path / "zeros.wav", [("zeros", 2.0)])
    assert mean_volume(tone) > -20.0
    assert mean_volume(zeros) <= -90.0
