"""Tests for transcription merging and the checks that catch a garbled chunk."""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import lecture_transcriber.asr
from lecture_transcriber.asr import (
    TranscriptionError,
    _judge_chunk,
    _transcribe_chunk,
    transcribe,
)
from lecture_transcriber.audio import AudioChunk
from lecture_transcriber.config import Settings
from lecture_transcriber.models import Segment

# Speech-level and near-silent mean volumes, as ffmpeg's volumedetect reports them.
_SPEECH_DB = -26.0
_QUIET_DB = -60.0


def _chunk(offset: float, duration: float, name: str = "chunk_000.wav") -> AudioChunk:
    """
    Build a chunk whose file is never opened.

    Args:
        offset: Position in the full recording, in seconds.
        duration: Chunk length in seconds.
        name: File name, used in error messages.

    Returns:
        The constructed AudioChunk.
    """
    return AudioChunk(path=Path(name), offset=offset, duration=duration)


def _text(chars_per_second: float, duration: float) -> str:
    """
    Produce text of a given density.

    Args:
        chars_per_second: Characters per second of audio.
        duration: Seconds of audio the text stands for.

    Returns:
        A string of the matching length.
    """
    return "а" * int(chars_per_second * duration)


def _stub_volume(monkeypatch: pytest.MonkeyPatch, volume: float) -> list[Path]:
    """
    Replace the ffmpeg loudness measurement with a fixed reading.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        volume: Mean volume in dB to report for every file.

    Returns:
        The list each measured path is appended to.
    """
    measured: list[Path] = []

    def fake(path: Path) -> float:
        measured.append(path)
        return volume

    monkeypatch.setattr(lecture_transcriber.asr, "mean_volume", fake)
    return measured


def test_a_short_chunk_is_not_judged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A few seconds between two silences hold too little speech to judge."""
    measured = _stub_volume(monkeypatch, _SPEECH_DB)
    assert _judge_chunk(_chunk(0.0, 2.7), []) is None
    assert measured == []


def test_a_healthy_chunk_reports_its_density(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lecture speech passes and its density feeds min_chunk_density."""
    _stub_volume(monkeypatch, _SPEECH_DB)
    chunk = _chunk(391.6, 389.3)
    segments = [Segment(391.6, 780.9, _text(4.1, 389.3))]
    assert _judge_chunk(chunk, segments) == pytest.approx(4.1, abs=0.01)


def test_speech_that_came_back_empty_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original failure: a speech-level chunk the server returned as ''."""
    _stub_volume(monkeypatch, _SPEECH_DB)
    with pytest.raises(TranscriptionError, match=r"chunk_000\.wav \(0-392 s"):
        _judge_chunk(_chunk(0.0, 391.6), [])


def test_speech_that_came_back_as_noise_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second failure: 161 characters of another language across 387 s."""
    _stub_volume(monkeypatch, _SPEECH_DB)
    segments = [Segment(0.0, 386.9, "E päris proastus na pale." + "x" * 136)]
    with pytest.raises(TranscriptionError, match="0.4 characters per second"):
        _judge_chunk(_chunk(0.0, 386.9), segments)


def test_a_quiet_chunk_with_little_text_is_reported_not_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Sparse text from near-silent audio may be genuine, so it only warns."""
    _stub_volume(monkeypatch, _QUIET_DB)
    with caplog.at_level(logging.WARNING, logger="lecture_transcriber.asr"):
        density = _judge_chunk(_chunk(0.0, 120.0), [])
    assert density == 0.0
    assert "may hold little speech" in caplog.text


def test_an_empty_segment_does_not_count_as_coverage(tmp_path: Path) -> None:
    """The server returns a segment spanning a garbled chunk even with no text."""
    audio = tmp_path / "chunk_000.wav"
    audio.write_bytes(b"")
    response = SimpleNamespace(
        text="", segments=[SimpleNamespace(start=0.0, end=391.6, text="")]
    )
    client: Any = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=lambda **_: response)
        )
    )
    assert _transcribe_chunk(client, _chunk(0.0, 391.6, str(audio)), "uk") == []


def _stub_server(
    monkeypatch: pytest.MonkeyPatch, densities: dict[float, float]
) -> list[float]:
    """
    Replace the ASR server with one returning text of a set density per chunk.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        densities: Characters per second to return, keyed by chunk offset.

    Returns:
        The list each transcribed chunk's offset is appended to.
    """
    sent: list[float] = []

    def fake(_client: object, chunk: AudioChunk, _language: str) -> list[Segment]:
        sent.append(chunk.offset)
        text = _text(densities[chunk.offset], chunk.duration)
        if not text:
            return []
        return [Segment(chunk.offset, chunk.offset + chunk.duration, text)]

    monkeypatch.setattr(lecture_transcriber.asr, "_client", lambda _settings: None)
    monkeypatch.setattr(lecture_transcriber.asr, "_transcribe_chunk", fake)
    return sent


def test_transcript_records_skipped_silence_and_the_weakest_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both new frontmatter metrics come from here."""
    _stub_volume(monkeypatch, _SPEECH_DB)
    _stub_server(monkeypatch, {0.0: 5.0, 2.7: 5.0, 186.3: 3.5, 576.3: 5.5})
    chunks = [
        _chunk(0.0, 2.7),
        _chunk(186.3, 390.0),
        _chunk(576.3, 400.0),
    ]
    transcript = transcribe(chunks, 976.3, Settings(_env_file=None))

    assert transcript.skipped_seconds == pytest.approx(183.6)
    # The 2.7 s chunk is too short to judge, so it does not set the minimum.
    assert transcript.min_chunk_density == pytest.approx(3.5, abs=0.01)
    assert transcript.coverage == pytest.approx(1.0)


def test_a_garbled_chunk_stops_the_run_before_the_rest_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing at once saves the server time a doomed run would spend."""
    _stub_volume(monkeypatch, _SPEECH_DB)
    sent = _stub_server(monkeypatch, {0.0: 0.0, 391.6: 5.0})
    chunks = [_chunk(0.0, 391.6), _chunk(391.6, 389.3)]

    with pytest.raises(TranscriptionError):
        transcribe(chunks, 780.9, Settings(_env_file=None))
    assert sent == [0.0]
