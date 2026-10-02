"""Audio extraction and silence-aware chunking via ffmpeg."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


class AudioError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails."""


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One piece of the lecture audio, positioned within the full recording."""

    path: Path
    offset: float
    duration: float


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """
    Execute a subprocess and capture its output.

    Args:
        command: argv of the process to run.

    Returns:
        The completed process.

    Raises:
        AudioError: If the binary is missing or exits non-zero.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise AudioError(f"{command[0]} is not installed") from exc
    if result.returncode != 0:
        raise AudioError(f"{command[0]} failed: {result.stderr.strip()[-500:]}")
    return result


def probe_duration(path: Path) -> float:
    """
    Measure the duration of a media file.

    Args:
        path: The media file to inspect.

    Returns:
        Duration in seconds.

    Raises:
        AudioError: If ffprobe fails or reports an unparseable duration.
    """
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AudioError(f"could not read duration of {path}") from exc


def extract_audio(video: Path, destination: Path) -> Path:
    """
    Strip the audio track into 16 kHz mono PCM.

    16 kHz mono is what the ASR server feeds the encoder, so handing it WAV skips
    its internal ffmpeg conversion entirely. That conversion is worth avoiding on
    two counts: it reports a bogus `duration` for compressed input (6.11 s for a
    60 s Opus file), which propagates into segment end times and makes
    transcript_coverage meaningless; and the 24 kbps Opus this used to produce
    cost real accuracy — 740 transcribed characters against WAV's 839 over the
    same 60 s. An hour of PCM is ~115 MB, which only ever moves over loopback.

    Args:
        video: Source video file.
        destination: Target `.wav` path; parent directories are created.

    Returns:
        The destination path.

    Raises:
        AudioError: If ffmpeg fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("extracting audio from %s", video.name)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    return destination


def _detect_silences(
    audio: Path, noise_db: int = -30, min_duration: float = 0.6
) -> list[float]:
    """
    Find quiet points suitable for cutting.

    Args:
        audio: The audio file to scan.
        noise_db: Threshold below which audio counts as silence.
        min_duration: Shortest silence worth considering, in seconds.

    Returns:
        Midpoints of every detected silence, in ascending order.
    """
    # silencedetect reports to stderr and the null muxer produces no output file.
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(audio),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    starts = [float(m) for m in _SILENCE_START.findall(result.stderr)]
    ends = [float(m) for m in _SILENCE_END.findall(result.stderr)]
    return [(start + end) / 2 for start, end in zip(starts, ends)]


def _cut_points(
    duration: float, silences: list[float], target: float, window: float
) -> list[float]:
    """
    Choose chunk boundaries, preferring silence over arbitrary cuts.

    Args:
        duration: Total audio length in seconds.
        silences: Candidate cut points from `_detect_silences`.
        target: Desired chunk length in seconds.
        window: How far from the target a silence may sit and still be used.

    Returns:
        Interior boundaries, ascending and strictly increasing.
    """
    points: list[float] = []
    position = target
    while position < duration:
        nearby = [
            s
            for s in silences
            if abs(s - position) <= window and s > (points[-1] if points else 0.0)
        ]
        # On a tie prefer the later silence: snapping backwards drags the next
        # target back too, compounding into chunks well short of the target.
        # Falling back to a hard cut risks clipping a word, but never stalls the split.
        chosen = (
            min(nearby, key=lambda s: (abs(s - position), -s)) if nearby else position
        )
        if chosen >= duration:
            break
        points.append(chosen)
        position = chosen + target
    return points


def split_audio(
    audio: Path,
    destination: Path,
    target_seconds: float,
    search_window: float,
) -> list[AudioChunk]:
    """
    Split audio into chunks on silence boundaries.

    A single chunk is returned unmodified when the recording is already short
    enough, avoiding a pointless copy.

    Args:
        audio: The WAV file to split.
        destination: Directory to write chunks into.
        target_seconds: Desired chunk length.
        search_window: How far to look for a silence near each boundary.

    Returns:
        Chunks in playback order, each carrying its offset in the full recording.

    Raises:
        AudioError: If ffmpeg fails.
    """
    duration = probe_duration(audio)
    if duration <= target_seconds:
        return [AudioChunk(path=audio, offset=0.0, duration=duration)]

    destination.mkdir(parents=True, exist_ok=True)
    silences = _detect_silences(audio)
    log.info("found %d silence candidates across %.0f s", len(silences), duration)

    boundaries = [
        0.0,
        *_cut_points(duration, silences, target_seconds, search_window),
        duration,
    ]
    chunks: list[AudioChunk] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        chunk_path = destination / f"chunk_{index:03d}.wav"
        if not chunk_path.exists():
            # Stream copy keeps this fast; PCM cuts cleanly on any sample boundary.
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-to",
                    f"{end:.3f}",
                    "-i",
                    str(audio),
                    "-c",
                    "copy",
                    str(chunk_path),
                ]
            )
        chunks.append(AudioChunk(path=chunk_path, offset=start, duration=end - start))

    log.info("split into %d chunks", len(chunks))
    return chunks
