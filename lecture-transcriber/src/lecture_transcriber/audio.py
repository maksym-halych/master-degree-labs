"""Audio extraction and silence-aware chunking via ffmpeg."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# The sign matters: silencedetect can place a start a few milliseconds before zero,
# and a start that failed to match would pair every later start with the wrong end.
_SILENCE_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[0-9.]+)")
_MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[0-9.]+|-inf) dB")

# The ASR server normalizes each mel band by its mean and spread over the whole
# request, so a long silence anywhere in a chunk skews those statistics for every
# frame and garbles the speech around it: 184 s of silence beside 205 s of speech
# decodes to nothing when the silence leads or trails, and to a third of the text
# when it sits in the middle. Silences this long are therefore never sent. -50 dB
# sits well below lecture speech (-25 to -35 dB mean) yet catches both quiet room
# tone and digital silence; pauses within speech stay far under 10 s.
_GAP_NOISE_DB = -50
_GAP_MIN_SECONDS = 10.0
# Kept from each end of a skipped silence, so speech next to it still opens and
# closes on a quiet lead-in the way it does everywhere else.
_GAP_PADDING_SECONDS = 0.5
# Sound this short between two skipped silences is a click or a cough, not speech.
_MIN_REGION_SECONDS = 1.0


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


def mean_volume(path: Path) -> float:
    """
    Measure the average loudness of an audio file.

    Args:
        path: The audio file to measure.

    Returns:
        Mean volume in dBFS. 16-bit digital silence reads -91.

    Raises:
        AudioError: If ffmpeg fails or reports no volume.
    """
    # volumedetect reports to stderr and the null muxer produces no output file.
    result = _run(["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"])
    match = _MEAN_VOLUME.search(result.stderr)
    if match is None:
        raise AudioError(f"ffmpeg reported no volume for {path}")
    return float(match.group(1))


def _parse_silences(report: str, duration: float) -> list[tuple[float, float]]:
    """
    Read the spans out of a silencedetect report.

    Args:
        report: ffmpeg's stderr from a silencedetect pass.
        duration: Length of the scanned audio in seconds.

    Returns:
        (start, end) of every silence, ascending, clamped to the audio.
    """
    starts = [max(0.0, float(m)) for m in _SILENCE_START.findall(report)]
    ends = [min(duration, float(m)) for m in _SILENCE_END.findall(report)]
    # A silence that runs into the end of the file is not always closed.
    if len(ends) < len(starts):
        ends.append(duration)
    return list(zip(starts, ends))


def _detect_silences(
    audio: Path, duration: float, noise_db: int, min_duration: float
) -> list[tuple[float, float]]:
    """
    Find the silent spans of an audio file.

    Args:
        audio: The audio file to scan.
        duration: Length of the audio in seconds.
        noise_db: Threshold below which audio counts as silence.
        min_duration: Shortest silence worth reporting, in seconds.

    Returns:
        (start, end) of every detected silence, ascending.

    Raises:
        AudioError: If ffmpeg fails.
    """
    # silencedetect reports to stderr and the null muxer produces no output file.
    result = _run(
        [
            "ffmpeg",
            "-i",
            str(audio),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f",
            "null",
            "-",
        ]
    )
    return _parse_silences(result.stderr, duration)


def _voiced_regions(
    duration: float, gaps: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    Take the complement of the silences to be skipped.

    Args:
        duration: Total audio length in seconds.
        gaps: Silences to leave out, ascending.

    Returns:
        (start, end) of every stretch between gaps, ascending.
    """
    regions: list[tuple[float, float]] = []
    position = 0.0
    for start, end in gaps:
        if start - position >= _MIN_REGION_SECONDS:
            regions.append((position, start))
        position = max(position, end)
    if duration - position >= _MIN_REGION_SECONDS:
        regions.append((position, duration))
    return regions


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


def _chunk_spans(
    duration: float,
    gaps: list[tuple[float, float]],
    silences: list[float],
    target: float,
    window: float,
) -> list[tuple[float, float]]:
    """
    Lay chunks over the audio between the silences to be skipped.

    Args:
        duration: Total audio length in seconds.
        gaps: Silences to leave out of every chunk, ascending.
        silences: Candidate cut points within speech, from `_detect_silences`.
        target: Desired chunk length in seconds.
        window: How far from the target a silence may sit and still be used.

    Returns:
        (start, end) of every chunk, ascending; none overlaps a gap.
    """
    spans: list[tuple[float, float]] = []
    for start, end in _voiced_regions(duration, gaps):
        inside = [s - start for s in silences if start < s < end]
        cuts = [start + cut for cut in _cut_points(end - start, inside, target, window)]
        edges = [start, *cuts, end]
        spans.extend(zip(edges, edges[1:]))
    return spans


def split_audio(
    audio: Path,
    destination: Path,
    target_seconds: float,
    search_window: float,
) -> list[AudioChunk]:
    """
    Split audio into chunks on silence boundaries, leaving out long silences.

    The chunks need not tile the recording: a long silence is skipped rather than
    sent, so the chunk after it starts at a later offset. A single chunk is the
    file itself when the recording is short and has nothing to skip, avoiding a
    pointless copy.

    Args:
        audio: The WAV file to split.
        destination: Directory to write chunks into.
        target_seconds: Desired chunk length.
        search_window: How far to look for a silence near each boundary.

    Returns:
        Chunks in playback order, each carrying its offset in the full recording.

    Raises:
        AudioError: If ffmpeg fails, or the recording holds no sound to transcribe.
    """
    duration = probe_duration(audio)
    gaps = [
        (start + _GAP_PADDING_SECONDS, end - _GAP_PADDING_SECONDS)
        for start, end in _detect_silences(
            audio, duration, _GAP_NOISE_DB, _GAP_MIN_SECONDS
        )
    ]
    log.info(
        "skipping %d silences of %.0f s in total",
        len(gaps),
        sum(end - start for start, end in gaps),
    )
    for start, end in gaps:
        log.debug("  silence %.1f-%.1f s", start, end)

    if not gaps and duration <= target_seconds:
        return [AudioChunk(path=audio, offset=0.0, duration=duration)]

    silences = [
        (start + end) / 2
        for start, end in _detect_silences(
            audio, duration, noise_db=-30, min_duration=0.6
        )
    ]
    log.info("found %d silence candidates across %.0f s", len(silences), duration)

    spans = _chunk_spans(duration, gaps, silences, target_seconds, search_window)
    if not spans:
        raise AudioError(f"{audio.name} holds no sound above {_GAP_NOISE_DB} dB")

    destination.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []
    for index, (start, end) in enumerate(spans):
        chunk_path = destination / f"chunk_{index:03d}.wav"
        # Always rewritten, never reused: a chunk left by an earlier run may span
        # different bounds under the same name, and its segments would then be
        # shifted by the wrong offset. A stream copy of PCM costs a fraction of a
        # second and cuts cleanly on any sample boundary.
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
        log.debug("  %s: %.1f-%.1f s", chunk_path.name, start, end)
        chunks.append(AudioChunk(path=chunk_path, offset=start, duration=end - start))

    log.info("split into %d chunks", len(chunks))
    return chunks
