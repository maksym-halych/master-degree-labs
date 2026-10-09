"""Transcription against a self-hosted Parakeet server (OpenAI-compatible API)."""

import logging

from openai import APIConnectionError, APIStatusError, OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lecture_transcriber.audio import AudioChunk, mean_volume
from lecture_transcriber.config import Settings
from lecture_transcriber.models import Segment, Transcript

log = logging.getLogger(__name__)

# The server accepts `model` but ignores it; sent only to satisfy the OpenAI schema.
_MODEL = "parakeet-tdt-0.6b"

# Below this ratio of transcribed-to-actual duration we assume spans were dropped
# rather than that the lecturer was silent for a third of the recording.
_MIN_COVERAGE = 0.5

# Coverage alone cannot detect a failing model: the server returns one segment
# spanning the whole chunk, so a run that decodes to near-silence still scores
# 100%. Lecture speech measures 13-15 characters per second of audio; a degraded
# encoder collapses to ~1. Four is clear of both.
_MIN_CHARS_PER_SECOND = 4.0

# The same blind spot hides a single chunk the server garbled, which the
# recording-wide density above averages away. Healthy lecture chunks measure 3.4
# to 10.7 characters per second; chunks garbled by a long silence, 0.0 to 0.4.
_MIN_CHUNK_CHARS_PER_SECOND = 2.0
# Below this length a chunk holds a sentence or two, too few to judge by density.
_MIN_JUDGED_CHUNK_SECONDS = 60.0
# Lecture speech measures -25 to -35 dB mean volume. A chunk quieter than this may
# genuinely hold little speech, so sparse text from it is reported, not fatal.
_SPEECH_VOLUME_DB = -40.0


class TranscriptionError(RuntimeError):
    """Raised when the ASR server cannot produce a usable transcript."""


def _client(settings: Settings) -> OpenAI:
    """
    Build an OpenAI client pointed at the local Parakeet server.

    Args:
        settings: Runtime configuration.

    Returns:
        A configured client.
    """
    return OpenAI(
        base_url=settings.parakeet_base_url, api_key=settings.parakeet_api_key
    )


@retry(
    retry=retry_if_exception_type((APIConnectionError, APIStatusError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _transcribe_chunk(
    client: OpenAI, chunk: AudioChunk, language: str
) -> list[Segment]:
    """
    Transcribe a single chunk and place its segments on the full timeline.

    Args:
        client: Client for the Parakeet server.
        chunk: The audio chunk to transcribe.
        language: ISO-639-1 code; the server defaults to English if omitted.

    Returns:
        Segments shifted by the chunk's offset.

    Raises:
        TranscriptionError: If the response carries no segments.
    """
    with chunk.path.open("rb") as handle:
        response = client.audio.transcriptions.create(
            file=handle,
            model=_MODEL,
            language=language,
            response_format="verbose_json",
        )

    raw = getattr(response, "segments", None)
    if not raw:
        # A chunk of pure silence legitimately yields nothing; a chunk of speech
        # yielding nothing means the server mis-handled the audio.
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return []
        raise TranscriptionError(
            f"{chunk.path.name}: text returned without segment timings"
        )

    # A segment with no text accounts for no speech, so it must not count as coverage.
    return [
        Segment(start=float(s.start), end=float(s.end), text=s.text).shifted(
            chunk.offset
        )
        for s in raw
        if s.text.strip()
    ]


def _judge_chunk(chunk: AudioChunk, segments: list[Segment]) -> float | None:
    """
    Measure a chunk's text density and reject speech that came back nearly empty.

    Args:
        chunk: The audio chunk that was transcribed.
        segments: What the server returned for it.

    Returns:
        Characters per second of the chunk's audio, or None for a chunk too short
        to judge.

    Raises:
        TranscriptionError: If a chunk loud enough to hold speech came back with
            almost no text — the signature of the server garbling the request.
        AudioError: If ffmpeg cannot measure the chunk's loudness.
    """
    if chunk.duration < _MIN_JUDGED_CHUNK_SECONDS:
        return None

    density = sum(len(s.text.strip()) for s in segments) / chunk.duration
    volume = mean_volume(chunk.path)
    log.debug("%s: %.1f chars/s at %.1f dB", chunk.path.name, density, volume)
    if density >= _MIN_CHUNK_CHARS_PER_SECOND:
        return density

    where = (
        f"{chunk.path.name} ({chunk.offset:.0f}-{chunk.offset + chunk.duration:.0f} s "
        f"of the recording)"
    )
    if volume > _SPEECH_VOLUME_DB:
        raise TranscriptionError(
            f"{where} is loud enough for speech ({volume:.0f} dB mean) but came back "
            f"as only {density:.1f} characters per second — the ASR server garbles "
            f"a request holding a long silence; look for one in this chunk that the "
            f"silence skipping missed"
        )
    log.warning(
        "%s came back as only %.1f characters per second, but at %.0f dB mean it "
        "may hold little speech",
        where,
        density,
        volume,
    )
    return density


def transcribe(
    chunks: list[AudioChunk], total_duration: float, settings: Settings
) -> Transcript:
    """
    Transcribe every chunk and merge the results into one transcript.

    Each chunk is judged as soon as it returns, so a garbled one stops the run
    before the rest of the recording is spent on the server.

    Args:
        chunks: Ordered audio chunks; long silences between them are skipped.
        total_duration: Length of the source audio in seconds.
        settings: Runtime configuration.

    Returns:
        The merged Transcript.

    Raises:
        TranscriptionError: If nothing was transcribed, a chunk of speech came back
            nearly empty, or coverage is implausibly low, which is the signature
            of the server silently dropping spans.
        AudioError: If ffmpeg cannot measure a chunk's loudness.
    """
    client = _client(settings)
    segments: list[Segment] = []
    densities: list[float] = []

    for index, chunk in enumerate(chunks, start=1):
        log.info(
            "transcribing chunk %d/%d (%.0f s at %.0f s)",
            index,
            len(chunks),
            chunk.duration,
            chunk.offset,
        )
        chunk_segments = _transcribe_chunk(client, chunk, settings.language)
        density = _judge_chunk(chunk, chunk_segments)
        if density is not None:
            densities.append(density)
        segments.extend(chunk_segments)

    if not segments:
        raise TranscriptionError("the ASR server returned no speech for this recording")

    sent = sum(chunk.duration for chunk in chunks)
    skipped = max(0.0, total_duration - sent)
    segments.sort(key=lambda s: s.start)
    transcript = Transcript(
        language=settings.language,
        duration=total_duration,
        segments=tuple(segments),
        skipped_seconds=skipped,
        min_chunk_density=min(densities, default=None),
    )

    coverage = transcript.coverage
    log.info(
        "transcribed %.0f s of %.0f s sent (%.0f%% coverage); %.0f s skipped as silence",
        transcript.covered_seconds,
        sent,
        coverage * 100,
        skipped,
    )
    if coverage < _MIN_COVERAGE:
        raise TranscriptionError(
            f"only {coverage:.0%} of the audio produced segments — the server likely "
            f"dropped spans; re-run or lower chunk_seconds"
        )

    # Over the audio sent, because skipped silence was never meant to yield text.
    density = sum(len(s.text) for s in transcript.segments) / sent
    log.info("transcript density %.1f chars/s of audio", density)
    if density < _MIN_CHARS_PER_SECOND:
        raise TranscriptionError(
            f"the transcript holds only {density:.1f} characters per second of audio — "
            f"the ASR model is producing near-silence; check that the server loaded "
            f"the fp32 weights and not the int8 ones"
        )

    return transcript
