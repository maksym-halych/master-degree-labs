"""Transcription against a self-hosted Parakeet server (OpenAI-compatible API)."""

import logging

from openai import APIConnectionError, APIStatusError, OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lecture_transcriber.audio import AudioChunk
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

    return [
        Segment(start=float(s.start), end=float(s.end), text=s.text).shifted(
            chunk.offset
        )
        for s in raw
    ]


def transcribe(
    chunks: list[AudioChunk], total_duration: float, settings: Settings
) -> Transcript:
    """
    Transcribe every chunk and merge the results into one transcript.

    Args:
        chunks: Ordered audio chunks covering the full recording.
        total_duration: Length of the source audio in seconds.
        settings: Runtime configuration.

    Returns:
        The merged Transcript.

    Raises:
        TranscriptionError: If nothing was transcribed, or coverage is implausibly
            low, which is the signature of the server silently dropping spans.
    """
    client = _client(settings)
    segments: list[Segment] = []

    for index, chunk in enumerate(chunks, start=1):
        log.info(
            "transcribing chunk %d/%d (%.0f s)", index, len(chunks), chunk.duration
        )
        segments.extend(_transcribe_chunk(client, chunk, settings.language))

    if not segments:
        raise TranscriptionError("the ASR server returned no speech for this recording")

    segments.sort(key=lambda s: s.start)
    transcript = Transcript(
        language=settings.language,
        duration=total_duration,
        segments=tuple(segments),
    )

    coverage = transcript.covered_seconds / total_duration if total_duration else 0.0
    log.info(
        "transcribed %.0f s of %.0f s (%.0f%% coverage)",
        transcript.covered_seconds,
        total_duration,
        coverage * 100,
    )
    if coverage < _MIN_COVERAGE:
        raise TranscriptionError(
            f"only {coverage:.0%} of the audio produced segments — the server likely "
            f"dropped spans; re-run or lower chunk_seconds"
        )

    density = sum(len(s.text) for s in transcript.segments) / total_duration
    log.info("transcript density %.1f chars/s of audio", density)
    if density < _MIN_CHARS_PER_SECOND:
        raise TranscriptionError(
            f"the transcript holds only {density:.1f} characters per second of audio — "
            f"the ASR model is producing near-silence; check that the server loaded "
            f"the fp32 weights and not the int8 ones"
        )

    return transcript
