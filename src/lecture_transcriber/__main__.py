"""Command-line entry point for the lecture transcription pipeline."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import click

from lecture_transcriber.asr import TranscriptionError, transcribe
from lecture_transcriber.audio import (
    AudioError,
    extract_audio,
    probe_duration,
    split_audio,
)
from lecture_transcriber.config import Settings
from lecture_transcriber.drive import (
    DriveError,
    build_service,
    download,
    fetch_metadata,
)
from lecture_transcriber.models import Transcript
from lecture_transcriber.render import write_artifacts
from lecture_transcriber.summarize import SummaryError, summarize

log = logging.getLogger(__name__)


def _configure_logging(logs_dir: Path, verbose: bool) -> None:
    """
    Send logs to the console and to a per-run file under `logs_dir`.

    The file always records DEBUG regardless of `verbose`: a transcription run is
    long and expensive, so the detail needed to diagnose it must survive a run
    nobody thought to pass `-v` to.

    Args:
        logs_dir: Directory for run logs; created if missing.
        verbose: Whether the console should show debug-level records too.

    Raises:
        OSError: If the log directory or file cannot be created.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)

    logs_dir.mkdir(parents=True, exist_ok=True)
    # Local time, matching the timestamps inside the file, so the name of a log
    # lines up with the run you remember starting.
    started = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"{started}.log"
    to_file = logging.FileHandler(log_file, encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    to_file.setFormatter(formatter)

    # force=True because basicConfig is a no-op once anything else has touched the
    # root logger — an imported library doing so would silently cost us the log file.
    logging.basicConfig(level=logging.DEBUG, handlers=[console, to_file], force=True)
    # The Google client is chatty about discovery and retries at INFO. The HTTP
    # stacks are capped because their DEBUG records dump request headers, which
    # would write the service-account bearer token into a file on disk.
    # httpx2/httpcore2 are the real logger names for the installed packages; the
    # unsuffixed spellings matched nothing and left 50 lines of HTTP DEBUG per run.
    for noisy in (
        "googleapiclient",
        "google_auth_httplib2",
        "google",
        "urllib3",
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "openai",
        "anthropic",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log.info("logging to %s", log_file)


def _log_settings(settings: Settings, file_id: str) -> None:
    """
    Report the effective configuration before any work starts.

    Args:
        settings: Runtime configuration.
        file_id: The resolved Drive file ID, which may override `settings.file_id`.
    """
    log.info("settings:")
    for name, value in settings.describe().items():
        log.info("  %-24s %s", name, value)
    log.info("  %-24s %s", "file_id", file_id)
    log.info("  %-24s %s", "job_dir", settings.job_dir(file_id))


def _load_or_transcribe(job_dir: Path, media: Path, settings: Settings) -> Transcript:
    """
    Return a cached transcript, or produce one from the media file.

    Transcription is by far the most expensive stage, so a failure in
    summarization or upload must never trigger a re-run.

    Args:
        job_dir: Per-lecture working directory.
        media: The downloaded video or audio file.
        settings: Runtime configuration.

    Returns:
        The lecture transcript.

    Raises:
        TranscriptionError: If transcription fails or coverage is implausible.
        AudioError: If ffmpeg or ffprobe fails.
    """
    transcript_path = job_dir / "transcript.json"
    if transcript_path.exists():
        log.info("reusing cached transcript")
        return Transcript.from_json(transcript_path)

    audio_path = job_dir / "audio.wav"
    if not audio_path.exists():
        extract_audio(media, audio_path)

    duration = probe_duration(audio_path)
    chunks = split_audio(
        audio_path,
        job_dir / "chunks",
        settings.chunk_seconds,
        settings.chunk_search_window,
    )

    transcript = transcribe(chunks, duration, settings)
    transcript.to_json(transcript_path)
    return transcript


def resolve_file_id(argument: str | None, configured: str) -> str:
    """
    Decide which recording to process.

    Args:
        argument: The file ID given on the command line, if any.
        configured: The `FILE_ID` setting from the environment or `.env`.

    Returns:
        The Drive file ID to process.

    Raises:
        click.UsageError: If neither source supplies an ID.
    """
    file_id = (argument or configured).strip()
    if not file_id:
        raise click.UsageError(
            "no recording specified — set FILE_ID in .env, or pass one as an argument"
        )
    return file_id


@click.command()
@click.argument("file_id", required=False)
@click.option("--verbose", "-v", is_flag=True, help="Emit debug logging.")
@click.option(
    "--refresh-summary",
    is_flag=True,
    help="Re-run summarization even if a cached summary exists.",
)
def main(file_id: str | None, verbose: bool, refresh_summary: bool) -> None:
    """
    Transcribe and summarize a Google Drive recording into docs/.

    FILE_ID defaults to the `FILE_ID` setting in .env; pass one to override it.
    """
    settings = Settings()
    _configure_logging(settings.logs_dir, verbose)
    file_id = resolve_file_id(file_id, settings.file_id)
    job_dir = settings.job_dir(file_id)
    _log_settings(settings, file_id)

    try:
        service = build_service(settings)
        source = fetch_metadata(service, file_id)
        log.info("resolved %s to %s (%s)", file_id, source.drive_path, source.mime_type)

        media = download(service, source, job_dir / f"source{Path(source.name).suffix}")
        transcript = _load_or_transcribe(job_dir, media, settings)

        cached_summary = job_dir / "summary.md"
        if cached_summary.exists() and not refresh_summary:
            log.info("reusing cached summary")
            summary_md = cached_summary.read_text(encoding="utf-8")
        else:
            summary_md = summarize(transcript, source.name, settings)
            cached_summary.write_text(summary_md, encoding="utf-8")

        summary_path, transcript_path = write_artifacts(
            summary_md, transcript, source, settings.docs_dir, settings.anthropic_model
        )
    except (DriveError, AudioError, TranscriptionError, SummaryError) as exc:
        # These are all expected, actionable failures — a traceback adds nothing.
        log.error("%s", exc)
        sys.exit(1)

    click.echo(summary_path)
    click.echo(transcript_path)


if __name__ == "__main__":
    main()
