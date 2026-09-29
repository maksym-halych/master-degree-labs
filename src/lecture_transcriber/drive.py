"""Read-only Google Drive access: authentication, path resolution, download."""

import logging
from pathlib import Path
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from lecture_transcriber.config import Settings
from lecture_transcriber.models import LectureSource

log = logging.getLogger(__name__)

# Read-only is the whole surface: artifacts are written to this repo, not to Drive.
# A service account has no storage quota, so it could not create Drive files anyway.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_FOLDER_MIME = "application/vnd.google-apps.folder"

# Walking parents is unbounded in principle; a real Drive hierarchy is a few levels
# deep, so this only guards against a cycle or a pathological tree.
_MAX_DEPTH = 20


class DriveError(RuntimeError):
    """Raised when a Drive operation fails."""


def build_service(settings: Settings) -> Any:
    """
    Build a Drive client authenticated as the service account.

    Args:
        settings: Runtime configuration.

    Returns:
        An authorised read-only Drive v3 service.

    Raises:
        DriveError: If the key file is missing or malformed.
    """
    key_path = settings.google_service_account
    if not key_path.exists():
        raise DriveError(
            f"missing service account key at {key_path} — download one from Google Cloud "
            f"Console and share the Drive folder with its client_email"
        )
    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(key_path), scopes=SCOPES
        )
    except (ValueError, KeyError) as exc:
        raise DriveError(
            f"{key_path} is not a valid service account key: {exc}"
        ) from exc

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _get(service: Any, file_id: str, fields: str) -> dict[str, Any]:
    """
    Fetch one file's metadata.

    Args:
        service: An authorised Drive service.
        file_id: The Drive file ID.
        fields: Partial-response field list.

    Returns:
        The metadata mapping.

    Raises:
        HttpError: Propagated so callers can distinguish "not shared" from other faults.
    """
    return (
        service.files()
        .get(fileId=file_id, fields=fields, supportsAllDrives=True)
        .execute()
    )


def _resolve_folders(
    service: Any, payload: dict[str, Any]
) -> tuple[tuple[str, ...], bool]:
    """
    Reconstruct the chain of folders containing a file.

    The service account only sees what has been shared with it, so the walk stops at
    the highest shared ancestor rather than failing. A partial path is still useful;
    the caller is told whether it reached the top.

    Args:
        service: An authorised Drive service.
        payload: Metadata of the file to start from, including `parents`.

    Returns:
        Ancestor folder names outermost-first, and whether the walk reached a true root.
    """
    names: list[str] = []
    parents = payload.get("parents") or []
    complete = not parents  # A file with no parent is already at a root.
    seen: set[str] = set()

    current = parents[0] if parents else None
    for _ in range(_MAX_DEPTH):
        if current is None or current in seen:
            break
        seen.add(current)
        try:
            folder = _get(service, current, "id,name,parents,mimeType")
        except HttpError as exc:
            # 403/404 here means the ancestor simply is not shared — expected, not fatal.
            log.info(
                "stopping path walk: ancestor %s is not accessible (%s)",
                current,
                exc.status_code,
            )
            break

        names.append(folder.get("name", current))
        next_parents = folder.get("parents") or []
        if not next_parents:
            complete = True
            break
        current = next_parents[0]
    else:
        log.warning("folder walk hit the depth limit; path may be truncated")

    names.reverse()
    return tuple(names), complete


def fetch_metadata(service: Any, file_id: str) -> LectureSource:
    """
    Look up a recording's metadata and its position in the Drive hierarchy.

    Args:
        service: An authorised Drive service.
        file_id: The Drive file ID.

    Returns:
        The recording's metadata.

    Raises:
        DriveError: If the file is missing, not shared, or not a recording.
    """
    try:
        payload = _get(service, file_id, "id,name,mimeType,size,parents,createdTime")
    except HttpError as exc:
        if exc.status_code in (403, 404):
            raise DriveError(
                f"cannot read Drive file {file_id} — share it with the service account's "
                f"client_email, or check the ID"
            ) from exc
        raise DriveError(f"cannot read Drive file {file_id}: {exc}") from exc

    mime_type: str = payload["mimeType"]
    if not mime_type.startswith(("video/", "audio/")):
        raise DriveError(
            f"{payload['name']} is {mime_type}, not a video or audio recording"
        )

    folders, complete = _resolve_folders(service, payload)
    if not complete:
        log.warning(
            "Drive path is partial (%s) — share the top-level folder to mirror it fully",
            "/".join(folders) or "<none>",
        )

    size = payload.get("size")
    return LectureSource(
        file_id=payload["id"],
        name=payload["name"],
        mime_type=mime_type,
        size_bytes=int(size) if size else None,
        created_time=payload.get("createdTime"),
        folders=folders,
        path_complete=complete,
    )


def download(service: Any, source: LectureSource, destination: Path) -> Path:
    """
    Stream a Drive file to local disk.

    Lecture recordings run to gigabytes, so the download is chunked to disk rather
    than buffered in memory.

    Args:
        service: An authorised Drive service.
        source: Metadata of the file to download.
        destination: Target path; parent directories are created.

    Returns:
        The destination path.

    Raises:
        DriveError: If the download fails.
    """
    if destination.exists():
        log.info("reusing cached download %s", destination.name)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = service.files().get_media(fileId=source.file_id, supportsAllDrives=True)

    try:
        with partial.open("wb") as handle:
            downloader = MediaIoBaseDownload(
                handle, request, chunksize=16 * 1024 * 1024
            )
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    log.info(
                        "downloading %s: %d%%",
                        source.name,
                        int(status.progress() * 100),
                    )
    except HttpError as exc:
        partial.unlink(missing_ok=True)
        raise DriveError(f"download of {source.name} failed: {exc}") from exc

    # Rename only on success so an interrupted run never leaves a truncated file
    # that the cache check above would happily reuse.
    partial.rename(destination)
    return destination
