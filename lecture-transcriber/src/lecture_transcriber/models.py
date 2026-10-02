"""Domain objects passed between pipeline stages."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Segment:
    """A timed span of transcribed speech."""

    start: float
    end: float
    text: str

    def shifted(self, offset: float) -> "Segment":
        """
        Return a copy translated along the timeline.

        Args:
            offset: Seconds to add to both boundaries.

        Returns:
            A new Segment positioned relative to the full recording.
        """
        return Segment(start=self.start + offset, end=self.end + offset, text=self.text)


@dataclass(frozen=True, slots=True)
class Transcript:
    """A complete lecture transcript with segment-level timings."""

    language: str
    duration: float
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        """The transcript as plain prose, without timestamps."""
        return " ".join(
            segment.text.strip() for segment in self.segments if segment.text.strip()
        )

    @property
    def covered_seconds(self) -> float:
        """Total audio duration actually accounted for by segments."""
        return sum(segment.end - segment.start for segment in self.segments)

    def to_json(self, path: Path) -> None:
        """
        Persist the transcript so a later stage can never force a re-transcription.

        Args:
            path: Destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text} for s in self.segments
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: Path) -> "Transcript":
        """
        Load a transcript previously written by `to_json`.

        Args:
            path: The file to read.

        Returns:
            The reconstructed Transcript.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid transcript JSON.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                language=payload["language"],
                duration=float(payload["duration"]),
                segments=tuple(
                    Segment(
                        start=float(s["start"]), end=float(s["end"]), text=s["text"]
                    )
                    for s in payload["segments"]
                ),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path} is not a valid transcript file") from exc


@dataclass(frozen=True, slots=True)
class LectureSource:
    """Metadata about the Drive video being processed."""

    file_id: str
    name: str
    mime_type: str
    size_bytes: int | None
    created_time: str | None
    # Ancestor folder names, outermost first. Truncated at the highest folder the
    # credentials can actually read, so it may be shorter than the real Drive path.
    folders: tuple[str, ...]
    path_complete: bool

    @property
    def stem(self) -> str:
        """The recording's name without its file extension."""
        return Path(self.name).stem

    @property
    def drive_path(self) -> str:
        """
        The recording's location in Drive, as a slash-separated path.

        A leading `.../` marks a path truncated by permissions, so a surprising
        `docs/` destination can be traced back to unreadable ancestor folders.
        """
        prefix = "" if self.path_complete else ".../"
        return prefix + "/".join((*self.folders, self.name))

    @property
    def recorded_date(self) -> str | None:
        """
        The lecture date.

        Prefers a `YYYY-MM-DD` prefix in the filename over Drive's own timestamp,
        because a re-upload changes the latter but not the former.

        Returns:
            An ISO date, or None if neither source supplies one.
        """
        match = re.match(r"(\d{4}-\d{2}-\d{2})", self.name)
        if match:
            return match.group(1)
        return self.created_time[:10] if self.created_time else None
