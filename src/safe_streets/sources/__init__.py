"""Where footage comes from. safe-streets never reads a live stream: it
analyses a recorder's footage in batches, so a source is a recorder that can
say how far its recording reaches and export a stretch of it as a clip with
each frame's real time.

  folder   a directory of segment files (any NVR, or the bundled ffmpeg
           recorder), named by their start time
  protect  UniFi Protect, through its private API
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol


class NoFootage(Exception):
    """Nothing recorded in the window asked for."""


class Source(Protocol):
    connected: bool
    last_error: str | None

    async def ensure_connected(self) -> bool: ...

    async def recorded_until(self) -> datetime | None:
        """The end of the newest complete footage (UTC-aware), or None."""

    async def export(self, start: datetime, end: datetime, dest: Path, *, low_res: bool = False) -> datetime | None:
        """Write the footage covering [start, end] to dest. Returns the wall
        time of the clip's first frame when it isn't `start` (a folder clip
        begins at the keyframe at or before it), else None."""

    async def snapshot(self) -> bytes:
        """A current frame, as JPEG."""

    async def close(self) -> None: ...


def make_source(settings) -> Source:
    kind = settings.source.strip().lower()
    if kind == "folder":
        from .folder import FolderSource
        return FolderSource(settings)
    if kind == "protect":
        from .protect import Protect
        return Protect(settings)
    raise ValueError(f"SOURCE must be folder or protect, not {settings.source!r}")
