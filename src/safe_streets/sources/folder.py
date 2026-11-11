"""A folder of recorded segments, named by their start time
(SOURCE_PATTERN, a strftime pattern, local time in TZ unless it has %z).

A segment modified within SOURCE_SETTLE_S is still being written and is
ignored. A clip is the segments overlapping the window, stream-copied (never
re-encoded) from the keyframe at or before the window's start, with every
frame's time kept: segments whose pts carry on from one to the next (ffmpeg's
`-reset_timestamps 0`) are joined on their pts; segments that start again
from 0 are placed by their file names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

from . import NoFootage

log = logging.getLogger(__name__)

# A segment whose first pts is within this of where the last one's ended
# (judged against their file names) continues it.
CONTINUES_S = 2.0
# Output timestamps start here, so a B-frame stream's dts stay positive.
BASE_S = 10.0


@dataclass(frozen=True)
class Segment:
    path: Path
    start: datetime  # from the file name, UTC
    first_pts: float  # seconds
    last_pts: float
    frame_s: float  # the median frame interval

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.last_pts - self.first_pts + self.frame_s)


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    """A subprocess that can't hang: a stale network mount can leave ffmpeg
    waiting for ever, so every call has a timeout (decision 0012)."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{args[0]} timed out after {timeout:g} s") from None


def _probe(path: Path, timeout: float = 120.0) -> tuple[float, float, float] | None:
    """(first pts, last pts, median frame interval) of the video stream, from
    its packets (no decoding)."""
    try:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                    "-of", "json", str(path)], timeout)
    except RuntimeError as e:
        log.warning("%s: %s", path.name, e)
        return None
    if out.returncode != 0:
        return None
    pts = sorted(float(p["pts_time"]) for p in json.loads(out.stdout or "{}").get("packets", []) if "pts_time" in p)
    if not pts:
        return None
    gaps = sorted(b - a for a, b in zip(pts, pts[1:]))
    return pts[0], pts[-1], gaps[len(gaps) // 2] if gaps else 0.0


def _ffmpeg(*args: str, timeout: float = 120.0) -> None:
    r = _run(["ffmpeg", "-v", "error", "-y", *args], timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-300:]}")


def _keyframe_at_or_before(path: Path, t: float, timeout: float = 120.0) -> float:
    """The pts of the last keyframe at or before t (or the first frame)."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                "-of", "json", str(path)], timeout)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[-300:]}")
    keys = sorted(float(p["pts_time"]) for p in json.loads(out.stdout).get("packets", [])
                  if "K" in p.get("flags", "") and "pts_time" in p)
    before = [k for k in keys if k <= t + 1e-6]
    return before[-1] if before else keys[0]


class FolderSource:
    def __init__(self, settings, clock=time.time):
        self.s = settings
        self.dir = Path(settings.source_dir) if settings.source_dir else None
        self.clock = clock
        self.connected = False
        self.last_error: str | None = None
        self._cache: dict[tuple[str, int, float], tuple[float, float, float] | None] = {}

    async def ensure_connected(self) -> bool:
        if self.dir is not None and self.dir.is_dir():
            self.connected, self.last_error = True, None
        else:
            self.connected, self.last_error = False, f"SOURCE_DIR {self.dir} isn't a folder"
        return self.connected

    def _start_of(self, name: str) -> datetime | None:
        try:
            dt = datetime.strptime(name, self.s.source_pattern)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(self.s.tz))
        return dt.astimezone(timezone.utc)

    def segments(self) -> list[Segment]:
        """The complete segments, oldest first."""
        if self.dir is None or not self.dir.is_dir():
            return []
        settled = self.clock() - self.s.source_settle_s
        out = []
        for p in self.dir.iterdir():
            st = p.stat() if p.is_file() else None
            if st is None or st.st_mtime > settled:
                continue
            start = self._start_of(p.name)
            if start is None:
                continue
            key = (p.name, st.st_size, st.st_mtime)
            if key not in self._cache:
                self._cache[key] = _probe(p, self.s.export_timeout_seconds)
            probed = self._cache[key]
            if probed is not None:
                out.append(Segment(p, start, *probed))
        return sorted(out, key=lambda s: s.start)

    async def recorded_until(self) -> datetime | None:
        segs = await asyncio.to_thread(self.segments)
        return segs[-1].end if segs else None

    async def export(self, start: datetime, end: datetime, dest: Path, *, low_res: bool = False) -> datetime | None:
        """low_res is ignored: the scan downsizes as it decodes."""
        return await asyncio.to_thread(self._export, start, end, dest)

    def _export(self, start: datetime, end: datetime, dest: Path) -> datetime | None:
        segs = [s for s in self.segments() if s.start < end and s.end > start]
        if not segs:
            raise NoFootage(f"nothing recorded from {start.isoformat()} to {end.isoformat()}")
        first = segs[0]
        # Where each segment's pts sit on one clock, in seconds from the first segment's first pts.
        rel = [0.0]
        for prev, cur in zip(segs, segs[1:]):
            by_name = (cur.start - first.start).total_seconds()
            by_pts = rel[-1] + (cur.first_pts - prev.first_pts)
            continues = abs((cur.first_pts - prev.first_pts) - (cur.start - prev.start).total_seconds()) <= CONTINUES_S
            rel.append(by_pts if continues else by_name)
        want = max(0.0, (start - first.start).total_seconds())
        key = _keyframe_at_or_before(first.path, first.first_pts + want, self.s.export_timeout_seconds)
        clip_start = first.start + timedelta(seconds=key - first.first_pts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            parts = []
            for i, (seg, r) in enumerate(zip(segs, rel)):
                ts = Path(tmp) / f"{i}.ts"
                # -ss counts from the file's start, even with -copyts.
                seek = ["-ss", f"{key - seg.first_pts:.6f}"] if i == 0 and key > seg.first_pts else []
                _ffmpeg("-copyts", *seek, "-i", str(seg.path), "-map", "0:v:0", "-c", "copy",
                        "-output_ts_offset", f"{BASE_S + r - seg.first_pts:.6f}", "-f", "mpegts", str(ts),
                        timeout=self.s.export_timeout_seconds)
                parts.append(str(ts))
            length = (end - clip_start).total_seconds() + 1.0
            _ffmpeg("-i", "concat:" + "|".join(parts), "-c", "copy", "-t", f"{length:.3f}", "-movflags", "+faststart",
                    str(dest), timeout=self.s.export_timeout_seconds)
        return None if clip_start == start else clip_start

    async def snapshot(self) -> bytes:
        return await asyncio.to_thread(self._snapshot)

    def _snapshot(self) -> bytes:
        segs = self.segments()
        if not segs:
            raise NoFootage("no complete segment yet")
        cap = cv2.VideoCapture(str(segs[-1].path))
        last = None
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                last = f
        finally:
            cap.release()
        if last is None:
            raise NoFootage(f"{segs[-1].path.name} has no frames")
        ok, buf = cv2.imencode(".jpg", last, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()

    async def close(self) -> None:
        pass
