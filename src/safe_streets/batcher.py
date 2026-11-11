"""The batch loop: every BATCH_INTERVAL_SECONDS (or on "Check now"), queue
the source's new events (a recorder's motion events, or windows of
continuous recording the scan found movement in), with a widened window,
then process them one at a time.

Busy (2026-11-06): CPU pressure, not the load average, when the kernel
reports it. On this 4-core box the 1-min load sat at 3-5 all day, partly
tasks waiting on disk and the GPU, so a MAX_LOAD of 4.0 held 222 of 250
events back for ~60 min each. /proc/pressure/cpu's `some avg60` (the share
of the last minute some task waited for a CPU; host-wide inside the
container) ran 6-40% on normal days, so over MAX_CPU_PRESSURE (60%) means
real contention. MAX_LOAD is the fallback where PSI can't be read.

Pacing (2026-11-03): the load is checked before every event, not once per
batch. With a backlog (a reanalyse or a backfill) and a quiet box, the next
batch starts at once; when the box is busy, the batch stops after the event
in hand and the loop looks again every LOAD_RETRY_SECONDS. Before, a fixed
5-minute gap and whole skipped batches held a backlog to ~150 events an hour
on a box that was often idle."""

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from .geometry import calibration_from_row, image_shift
from .pipeline import process_event
from .backfill import WARMUP_S, chunks
from .scan import find_windows
from .sources import NoFootage
from .vision import crop_for

log = logging.getLogger(__name__)

# Px in the 640x360 snapshot. A still camera read up to ~8 on the reference
# install (low sun, shadows); a real move read ~60.
MOVED_PX, BACK_PX = 15.0, 3.0
# Unanalysed events this recent are live: done oldest first, ahead of any
# backlog, so the bot posts in time order (db.next_events). An hour covers
# the ~30 min load can hold a batch back.
LIVE_WINDOW_S = 3600
PSI_CPU = Path("/proc/pressure/cpu")


def cpu_pressure(path: Path = PSI_CPU) -> float | None:
    """The kernel's CPU pressure, `some avg60` in percent, or None where it
    can't be read (no PSI, or not Linux)."""
    try:
        for line in path.read_text().splitlines():
            if line.startswith("some "):
                fields = dict(f.split("=", 1) for f in line.split()[1:])
                return float(fields["avg60"])
    except (OSError, ValueError, KeyError):
        pass
    return None


def camera_mask(shape, road) -> np.ndarray:
    """Where to look for the camera moving: everywhere but the road (cars
    aren't the camera moving) and the timestamp overlay. The road outline is
    in 1080p px; Protect's snapshots come back 640x360, so it is scaled."""
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    x0, y0, x1, y1 = crop_for(road)
    sx, sy = w / 1920, h / 1080
    mask[int(y0 * sy):int(np.ceil(y1 * sy)), int(x0 * sx):int(np.ceil(x1 * sx))] = 0
    mask[: int(0.05 * h)] = 0
    return mask


class Batcher:
    def __init__(self, settings, db, source, vision, load_fn=os.getloadavg, clock=time.time,
                 pressure_fn: Callable[[], float | None] = lambda: None):
        self.s = settings
        self.db = db
        self.source = source
        self.vision = vision
        self.load_fn = load_fn
        # main.py passes cpu_pressure; None (the default) gates on load alone.
        self.pressure_fn = pressure_fn
        self.clock = clock
        self.running = False
        # Called with the event id after an event's first live analysis found cars.
        self.on_first_analysis: Callable[[str], Awaitable[None]] | None = None
        # Called when a batch ends, and before it reaches a reanalysis: the
        # poster holds live cars until then, to post them in time order.
        self.on_batch_end: Callable[[], None] | None = None
        # HOOKS (hooks.py): called after each batch with its new passes.
        self.hooks: list = []
        self.next_run_at = clock() + 5
        self.last_run: dict = {}
        self._started = clock()
        self._skips = 0
        self._busy_since: float | None = None
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_camera_check = float("-inf")
        self._last_prune = float("-inf")

    # loop ---------------------------------------------------------------

    def start(self) -> None:
        self._started = self.clock()
        self._task = asyncio.create_task(self._loop())

    def age(self) -> float:
        """Seconds since a batch last finished (a busy skip counts), or since
        the start if none has yet: a stall shows as this growing (/healthz)."""
        at = self.last_run.get("at")
        last = datetime.fromisoformat(at).timestamp() if at else float("-inf")
        return self.clock() - max(last, self._started)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    def trigger(self) -> bool:
        if self.running:
            return False
        self._wake.set()
        return True

    async def _loop(self) -> None:
        self.last_run = await self.db.get_state("last_run") or {}
        while True:
            manual = False
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=max(0.0, self.next_run_at - self.clock()))
                manual = True
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            r: dict = {}
            try:
                r = await self.run_once(manual=manual)
            except Exception:  # noqa: BLE001  (the loop must survive anything)
                log.exception("batch failed")
            self.next_run_at = self.clock() + self._delay(r)

    def _delay(self, r: dict) -> float:
        """Seconds to the next batch: none while a backlog waits on a quiet
        box, LOAD_RETRY_SECONDS while it waits on a busy one, else the interval."""
        if r.get("error") or not r.get("queued"):
            return self.s.batch_interval_seconds
        if r.get("skipped") or r.get("paused"):
            return self.s.load_retry_seconds
        return 0

    def _gauge(self) -> dict:
        """What the busy check read: CPU pressure where the kernel reports
        it, else the 1-min load."""
        cpu = self.pressure_fn()
        if cpu is not None:
            return {"cpu": round(cpu, 1)}
        return {"load": round(self.load_fn()[0], 2)}

    def _over(self, gauge: dict) -> bool:
        if "cpu" in gauge:
            return gauge["cpu"] > self.s.max_cpu_pressure
        return gauge["load"] > self.s.max_load

    def _busy(self) -> bool:
        """The box is busy (CPU pressure over MAX_CPU_PRESSURE, or the load
        over MAX_LOAD where pressure can't be read) and hasn't yet held work
        back for MAX_SKIPS intervals (~30 min). Past that, a batch runs anyway.
        Without MAX_CPU_PRESSURE nothing is read and the box is never busy."""
        if self.s.max_cpu_pressure is None or not self._over(self._gauge()):
            self._busy_since = None
            return False
        if self._busy_since is None:
            self._busy_since = self.clock()
        return self.clock() - self._busy_since < self.s.max_skips * self.s.batch_interval_seconds

    # one batch ----------------------------------------------------------

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), timezone.utc)

    async def _finish(self, summary: dict, quiet: bool = False) -> dict:
        summary["at"] = self._now().isoformat()
        (log.debug if quiet else log.info)("batch %s", " ".join(f"{k}={v}" for k, v in summary.items() if k != "at"))
        self.last_run = summary
        await self.db.set_state("last_run", summary)
        return summary

    async def run_once(self, manual: bool) -> dict:
        if self.running:
            return {"skipped": "running"}
        if not manual and self._busy():
            # Retried every LOAD_RETRY_SECONDS: log the first skip of a run only.
            self._skips += 1
            queued = (await self.db.event_counts()).get("queued", 0)
            return await self._finish({"skipped": "load", **self._gauge(), "queued": queued,
                                       "manual": manual}, quiet=self._skips > 1)
        # Held back ~30 min by load: this batch runs to the end regardless.
        forced = not manual and self._busy_since is not None
        self._skips = 0
        paused = False
        self.running = True
        started = time.monotonic()
        try:
            if not await self.source.ensure_connected():
                return await self._finish({"error": self.source.last_error or "not connected", "manual": manual})
            await self.check_camera()
            windows = await self.queue_events()
            polled = self.clock()
            processed = cars = 0
            uncalibrated = False
            fresh: list[str] = []  # events whose first live analysis found cars: the hooks' passes
            while processed < self.s.max_per_run:
                if processed and not manual and not forced and self._busy():
                    paused = True
                    break
                # A batch working through a backlog can run for many minutes;
                # keep picking up new motion events so live cars go next.
                if self.clock() - polled >= self.s.batch_interval_seconds:
                    windows += await self.queue_events()
                    polled = self.clock()
                live_since = datetime.fromtimestamp(self.clock() - LIVE_WINDOW_S, timezone.utc).isoformat()
                evs = await self.db.next_events(1, live_since=live_since)
                if not evs:
                    break
                cal = await self._calibration()
                if cal is None:  # nothing is analysed until the camera is calibrated
                    uncalibrated = True
                    break
                first = (await self.db.get_event(evs[0]["id"]))["first_done_at"] is None
                if not first:
                    self._end_batch()
                r = await process_event(evs[0], source=self.source, vision=self.vision, cal=cal,
                                        settings=self.s, db=self.db)
                if first and r.get("status") in ("done", "no_vehicle"):
                    await self.db.set_event(evs[0]["id"], first_done_at=self._now().isoformat())
                    # Reanalyses requeue with first_done_at kept, and backfills
                    # (scan-) are old footage: neither may reach the poster.
                    if r["status"] == "done" and not evs[0]["id"].startswith("scan-"):
                        fresh.append(evs[0]["id"])
                    if r["status"] == "done" and self.on_first_analysis and not evs[0]["id"].startswith("scan-"):
                        try:
                            await self.on_first_analysis(evs[0]["id"])
                        except Exception:  # noqa: BLE001
                            log.exception("first-analysis hook failed for %s", evs[0]["id"])
                processed += 1
                cars += r["passes"]
            if self.hooks and fresh:
                from .hooks import PassRecord, run_hooks
                rows = [r for eid in fresh for r in await self.db.passes_for_event(eid)]
                await run_hooks(self.hooks, [PassRecord.from_row(r) for r in rows])
            await self.prune_clips()
            caught = {}
            queued = (await self.db.event_counts()).get("queued", 0)
            summary = {"windows": windows, "processed": processed, "cars": cars, "queued": queued,
                       "seconds": round(time.monotonic() - started, 1), "manual": manual, **caught}
            if uncalibrated:
                summary["uncalibrated"] = True
            if paused:
                summary |= {"paused": "load", **self._gauge()}
            return await self._finish(summary)
        finally:
            self._end_batch()
            if forced:
                self._busy_since = None
            self.running = False

    def _end_batch(self) -> None:
        if self.on_batch_end is None:
            return
        try:
            self.on_batch_end()
        except Exception:  # noqa: BLE001
            log.exception("batch-end hook failed")

    async def _calibration(self):
        row = await self.db.active_calibration()
        return calibration_from_row(row) if row else None

    async def queue_events(self) -> int:
        """Queue new windows of road motion. A source with motion events of
        its own (Protect) is asked for them; any other is scanned: its
        low-res footage from the cursor to where its recording reaches."""
        if hasattr(self.source, "list_events"):
            return await self._queue_source_events()
        return await self._scan_recording()

    async def _scan_recording(self) -> int:
        """Scan [cursor, recorded_until] in chunks (backfill.chunks) and queue
        each window of motion inside the calibration's road. A window still
        open at the end of the recording waits for the next batch, so a car
        isn't cut in two; the cursor stops just before it. Without a
        calibration there's no road to watch: nothing is queued."""
        cal = await self._calibration()
        until = await self.source.recorded_until()
        if cal is None or until is None:
            return 0
        raw = await self.db.get_state("scan_cursor")
        cursor = datetime.fromisoformat(raw) if raw else until - timedelta(hours=self.s.lookback_hours)
        found = 0
        tmp = self.s.data_dir / "scan"
        for a, b in chunks(cursor, until):
            clip = tmp / f"live-{a:%Y%m%dT%H%M%S}.mp4"
            try:
                got = await self.source.export(a, b, clip, low_res=True)
                wins = await asyncio.to_thread(find_windows, clip, cal.road, warmup_s=WARMUP_S, mask=cal.mask)
            except NoFootage:
                await self.db.set_state("scan_cursor", b.isoformat())
                continue
            finally:
                clip.unlink(missing_ok=True)
            base = got or a
            hold = None
            for s0, s1 in wins:
                ws, we = base + timedelta(seconds=s0), base + timedelta(seconds=s1)
                if b >= until and we >= b - timedelta(seconds=1) and (we - ws).total_seconds() < self.s.clip_max_seconds:
                    hold = ws  # still open at the recording's end
                    break
                if await self.db.event_covering(ws.isoformat(), we.isoformat()):
                    continue
                if await self.db.upsert_event(f"seg-{ws:%Y%m%dT%H%M%S}", ws.isoformat(), we.isoformat()):
                    found += 1
            # The next chunk starts WARMUP_S before the cursor; motion before the cursor is ignored.
            await self.db.set_state("scan_cursor", (hold - timedelta(seconds=1) if hold else b).isoformat())
            if hold:
                break
        return found

    async def _queue_source_events(self) -> int:
        """Queue ended motion events since the cursor (with 5 minutes of
        overlap, so an event that ended late isn't lost). Each window is the
        event widened by CLIP_PAD_SECONDS each side, capped at CLIP_MAX_SECONDS.
        An event whose window an existing event already contains is skipped."""
        s = self.s
        now = self._now()
        raw = await self.db.get_state("event_cursor")
        since = (datetime.fromisoformat(raw) - timedelta(minutes=5) if raw
                 else now - timedelta(hours=s.lookback_hours))
        pad = timedelta(seconds=s.clip_pad_seconds)
        found = 0
        for eid, start, end in await self.source.list_events(since, now):
            ws = start - pad
            we = min(end + pad, ws + timedelta(seconds=s.clip_max_seconds))
            # An event inside another's window would analyse the same video
            # again (triage step 4). Partial overlaps stay: cutting a window
            # could split a car.
            if await self.db.event_covering(ws.isoformat(), we.isoformat()):
                continue
            if await self.db.upsert_event(eid, ws.isoformat(), we.isoformat()):
                found += 1
        await self.db.set_state("event_cursor", now.isoformat())
        return found

    async def check_camera(self) -> None:
        """About once an hour: compare a snapshot with the calibration's
        reference frame. A shift pauses speeds; shifting back resumes them."""
        if self.clock() - self._last_camera_check < self.s.camera_check_seconds:
            return
        row = await self.db.active_calibration()
        if row is None:
            return
        self._last_camera_check = self.clock()
        try:
            snap = await self.source.snapshot()
            cur = cv2.imdecode(np.frombuffer(snap, np.uint8), cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
            if cur is None or ref is None:
                return
            if cur.shape != ref.shape:
                cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]))
            shift = image_shift(ref, cur, camera_mask(ref.shape, calibration_from_row(row).road))
        except Exception:  # noqa: BLE001
            log.exception("camera check failed")
            return
        await self.db.set_state("camera_shift", {"px": shift, "at": self._now().isoformat()})
        if shift is None:
            return
        if shift > MOVED_PX and row["status"] != "stale":
            await self.db.set_state("status_before_stale", row["status"])
            await self.db.set_calibration_status(row["id"], "stale")
            log.warning("camera moved %.1f px since calibration %s: speeds paused", shift, row["id"])
        elif shift < BACK_PX and row["status"] == "stale":
            await self.db.set_calibration_status(row["id"], await self.db.get_state("status_before_stale") or "active")

    async def prune_clips(self) -> None:
        if self.clock() - self._last_prune < 86400:
            return
        self._last_prune = self.clock()
        for ev in await self.db.old_clip_events(self.s.clip_retention_days):
            Path(ev["clip_path"]).unlink(missing_ok=True)
            await self.db.set_event(ev["id"], clip_path=None)
        for p in await self.db.old_hero_passes(self.s.clip_retention_days):
            for col in ("hero_path", "card_path"):
                if p[col]:
                    Path(p[col]).unlink(missing_ok=True)
            await self.db.set_pass(p["id"], hero_path=None, card_path=None)
        # MOTION_GATE=shadow's kept clips (pipeline._shadow_gate) aren't in the DB.
        cutoff = self.clock() - self.s.clip_retention_days * 86400
        for f in (self.s.data_dir / "shadow").glob("*.mp4"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
