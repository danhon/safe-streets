"""Backfill: find road motion in the continuous recording for a stretch of
time, and queue each window as an event for the normal pipeline.

For stretches when Protect's motion events missed cars (on 2026-11-02 its
motion zone didn't cover the whole road until the afternoon). The live source
is still the motion events.

Each 5-minute chunk of the Low channel (640×360) exports in about 1.4 s and
scans in about 7 s of CPU (measured 2026-11-02). That's ~5 minutes for 3
hours of footage. Windows already inside a motion event are skipped, so a
car isn't analysed twice.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from .logtime import local
from .scan import find_windows

log = logging.getLogger(__name__)

CHUNK_S = 300
WARMUP_S = 10.0
RECONNECT_WAIT_S = 30


def chunks(start: datetime, end: datetime, *, chunk_s: int = CHUNK_S, warmup_s: float = WARMUP_S):
    """(export_start, export_end) pairs covering [start, end]. Each starts
    warmup_s early: find_windows ignores motion during its warmup, which the
    previous chunk has already covered."""
    a = start
    while a < end:
        b = min(a + timedelta(seconds=chunk_s), end)
        yield a - timedelta(seconds=warmup_s), b
        a = b


STATE_KEY = "backfill"


async def run_backfill(settings, db, source, road, start: datetime, end: datetime) -> int:
    """Queue the motion windows in [start, end]. Returns how many were new.

    Records how far it has got in the state table after each chunk, so a
    backfill a deploy or restart cut off can carry on (resume_backfill).
    Clears the record when it finishes."""
    tmp = settings.data_dir / "scan"
    queued = skipped = 0
    progress = {"start": start.isoformat(), "end": end.isoformat(), "done_to": start.isoformat()}
    await db.set_state(STATE_KEY, progress)
    for a, b in chunks(start, end):
        clip = tmp / f"scan-{a:%Y%m%dT%H%M%S}.mp4"
        try:
            got = await source.export(a, b, clip, low_res=True)
            wins = await asyncio.to_thread(find_windows, clip, road, warmup_s=WARMUP_S)
        except Exception:  # noqa: BLE001  (one bad chunk shouldn't stop the rest)
            log.exception("backfill chunk %s failed", local(a, settings.tz))
            continue
        finally:
            clip.unlink(missing_ok=True)
        base = got or a  # a folder clip starts at the keyframe at or before a
        for s0, s1 in wins:
            ws, we = (base + timedelta(seconds=s0)).isoformat(), (base + timedelta(seconds=s1)).isoformat()
            if await db.event_covering(ws, we):
                skipped += 1
                continue
            if await db.upsert_event(f"scan-{base + timedelta(seconds=s0):%Y%m%dT%H%M%S}", ws, we):
                queued += 1
        log.info("backfill chunk %s windows=%d", local(a, settings.tz), len(wins))
        await db.set_state(STATE_KEY, {**progress, "done_to": b.isoformat()})
    await db.clear_state(STATE_KEY)
    log.info("backfill %s to %s: queued=%d skipped_covered=%d", local(start, settings.tz), local(end, settings.tz),
             queued, skipped)
    return queued


async def resume_backfill(settings, db, source, road) -> int | None:
    """Carry on with a backfill that was cut off, from the end of its last
    finished chunk. Returns how many windows it queued, or None if there was
    nothing to resume."""
    p = await db.get_state(STATE_KEY)
    if not p:
        return None
    a, b = datetime.fromisoformat(p["done_to"]), datetime.fromisoformat(p["end"])
    if a >= b:
        await db.clear_state(STATE_KEY)
        return 0
    # At startup the source may not be connected yet, and an export without a
    # connection fails: the chunk would be skipped.
    while not await source.ensure_connected():
        await asyncio.sleep(RECONNECT_WAIT_S)
    log.info("backfill resuming at %s (started from %s) to %s",
             *(local(p[k], settings.tz) for k in ("done_to", "start", "end")))
    return await run_backfill(settings, db, source, road, a, b)
