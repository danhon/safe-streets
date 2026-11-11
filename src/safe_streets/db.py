"""All SQL. One aiosqlite connection per process."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite

from .kinds import BIKE_SQL, CAR_SQL, type_sql

SCHEMA = (Path(__file__).parent / "schema.sql").read_text()

# Flags that don't make a speed a rough estimate: a caveat about the clip, or
# no speed at all (shown as "—"). Any other flag marks the row "rough".
SHOWN_FLAGS = {"low_fps", "low_res", "uncalibrated", "camera_moved"}


def is_hidden(flags: list[str] | str) -> bool:
    if isinstance(flags, str):
        flags = [f for f in flags.split(",") if f]
    return any(f not in SHOWN_FLAGS for f in flags)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


MPS_PER_MPH = 0.44704


def _with_mps(f: dict) -> None:
    """The analysis measures in mph; a pass also keeps its speed in m/s."""
    if "mph" in f and "mps" not in f:
        f["mps"] = None if f["mph"] is None else f["mph"] * MPS_PER_MPH


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.executescript(SCHEMA)
        cols = [r["name"] for r in await self._all("PRAGMA table_info(events)")]
        if "first_done_at" not in cols:
            # An old DB: its analysed events must not look new to the poster.
            await self.conn.execute("ALTER TABLE events ADD COLUMN first_done_at TEXT")
            # Every event that exists now is old, even one mid-reanalysis.
            await self.conn.execute("UPDATE events SET first_done_at = created_at")
        if "stages_json" not in cols:
            await self.conn.execute("ALTER TABLE events ADD COLUMN stages_json TEXT")
        cols = [r["name"] for r in await self._all("PRAGMA table_info(posts)")]
        if "yell_key" not in cols:
            await self.conn.execute("ALTER TABLE posts ADD COLUMN yell_key TEXT")
        cols = [r["name"] for r in await self._all("PRAGMA table_info(passes)")]
        if "hero_path" not in cols:
            await self.conn.execute("ALTER TABLE passes ADD COLUMN hero_path TEXT")
        for col in ("card_path", "track_json", "body", "vehicle_json", "colour"):
            if col not in cols:
                await self.conn.execute(f"ALTER TABLE passes ADD COLUMN {col} TEXT")
        if "mps" not in cols:  # speeds are kept in m/s; every display converts
            await self.conn.execute("ALTER TABLE passes ADD COLUMN mps REAL")
            await self.conn.execute(f"UPDATE passes SET mps = mph * {MPS_PER_MPH}")
        cols = [r["name"] for r in await self._all("PRAGMA table_info(calibrations)")]
        for col in ("lane", "trusted", "night_lane", "occluders", "mask", "size"):
            if col not in cols:
                await self.conn.execute(f"ALTER TABLE calibrations ADD COLUMN {col} TEXT")
                if col == "lane":  # an old calibration's lane is its road
                    await self.conn.execute("UPDATE calibrations SET lane = road")
        await self.conn.commit()

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()

    async def drain(self) -> None:
        """Return once every query already queued has run: aiosqlite runs
        them in order, on one thread."""
        await self._one("SELECT 1")

    async def _one(self, sql: str, args=()):
        async with self.conn.execute(sql, args) as cur:
            return await cur.fetchone()

    async def _all(self, sql: str, args=()):
        async with self.conn.execute(sql, args) as cur:
            return await cur.fetchall()

    # events -------------------------------------------------------------

    async def upsert_event(self, id: str, start: str, end: str) -> bool:
        cur = await self.conn.execute(
            "INSERT OR IGNORE INTO events (id, start_ts, end_ts) VALUES (?, ?, ?)", (id, start, end)
        )
        await self.conn.commit()
        return cur.rowcount == 1

    async def next_events(self, limit: int, live_since: str | None = None):
        """Newest first, so live cars aren't stuck behind a backlog (a
        reanalyse or a backfill), as they were on 2026-11-02. Except that
        events never analysed, not backfill scans, and starting at or after
        `live_since` go first and oldest first: the bot posts as they're
        analysed, and load can split them across batches (2026-11-04)."""
        live = "(first_done_at IS NULL AND id NOT LIKE 'scan-%' AND start_ts >= ?)"
        return await self._all(
            "SELECT * FROM events WHERE status = 'queued' OR (status = 'failed' AND attempts < 3) "
            f"ORDER BY {live} DESC, CASE WHEN {live} THEN start_ts END ASC, start_ts DESC LIMIT ?",
            (live_since or "9999", live_since or "9999", limit),
        )

    async def get_event(self, id: str):
        return await self._one("SELECT * FROM events WHERE id = ?", (id,))

    async def set_event(self, id: str, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        await self.conn.execute(f"UPDATE events SET {cols} WHERE id = ?", (*fields.values(), id))
        await self.conn.commit()

    async def event_counts(self) -> dict:
        rows = await self._all("SELECT status, COUNT(*) AS n FROM events GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    async def old_clip_events(self, days: int):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return await self._all(
            "SELECT * FROM events WHERE clip_path IS NOT NULL AND start_ts < ?", (cutoff,)
        )

    async def old_hero_passes(self, days: int):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return await self._all(
            "SELECT * FROM passes WHERE (hero_path IS NOT NULL OR card_path IS NOT NULL) AND ts < ?", (cutoff,)
        )

    async def requeue_interrupted(self) -> int:
        """At startup: events a restart cut off mid-export or mid-analysis."""
        cur = await self.conn.execute(
            "UPDATE events SET status = 'queued' WHERE status IN ('exporting', 'processing')")
        await self.conn.commit()
        return cur.rowcount

    async def requeue_done_with_clips(self) -> int:
        """For reanalyse: queue done events that still have a clip again.
        Their passes stay until each is analysed again (process_event
        replaces them), so the page isn't emptied in the meantime."""
        rows = await self._all("SELECT id FROM events WHERE status = 'done' AND clip_path IS NOT NULL")
        ids = [r["id"] for r in rows]
        for id in ids:
            await self.conn.execute("UPDATE events SET status = 'queued', attempts = 0 WHERE id = ?", (id,))
        await self.conn.commit()
        return len(ids)

    async def event_covering(self, start: str, end: str) -> bool:
        """True if an event's window already contains [start, end]."""
        row = await self._one("SELECT 1 FROM events WHERE start_ts <= ? AND end_ts >= ? LIMIT 1", (start, end))
        return row is not None

    async def requeue_since(self, since: str) -> int:
        """Re-run every event that started at or after `since`, whatever its
        status. Its clip is exported from Protect again (no_vehicle events
        and those with no passes don't keep one). Its passes stay until it
        is analysed again: on 2026-11-02 deleting them up front emptied the
        page for the length of a 334-event re-run."""
        rows = await self._all("SELECT id FROM events WHERE start_ts >= ?", (since,))
        ids = [r["id"] for r in rows]
        for id in ids:
            await self.conn.execute("UPDATE events SET status = 'queued', attempts = 0 WHERE id = ?", (id,))
        await self.conn.commit()
        return len(ids)

    async def delete_passes(self, event_id: str) -> None:
        """An event's passes and their screenshots, heroes and cards: called when the event is
        analysed again, just before its new results are stored."""
        for p in await self._all("SELECT shot_path, hero_path, card_path FROM passes WHERE event_id = ?", (event_id,)):
            for path in (p["shot_path"], p["hero_path"], p["card_path"]):
                if path:
                    Path(path).unlink(missing_ok=True)
            if p["shot_path"]:  # its Log thumbnail (web.py), made on first view beside the shots
                shot = Path(p["shot_path"])
                (shot.parent.parent / "thumbs" / shot.name).unlink(missing_ok=True)
        await self.conn.execute("DELETE FROM passes WHERE event_id = ?", (event_id,))
        await self.conn.commit()

    # passes -------------------------------------------------------------

    # rain ------------------------------------------------------------------

    async def put_rain(self, slots: list[tuple[str, float]], *, fetched_at: str) -> None:
        await self.conn.executemany(
            "INSERT INTO rain (slot, mm, fetched_at) VALUES (?, ?, ?) "
            "ON CONFLICT(slot) DO UPDATE SET mm = excluded.mm, fetched_at = excluded.fetched_at",
            [(s, mm, fetched_at) for s, mm in slots])
        await self.conn.commit()

    async def rain_slots(self, start: str, end: str) -> dict[str, float]:
        """{slot: mm} for the slots ending in (start, end] (UTC ISO)."""
        rows = await self._all("SELECT slot, mm FROM rain WHERE slot > ? AND slot <= ?", (start, end))
        return {r["slot"]: r["mm"] for r in rows}

    async def latest_rain_slot(self) -> str | None:
        row = await self._one("SELECT MAX(slot) AS s FROM rain")
        return row["s"] if row else None

    async def oldest_pass_ts(self) -> str | None:
        row = await self._one("SELECT MIN(ts) AS t FROM passes")
        return row["t"] if row else None

    async def rain_before(self, ts: str, hours: float = 2, allow_missing: int = 0) -> float | None:
        """mm of rain in the 15-minute slots that ended in the `hours` up to
        `ts`, or None unless every one of them is stored (not fetched yet, or
        a gap). The newest `allow_missing` may be absent (a live post made
        before the next fetch); the sum is then of the rest."""
        end = _ts(ts).astimezone(timezone.utc)
        last = end.replace(minute=end.minute - end.minute % 15, second=0, microsecond=0)
        first = end - timedelta(hours=hours)
        want = []
        s = last
        while s > first:
            want.append(s.isoformat())
            s -= timedelta(minutes=15)
        if not want:
            return None
        marks = ", ".join("?" for _ in want)
        row = await self._one(f"SELECT COUNT(*) AS n, COALESCE(SUM(mm), 0) AS mm FROM rain WHERE slot IN ({marks})",
                              tuple(want))
        if not row["n"]:
            return None
        if row["n"] == len(want):
            return row["mm"]
        older = want[allow_missing:]
        if not older:
            return None
        marks = ", ".join("?" for _ in older)
        n = (await self._one(f"SELECT COUNT(*) AS n FROM rain WHERE slot IN ({marks})", tuple(older)))["n"]
        return row["mm"] if n == len(older) else None

    async def public_rows(self, start: str, end: str):
        """public_view's rows: start <= ts < end, every stored pass (night and
        rough ones too: `hidden` only folds rough rows in the operator's
        table), only the fields it may use (never ids, pictures or the
        classifier's reply)."""
        return await self._all("SELECT ts, mps, flags, direction, cls FROM passes "
                               "WHERE ts >= ? AND ts < ? ORDER BY ts", (start, end))

    async def add_pass(self, **f) -> int:
        f.setdefault("flags", "")
        _with_mps(f)
        if isinstance(f["flags"], list):
            f["flags"] = ",".join(f["flags"])
        f["hidden"] = 1 if is_hidden(f["flags"]) else 0
        cols = ", ".join(f)
        marks = ", ".join("?" for _ in f)
        cur = await self.conn.execute(f"INSERT INTO passes ({cols}) VALUES ({marks})", tuple(f.values()))
        await self.conn.commit()
        return cur.lastrowid

    async def set_pass(self, id: int, **fields) -> None:
        _with_mps(fields)
        cols = ", ".join(f"{k} = ?" for k in fields)
        await self.conn.execute(f"UPDATE passes SET {cols} WHERE id = ?", (*fields.values(), id))
        await self.conn.commit()

    async def get_pass(self, id: int):
        return await self._one("SELECT * FROM passes WHERE id = ?", (id,))

    _LISTED = ("SELECT p.*, e.clip_path, e.ms AS event_ms, e.stages_json, "
               "(SELECT uri FROM posts q WHERE q.event_id = p.event_id AND q.ts = p.ts AND q.status = 'posted' "
               "ORDER BY q.id DESC LIMIT 1) AS post_uri FROM passes p LEFT JOIN events e ON e.id = p.event_id")

    async def passes_list(self, where: str = "", params: tuple = (), order: str = "p.ts DESC, p.id DESC",
                          limit: int | None = None):
        """Every pass, uncertain ones included (2026-11-02): the page
        marks how rough each estimate is instead of hiding it. `where` and
        `order` come from `table.where`, `table.after` and `table.order_by`."""
        cond = f" WHERE {where}" if where else ""
        lim = " LIMIT ?" if limit is not None else ""
        return await self._all(f"{self._LISTED}{cond} ORDER BY {order}{lim}",
                               (*params, limit) if limit is not None else params)

    async def count_passes(self, where: str = "", params: tuple = ()) -> int:
        cond = f" WHERE {where}" if where else ""
        return (await self._one(f"SELECT COUNT(*) AS n FROM passes p{cond}", params))["n"]

    async def stats_rows(self, where: str = "", params: tuple = ()):
        """The time, speed and flags of the passes `where` picks, in time order:
        the table's day and hour headers count them."""
        cond = f" WHERE {where}" if where else ""
        return await self._all(f"SELECT p.ts, p.mph, p.mps, p.plus_minus, p.flags FROM passes p{cond} ORDER BY p.ts", params)

    async def pass_hours(self) -> list[str]:
        """Every UTC hour with a pass ("2026-11-02T19"): which local days have
        any, for the day picker, without reading every pass."""
        return [r["h"] for r in await self._all("SELECT DISTINCT substr(ts, 1, 13) AS h FROM passes ORDER BY h")]

    async def pass_classes(self, body: bool = False) -> list[str]:
        """The Type filter's values: YOLO's classes, or with `body` what the
        Type column shows (the vehicle type where known, else YOLO's class)."""
        return [r["t"] for r in await self._all(f"SELECT DISTINCT {type_sql(body)} AS t FROM passes ORDER BY t")]

    async def passes_to_classify(self, encoder: str, limit: int, since: str | None = None,
                                 retry_skipped: tuple[str, ...] = ()):
        """Passes with no answer from `encoder` (none yet, or another
        encoder's), or marked skipped for one of `retry_skipped`, newest
        first, none before `since`: the vehicle catch-up's work."""
        retry = (" OR json_extract(vehicle_json, '$.skipped') IN (" + ", ".join("?" for _ in retry_skipped) + ")"
                 if retry_skipped else "")
        return await self._all(
            "SELECT * FROM passes WHERE (vehicle_json IS NULL OR json_extract(vehicle_json, '$.encoder') IS NOT ?"
            f"{retry}) AND ts >= COALESCE(?, '') ORDER BY ts DESC LIMIT ?", (encoder, *retry_skipped, since, limit))

    async def sync_two_wheeler_kinds(self, on: bool) -> int:
        """Make two-wheelers' body match TWO_WHEELER_TYPE, at start: on, the
        sidecar's kind from vehicle_json (NULL when unsure); otherwise NULL,
        so YOLO's class decides. How many passes changed."""
        kind = "json_extract(vehicle_json, '$.two_wheeler.kind')"
        if on:
            sql = (f"UPDATE passes SET body = {kind} WHERE cls IN ('motorcycle', 'bicycle') "
                   f"AND json_extract(vehicle_json, '$.two_wheeler') IS NOT NULL AND body IS NOT {kind}")
        else:
            sql = "UPDATE passes SET body = NULL WHERE cls IN ('motorcycle', 'bicycle') AND body IS NOT NULL"
        cur = await self.conn.execute(sql)
        await self.conn.commit()
        return cur.rowcount

    async def passes_without_colour(self, limit: int):
        """Passes the sidecar answered (not skipped) with no colour, newest
        first: the colour backfill's work."""
        return await self._all(
            "SELECT * FROM passes WHERE colour IS NULL AND vehicle_json IS NOT NULL "
            "AND json_extract(vehicle_json, '$.skipped') IS NULL ORDER BY ts DESC LIMIT ?", (limit,))

    async def newest_passes(self, limit: int, offset: int = 0):
        """Every pass, newest first, a page at a time: only what
        vehicle.skip_reason reads (the window scan runs every batch)."""
        return await self._all("SELECT id, ts, cls, flags, shot_path FROM passes "
                               "ORDER BY ts DESC LIMIT ? OFFSET ?", (limit, offset))

    async def pass_near(self, ts: str, direction: int, seconds: float, exclude_event: str | None = None,
                        two_wheeler: bool | None = None) -> bool:
        """A pass in `direction` within `seconds` of `ts`: the same vehicle.
        `two_wheeler` (True/False) looks only in that group, so a car
        overtaking a cyclist isn't taken for the cyclist (plan
        2026-11-09-two-wheelers); None looks at every pass."""
        t = _ts(ts)
        lo = (t - timedelta(seconds=seconds)).isoformat()
        hi = (t + timedelta(seconds=seconds)).isoformat()
        group = "" if two_wheeler is None else \
            " AND cls " + ("IN" if two_wheeler else "NOT IN") + " ('motorcycle', 'bicycle')"
        row = await self._one(
            "SELECT 1 FROM passes WHERE direction = ? AND ts BETWEEN ? AND ? AND event_id IS NOT ?" + group + " LIMIT 1",
            (direction, lo, hi, exclude_event),
        )
        return row is not None

    async def add_post(self, *, event_id: str, ts: str, direction: int, mph: float | None, text: str,
                       status: str, uri: str | None = None, error: str | None = None,
                       yell_key: str | None = None) -> int:
        cur = await self.conn.execute(
            "INSERT INTO posts (event_id, ts, direction, mph, text, uri, status, error, yell_key) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (event_id, ts, direction, mph, text, uri, status, error, yell_key),
        )
        await self.conn.commit()
        return cur.lastrowid

    async def set_post(self, id: int, **fields) -> None:
        """Update a ledger row's outcome: only `status`, `uri` and `error` change."""
        if not fields or set(fields) - {"status", "uri", "error"}:
            raise ValueError(f"set_post: bad fields {sorted(fields)}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        await self.conn.execute(f"UPDATE posts SET {cols} WHERE id = ?", (*fields.values(), id))
        await self.conn.commit()

    async def recent_yell_keys(self, since: str) -> set[str]:
        """Yell keys of posts (or ones of unknown fate) created at or after
        `since` (UTC, `%Y-%m-%dT%H:%M:%SZ`)."""
        rows = await self._all(
            "SELECT yell_key FROM posts WHERE status IN ('posted','unknown') "
            "AND yell_key IS NOT NULL AND created_at >= ?", (since,))
        return {r["yell_key"] for r in rows}

    async def posted_near(self, event_id: str, ts: str, direction: int, seconds: float = 2) -> bool:
        """A post (or one of unknown fate) for a car in this direction within
        `seconds` of `ts`, from any event: overlapping windows see one car twice.
        `event_id` is not used for matching."""
        t = _ts(ts)
        lo = (t - timedelta(seconds=seconds)).isoformat()
        hi = (t + timedelta(seconds=seconds)).isoformat()
        row = await self._one(
            "SELECT 1 FROM posts WHERE status IN ('posted','unknown') AND direction = ? AND ts BETWEEN ? AND ? LIMIT 1",
            (direction, lo, hi),
        )
        return row is not None

    async def passes_for_event(self, event_id: str):
        return await self._all("SELECT * FROM passes WHERE event_id = ? ORDER BY ts", (event_id,))

    async def passes_between(self, start: str, end: str):
        return await self._all("SELECT * FROM passes WHERE ts BETWEEN ? AND ? ORDER BY ts", (start, end))

    async def post_counts_since(self, since: str) -> dict[str, int]:
        rows = await self._all("SELECT status, COUNT(*) AS n FROM posts WHERE created_at >= ? GROUP BY status",
                               (since,))
        return {r["status"]: r["n"] for r in rows}

    async def last_post_at(self) -> str | None:
        row = await self._one("SELECT MAX(created_at) AS t FROM posts WHERE status = 'posted'")
        return row["t"]

    async def recent_lengths(self, n: int) -> list[float]:
        rows = await self._all(
            f"SELECT length_m FROM passes WHERE hidden = 0 AND length_m IS NOT NULL AND {CAR_SQL} ORDER BY ts DESC LIMIT ?", (n,)
        )
        return [r["length_m"] for r in rows]

    async def recent_passes(self, n: int):
        return await self._all("SELECT * FROM passes WHERE hidden = 0 ORDER BY ts DESC LIMIT ?", (n,))

    async def day_so_far(self, ts: str, tz: str) -> list[float | None]:
        """Speeds of every pass from local midnight of `ts`'s day (tz) up to
        and including `ts`, oldest first, None for a car without one: the day
        so far, for the yells' statistics (yells.tally)."""
        zone = ZoneInfo(tz)
        day = _ts(ts).astimezone(zone).date()
        start = datetime(day.year, day.month, day.day, tzinfo=zone).astimezone(timezone.utc).isoformat()
        rows = await self._all(f"SELECT mph FROM passes WHERE ts >= ? AND ts <= ? AND {CAR_SQL} ORDER BY ts", (start, ts))
        return [r["mph"] for r in rows]

    async def fastest_ever(self):
        """The fastest pass with a speed, rough estimates too (its ts and
        mph), or None: the fastest card's all-time comparison."""
        return await self._one(f"SELECT ts, mph FROM passes WHERE mph IS NOT NULL AND {CAR_SQL} ORDER BY mph DESC, ts LIMIT 1")

    async def speeds_between(self, start: str, end: str):
        """Every pass in [start, end) (UTC ISO), uncertain ones included, for
        the histogram and the hourly charts: its time, speed (or None) and flags."""
        return await self._all(f"SELECT ts, mph, mps, plus_minus, flags FROM passes WHERE ts >= ? AND ts < ? AND {CAR_SQL} ORDER BY ts",
                               (start, end))

    async def bikes_between(self, start: str, end: str) -> int:
        """Bikes (pedal and cargo: kinds.BIKE_SQL) in [start, end) (UTC ISO),
        with a speed or not: counted apart from the cars (2026-11-09)."""
        row = await self._one(f"SELECT COUNT(*) AS n FROM passes WHERE ts >= ? AND ts < ? AND {BIKE_SQL}",
                              (start, end))
        return row["n"]

    async def events_between(self, start: str, end: str):
        """Analysed events starting in [start, end) (UTC ISO), for the
        processing-time charts: status, start, total ms and stages."""
        return await self._all(
            "SELECT status, start_ts, ms, stages_json FROM events WHERE start_ts >= ? AND start_ts < ? "
            "AND status IN ('done', 'no_vehicle') AND ms IS NOT NULL ORDER BY start_ts", (start, end))

    async def wall_rows(self, start: str, end: str):
        """Every pass with a speed in [start, end) (UTC ISO), oldest first, with
        `post_text`: the latest posted/dry-run post of that pass that has a yell."""
        return await self._all(
            "SELECT p.*, (SELECT text FROM posts q WHERE q.event_id = p.event_id AND q.ts = p.ts "
            "AND q.status IN ('posted','dry-run') AND q.yell_key IS NOT NULL "
            "ORDER BY q.id DESC LIMIT 1) AS post_text "
            "FROM passes p WHERE p.ts >= ? AND p.ts < ? AND p.mph IS NOT NULL ORDER BY p.ts", (start, end))

    async def strudel_rows(self, start: str, end: str):
        """Every pass in [start, end) (UTC ISO), oldest first, bikes and rough
        ones too: what the /strudel prototypes play (strudel.payload)."""
        return await self._all("SELECT id, ts, mph, mps, plus_minus, flags, direction, cls, body, colour, "
                               "vehicle_json, length_m FROM passes WHERE ts >= ? AND ts < ? ORDER BY ts, id",
                               (start, end))

    # the hourly report ----------------------------------------------------

    async def report_rows(self, start: str, end: str):
        """Every pass in [start, end) (UTC ISO), oldest first: what the report's cards read."""
        return await self._all("SELECT ts, mph, mps, plus_minus, flags, direction, cls, colour, body, "
                               "json_extract(vehicle_json, '$.fine') AS fine FROM passes "
                               f"WHERE ts >= ? AND ts < ? AND {CAR_SQL} ORDER BY ts", (start, end))

    async def wheel_rows(self, start: str, end: str):
        """Every daytime two-wheeler in [start, end), bikes too, oldest first:
        the report's two-wheeler cards (kinds.kind decides each one's kind)."""
        return await self._all("SELECT ts, mph, mps, plus_minus, flags, direction, cls, colour, body, "
                               "NULL AS fine FROM passes WHERE ts >= ? AND ts < ? AND cls IN ('motorcycle', "
                               "'bicycle') AND instr(COALESCE(flags, ''), 'headlights') = 0 ORDER BY ts",
                               (start, end))

    async def fastest_before(self, end: str):
        """The fastest pass with a speed before `end`, rough estimates too, or None."""
        return await self._one("SELECT ts, mph, mps, plus_minus, flags, direction, cls FROM passes "
                               f"WHERE mph IS NOT NULL AND ts < ? AND {CAR_SQL} ORDER BY mph DESC, ts LIMIT 1", (end,))

    async def totals_before(self, end: str, limit: float) -> dict:
        """Every pass before `end`: how many, how many over the limit, over it
        by more than 10, and the first one's ts."""
        row = await self._one("SELECT COUNT(*) AS n, COALESCE(SUM(mph > ?), 0) AS over, "
                              f"COALESCE(SUM(mph > ?), 0) AS over10, MIN(ts) AS first FROM passes WHERE ts < ? AND {CAR_SQL}",
                              (limit, limit + 10, end))
        return dict(row)

    async def events_waiting(self, start: str, end: str) -> int:
        """Events starting in [start, end) still to analyse: the report waits for them."""
        row = await self._one("SELECT COUNT(*) AS n FROM events WHERE start_ts >= ? AND start_ts < ? "
                              "AND (status IN ('queued', 'exporting', 'processing') "
                              "OR (status = 'failed' AND attempts < 3))", (start, end))
        return row["n"]

    async def add_report(self, *, hour: str, cards: list[dict], text: str, status: str,
                         created_at: str) -> int | None:
        """The hour's ledger row; None if the hour already has one. created_at:
        the reporter's clock (UTC, `%Y-%m-%dT%H:%M:%SZ`), which its cooldowns read."""
        cur = await self.conn.execute(
            "INSERT OR IGNORE INTO reports (hour, cards, text, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (hour, json.dumps(cards), text, status, created_at))
        await self.conn.commit()
        return cur.lastrowid if cur.rowcount == 1 else None

    async def set_report(self, id: int, **fields) -> None:
        if not fields or set(fields) - {"status", "uri", "error"}:
            raise ValueError(f"set_report: bad fields {sorted(fields)}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        await self.conn.execute(f"UPDATE reports SET {cols} WHERE id = ?", (*fields.values(), id))
        await self.conn.commit()

    async def report_for(self, hour: str):
        return await self._one("SELECT * FROM reports WHERE hour = ?", (hour,))

    async def get_report(self, id: int):
        return await self._one("SELECT * FROM reports WHERE id = ?", (id,))

    async def recent_reports(self, n: int):
        return await self._all("SELECT * FROM reports ORDER BY hour DESC LIMIT ?", (n,))

    async def old_reports(self, before: str):
        return await self._all("SELECT * FROM reports WHERE created_at < ?", (before,))

    async def cards_last_out(self, since: str) -> dict[str, str]:
        """Each card's latest report or post by hand (card_posts) created at
        or after `since` (UTC, `%Y-%m-%dT%H:%M:%SZ`), posted, of unknown fate
        or a dry run: {card id: created_at}. A card posted by hand rests like
        one the report posted."""
        rows = await self._all("SELECT cards, created_at FROM reports WHERE created_at >= ? "
                               "AND status IN ('posted', 'unknown', 'dry-run') ORDER BY created_at", (since,))
        out: dict[str, str] = {}
        for r in rows:
            for c in json.loads(r["cards"]):
                out[c["id"]] = max(out.get(c["id"], ""), r["created_at"])
        for r in await self._all("SELECT card, created_at FROM card_posts WHERE created_at >= ? "
                                 "AND status IN ('posted', 'unknown', 'dry-run')", (since,)):
            out[r["card"]] = max(out.get(r["card"], ""), r["created_at"])
        return out

    async def add_card_post(self, *, card: str, hour: str, title: str, alt: str, text: str, path: str | None,
                            status: str, created_at: str) -> int | None:
        """A card posted by hand: its ledger row; None if that card already
        went out for that hour."""
        cur = await self.conn.execute(
            "INSERT OR IGNORE INTO card_posts (card, hour, title, alt, text, path, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (card, hour, title, alt, text, path, status, created_at))
        await self.conn.commit()
        return cur.lastrowid if cur.rowcount == 1 else None

    async def set_card_post(self, id: int, **fields) -> None:
        if not fields or set(fields) - {"status", "uri", "error"}:
            raise ValueError(f"set_card_post: bad fields {sorted(fields)}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        await self.conn.execute(f"UPDATE card_posts SET {cols} WHERE id = ?", (*fields.values(), id))
        await self.conn.commit()

    async def card_post_for(self, card: str, hour: str):
        return await self._one("SELECT * FROM card_posts WHERE card = ? AND hour = ?", (card, hour))

    async def get_card_post(self, id: int):
        return await self._one("SELECT * FROM card_posts WHERE id = ?", (id,))

    async def recent_card_posts(self, n: int):
        return await self._all("SELECT * FROM card_posts ORDER BY created_at DESC, id DESC LIMIT ?", (n,))

    # calibrations -------------------------------------------------------

    async def save_calibration(self, *, image_pts, world_pts, H, road, reference_path, status, lane=None,
                               trusted=None, night_lane=None, occluders=None, mask=None, size=None) -> int:
        """Points in the snapshot's pixels (size: its [w, h]; None: 1920x1080)."""
        def j(v):
            return None if v is None else json.dumps(v)

        await self.conn.execute("UPDATE calibrations SET active = 0")
        cur = await self.conn.execute(
            "INSERT INTO calibrations (image_pts, world_pts, H, road, reference_path, status, active, "
            "lane, trusted, night_lane, occluders, mask, size) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
            (json.dumps(image_pts), json.dumps(world_pts), json.dumps(H), json.dumps(road), reference_path, status,
             j(lane), j(trusted), j(night_lane), j(occluders), j(mask), j(size)),
        )
        await self.conn.commit()
        return cur.lastrowid

    async def get_operator(self) -> dict | None:
        row = await self._one("SELECT * FROM operator WHERE id = 1")
        return dict(row) if row else None

    async def update_operator(self, **f) -> None:
        await self.conn.execute("INSERT OR IGNORE INTO operator (id) VALUES (1)")
        if f:
            cols = ", ".join(f"{k} = ?" for k in f)
            await self.conn.execute(f"UPDATE operator SET {cols} WHERE id = 1", tuple(f.values()))
        await self.conn.commit()

    async def active_calibration(self):
        return await self._one("SELECT * FROM calibrations WHERE active = 1 ORDER BY id DESC LIMIT 1")

    async def set_calibration_status(self, id: int, status: str) -> None:
        await self.conn.execute("UPDATE calibrations SET status = ? WHERE id = ?", (status, id))
        await self.conn.commit()

    # state --------------------------------------------------------------

    async def get_state(self, key: str):
        row = await self._one("SELECT value FROM state WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else None

    async def set_state(self, key: str, value) -> None:
        await self.conn.execute(
            "INSERT INTO state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
        await self.conn.commit()

    async def clear_state(self, key: str) -> None:
        await self.conn.execute("DELETE FROM state WHERE key = ?", (key,))
        await self.conn.commit()
