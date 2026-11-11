from datetime import datetime, timedelta, timezone

import pytest

T0 = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat()


async def add(db, i, flags="", mph=22.0):
    return await db.add_pass(
        event_id=f"e{i}", ts=iso(T0 + timedelta(minutes=i)), cls="car", direction=1,
        mph=mph, plus_minus=1.0, metres=12.0, seconds=1.5, n=30, flags=flags,
        shot_path=f"/s/{i}.jpg", calibration_id=1, clip_t=1.0,
    )


async def test_upsert_is_idempotent(db):
    assert await db.upsert_event("a", iso(T0), iso(T0 + timedelta(seconds=9))) is True
    assert await db.upsert_event("a", iso(T0), iso(T0 + timedelta(seconds=9))) is False


async def test_passes_list_orders_and_limits(db):
    for i in range(5):
        await db.add_pass(event_id=f"e{i}", ts=f"2026-11-02T19:0{i}:00+00:00", cls="car", direction=1, mph=20.0 + i)
    rows = await db.passes_list(limit=2)
    assert [r["mph"] for r in rows] == [24.0, 23.0] and "post_uri" in rows[0].keys()
    assert await db.count_passes("p.mph > ?", (21.0,)) == 3
    assert [r["mph"] for r in await db.stats_rows("p.mph < ?", (22.0,))] == [20.0, 21.0]
    assert await db.pass_hours() == ["2026-11-02T19"]


async def test_uncertain_passes_always_listed(db):
    await add(db, 1, flags="partial")
    await add(db, 2, flags="low_fps")
    assert len(await db.passes_list()) == 2 and await db.count_passes() == 2




async def test_failed_retried_until_3_attempts(db):
    await db.upsert_event("a", iso(T0), iso(T0))
    for attempt in range(1, 4):
        await db.set_event("a", status="failed", attempts=attempt)
        ids = [r["id"] for r in await db.next_events(10)]
        assert ("a" in ids) == (attempt < 3)


async def test_save_calibration_deactivates_previous(db):
    a = await db.save_calibration(image_pts=[[0, 0]], world_pts=[[0, 0]], H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  road=[[0, 0]], reference_path="/r1.jpg", status="provisional")
    b = await db.save_calibration(image_pts=[[0, 0]], world_pts=[[0, 0]], H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  road=[[0, 0]], reference_path="/r2.jpg", status="active")
    active = await db.active_calibration()
    assert active["id"] == b and a != b


async def test_pass_near(db):
    await add(db, 0)
    ts = T0 + timedelta(seconds=1)
    assert await db.pass_near(iso(ts), 1, 2) is True
    assert await db.pass_near(iso(ts), -1, 2) is False
    assert await db.pass_near(iso(T0 + timedelta(seconds=5)), 1, 2) is False


async def test_pass_near_exclude_event(db):
    await add(db, 0)
    ts = iso(T0 + timedelta(seconds=1))
    assert await db.pass_near(ts, 1, 2, exclude_event="e0") is False
    assert await db.pass_near(ts, 1, 2, exclude_event="other") is True


async def test_state_roundtrip(db):
    assert await db.get_state("k") is None
    await db.set_state("k", {"a": 1})
    assert await db.get_state("k") == {"a": 1}


async def test_interrupted_events_requeued(db):
    await db.upsert_event("a", iso(T0), iso(T0))
    await db.upsert_event("b", iso(T0), iso(T0))
    await db.set_event("a", status="processing")
    await db.set_event("b", status="exporting")
    assert await db.requeue_interrupted() == 2
    assert {r["id"] for r in await db.next_events(10)} == {"a", "b"}


async def test_requeue_since(db, tmp_path):
    shot = tmp_path / "s.jpg"
    shot.write_bytes(b"x")
    await db.upsert_event("old", iso(T0 - timedelta(hours=2)), iso(T0 - timedelta(hours=2)))
    await db.upsert_event("nv", iso(T0), iso(T0))
    await db.upsert_event("done", iso(T0 + timedelta(minutes=5)), iso(T0 + timedelta(minutes=5)))
    await db.set_event("old", status="no_vehicle")
    await db.set_event("nv", status="no_vehicle", attempts=1)
    await db.set_event("done", status="done", attempts=1)
    pid = await db.add_pass(event_id="done", ts=iso(T0), cls="car", direction=1)
    await db.set_pass(pid, shot_path=str(shot))
    assert await db.requeue_since(iso(T0 - timedelta(minutes=1))) == 2
    assert {r["id"] for r in await db.next_events(10)} == {"nv", "done"}
    assert (await db.get_event("old"))["status"] == "no_vehicle"
    # Passes stay until the event is analysed again, so the page isn't emptied.
    assert await db.get_pass(pid) is not None and shot.exists()
    await db.delete_passes("done")
    assert await db.get_pass(pid) is None and not shot.exists()


async def test_event_covering(db):
    await db.upsert_event("m", iso(T0), iso(T0 + timedelta(seconds=30)))
    assert await db.event_covering(iso(T0 + timedelta(seconds=5)), iso(T0 + timedelta(seconds=25)))
    assert not await db.event_covering(iso(T0 + timedelta(seconds=20)), iso(T0 + timedelta(seconds=40)))


async def test_next_events_newest_first(db):
    # Live cars go ahead of a backlog (a reanalyse or a backfill).
    for i in range(3):
        await db.upsert_event(f"e{i}", iso(T0 + timedelta(minutes=i)), iso(T0 + timedelta(minutes=i)))
    assert [r["id"] for r in await db.next_events(10)] == ["e2", "e1", "e0"]


async def test_speeds_between(db):
    for i, (ts, mph) in enumerate([("2026-11-01T18:59:00+00:00", 30.0), ("2026-11-01T19:00:00+00:00", 21.0),
                                   ("2026-11-02T18:00:00+00:00", None), ("2026-11-02T19:00:00+00:00", 25.0)]):
        await db.add_pass(event_id=f"s{i}", ts=ts, cls="car", direction=1, mph=mph, plus_minus=1.0,
                          flags="coarse" if i == 1 else "")
    rows = await db.speeds_between("2026-11-01T19:00:00+00:00", "2026-11-02T19:00:00+00:00")
    assert [(r["mph"], r["flags"]) for r in rows] == [(21.0, "coarse"), (None, "")]


async def test_migration_marks_existing_events_analysed(tmp_path):
    import sqlite3

    from safe_streets.db import Database

    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE events (
        id TEXT PRIMARY KEY, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT, clip_path TEXT, fps REAL, ms INTEGER,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')))""")
    c.execute("INSERT INTO events (id, start_ts, end_ts, status) VALUES ('d', 'a', 'b', 'done')")
    c.execute("INSERT INTO events (id, start_ts, end_ts, status) VALUES ('q', 'a', 'b', 'queued')")
    c.execute("INSERT INTO events (id, start_ts, end_ts, status) VALUES ('rq', 'a', 'b', 'queued')")
    from safe_streets.db import SCHEMA

    c.execute("CREATE TABLE passes" + SCHEMA.split("CREATE TABLE IF NOT EXISTS passes", 1)[1].split(");", 1)[0] + ")")
    c.execute("INSERT INTO passes (event_id, ts, cls, direction) VALUES ('rq', 'a', 'car', 1)")
    c.commit()
    c.close()
    d = Database(path)
    await d.init()
    try:
        assert (await d.get_event("d"))["first_done_at"] is not None
        assert (await d.get_event("q"))["first_done_at"] is not None
        assert (await d.get_event("rq"))["first_done_at"] is not None
        assert (await d.get_event("d"))["stages_json"] is None  # the timing column, added
    finally:
        await d.close()
    # A second init must not touch anything: the UPDATE runs only on migration.
    d = Database(path)
    await d.init()
    try:
        await d.upsert_event("new", iso(T0), iso(T0 + timedelta(seconds=9)))
        await d.close()
        d = Database(path)
        await d.init()
        assert (await d.get_event("new"))["first_done_at"] is None
    finally:
        await d.close()


async def post(db, status, event="e1", ts=None, direction=1):
    return await db.add_post(event_id=event, ts=ts or iso(T0), direction=direction, mph=25.0,
                             text="t", status=status)


async def test_posted_near_ignores_dry_run_and_failed(db):
    await post(db, "dry-run")
    await post(db, "failed")
    assert not await db.posted_near("e2", iso(T0), 1)
    await post(db, "unknown")
    assert await db.posted_near("e2", iso(T0), 1)
    assert not await db.posted_near("e2", iso(T0), -1)


async def test_set_post_updates_only_the_outcome(db):
    import pytest

    a = await post(db, "unknown")
    b = await post(db, "unknown", ts=iso(T0 + timedelta(minutes=1)))
    await db.set_post(a, status="posted", uri="at://x/1")
    rows = await db._all("SELECT id, status, uri, error, text FROM posts ORDER BY id")
    assert [tuple(r) for r in rows] == [(a, "posted", "at://x/1", None, "t"), (b, "unknown", None, None, "t")]
    with pytest.raises(ValueError):
        await db.set_post(a, text="changed")


async def test_posted_near_matches_another_event_within_2s(db):
    await post(db, "posted", event="e1")
    assert await db.posted_near("e2", iso(T0 + timedelta(seconds=1.5)), 1)
    assert not await db.posted_near("e2", iso(T0 + timedelta(seconds=3)), 1)


async def test_post_queries(db):
    await add(db, 1)
    await add(db, 2)
    assert len(await db.passes_for_event("e1")) == 1
    rows = await db.passes_between(iso(T0), iso(T0 + timedelta(minutes=5)))
    assert [r["event_id"] for r in rows] == ["e1", "e2"]
    assert await db.last_post_at() is None
    await post(db, "posted")
    await post(db, "failed")
    await post(db, "posted")
    assert await db.post_counts_since("2000-01-01T00:00:00Z") == {"posted": 2, "failed": 1}
    assert await db.last_post_at() is not None


async def test_next_events_recent_live_ones_go_oldest_first(db):
    """Unanalysed events since `live_since` go first and oldest first, so the
    bot posts in time order even when load splits them across batches
    (2026-11-04); everything else stays newest first."""
    for eid, m in [("old", -120), ("live1", -10), ("live2", -5), ("re", -8), ("scan-x", -6)]:
        await db.upsert_event(eid, iso(T0 + timedelta(minutes=m)), iso(T0 + timedelta(minutes=m)))
    await db.set_event("re", first_done_at=iso(T0))
    got = [r["id"] for r in await db.next_events(10, live_since=iso(T0 - timedelta(hours=1)))]
    assert got == ["live1", "live2", "scan-x", "re", "old"]


async def test_old_posts_table_gains_yell_key(tmp_path):
    import sqlite3
    from safe_streets.db import Database, SCHEMA

    path = tmp_path / "old.db"
    old = SCHEMA.replace(",\n    yell_key    TEXT                        -- which yell the post used", "")
    assert old != SCHEMA
    c = sqlite3.connect(path)
    c.executescript(old)
    c.execute("INSERT INTO posts (event_id, ts, direction, text, status) VALUES ('e','t',1,'x','posted')")
    c.commit()
    c.close()
    d = Database(path)
    await d.init()
    try:
        row = await d._one("SELECT yell_key FROM posts")
        assert row["yell_key"] is None
        assert await d.recent_yell_keys("2000-01-01T00:00:00Z") == set()
    finally:
        await d.close()


async def test_recent_yell_keys_counts_posted_and_unknown_only(db):
    for status, key in (("posted", "a"), ("unknown", "b"), ("failed", "c"), ("dry-run", "d")):
        await db.add_post(event_id="e", ts="2026-11-04T20:00:00+00:00", direction=1, mph=30.0,
                          text="t", status=status, yell_key=key)
    assert await db.recent_yell_keys("2000-01-01T00:00:00Z") == {"a", "b"}
    assert await db.recent_yell_keys("2999-01-01T00:00:00Z") == set()


async def test_hero_path_migration(tmp_path):
    import sqlite3
    from safe_streets.db import Database, SCHEMA

    path = tmp_path / "old.db"
    old = SCHEMA.replace("    hero_path       TEXT,\n", "")
    assert old != SCHEMA
    c = sqlite3.connect(path)
    c.executescript(old)
    c.commit()
    c.close()
    d = Database(path)
    await d.init()
    try:
        cols = [r["name"] for r in await d._all("PRAGMA table_info(passes)")]
        assert "hero_path" in cols
    finally:
        await d.close()


async def test_delete_passes_unlinks_hero_and_card(db, tmp_path):
    shot, hero, card = tmp_path / "s.jpg", tmp_path / "h.jpg", tmp_path / "c.jpg"
    for f in (shot, hero, card):
        f.write_bytes(b"x")
    await db.add_pass(event_id="e1", ts=iso(T0), cls="car", direction=1, mph=20, plus_minus=1,
                      shot_path=str(shot), hero_path=str(hero), card_path=str(card))
    await db.delete_passes("e1")
    assert not shot.exists() and not hero.exists() and not card.exists()


async def test_card_columns_migration(tmp_path):
    import sqlite3
    from safe_streets.db import Database, SCHEMA

    path = tmp_path / "old.db"
    old = "\n".join(ln for ln in SCHEMA.split("\n") if not ln.lstrip().startswith(("card_path", "track_json")))
    assert old != SCHEMA
    c = sqlite3.connect(path)
    c.executescript(old)
    c.commit()
    c.close()
    d = Database(path)
    await d.init()
    try:
        cols = [r["name"] for r in await d._all("PRAGMA table_info(passes)")]
        assert "card_path" in cols and "track_json" in cols
    finally:
        await d.close()


async def wall_pass(db, ts=None, i=1):
    return await db.add_pass(event_id=f"w{i}", ts=ts or iso(T0), cls="car", direction=1, mph=30.0,
                             plus_minus=1.0, flags="")


async def test_wall_rows_joins_yell_post(db):
    await wall_pass(db)
    kw = dict(event_id="w1", ts=iso(T0), direction=1, mph=30.0)
    await db.add_post(**kw, text="with yell\nbody", status="posted", yell_key="k")
    await db.add_post(**kw, text="old\nbody", status="posted")
    await db.add_post(**kw, text="failed\nbody", status="failed", yell_key="k2")
    (r,) = await db.wall_rows(iso(T0 - timedelta(hours=1)), iso(T0 + timedelta(hours=1)))
    assert r["post_text"] == "with yell\nbody"


async def test_yell_needs_yell_key(db):
    await wall_pass(db)
    await db.add_post(event_id="w1", ts=iso(T0), direction=1, mph=30.0, text="24 mph, 4 over", status="posted")
    (r,) = await db.wall_rows(iso(T0 - timedelta(hours=1)), iso(T0 + timedelta(hours=1)))
    assert r["post_text"] is None


async def test_wall_rows_bounds(db):
    await wall_pass(db, iso(T0), 1)
    await wall_pass(db, iso(T0 + timedelta(hours=1)), 2)
    rows = await db.wall_rows(iso(T0), iso(T0 + timedelta(hours=1)))
    assert [r["event_id"] for r in rows] == ["w1"]


async def test_an_old_db_gains_the_vehicle_columns(tmp_path):
    import sqlite3
    from safe_streets.db import Database, SCHEMA

    path = tmp_path / "old.db"
    old = "\n".join(ln for ln in SCHEMA.split("\n") if not ln.lstrip().startswith(("body ", "vehicle_json")))
    assert old != SCHEMA
    c = sqlite3.connect(path)
    c.executescript(old)
    c.commit()
    c.close()
    d = Database(path)
    await d.init()
    try:
        cols = [r["name"] for r in await d._all("PRAGMA table_info(passes)")]
        assert "body" in cols and "vehicle_json" in cols
    finally:
        await d.close()


async def test_passes_to_classify_skips_the_current_encoders_answers(db):
    import json

    ids = []
    for i, enc in enumerate([None, "A", "B", None]):
        pid = await wall_pass(db, ts=iso(T0 + timedelta(minutes=i)), i=i)
        if enc:
            await db.set_pass(pid, vehicle_json=json.dumps({"encoder": enc}))
        ids.append(pid)
    got = [r["id"] for r in await db.passes_to_classify("A", 2)]
    assert got == [ids[3], ids[2]]  # newest first; A's answer is left alone
    assert [r["id"] for r in await db.passes_to_classify("A", 10)] == [ids[3], ids[2], ids[0]]


async def test_pass_classes_shows_the_body_where_known(db):
    for i, (cls, body) in enumerate([("car", "SUV"), ("car", None), ("truck", "pickup")]):
        pid = await wall_pass(db, ts=iso(T0 + timedelta(minutes=i)), i=i)
        await db.set_pass(pid, cls=cls, body=body)
    assert set(await db.pass_classes(body=True)) == {"SUV", "car", "pickup"}
    assert set(await db.pass_classes()) == {"car", "truck"}


async def test_day_so_far_is_from_local_midnight_to_the_pass(db):
    """The yells' statistics count the pass's own local day up to and
    including it (2026-11-08)."""
    async def put(ts, mph):
        await db.add_pass(event_id=f"d{ts.isoformat()}", ts=iso(ts), cls="car", direction=1, mph=mph,
                          plus_minus=1.0, metres=10, seconds=1.2, n=20, flags="", calibration_id=1,
                          clip_t=1.5)
    midnight = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)  # 00:00 PST
    await put(midnight - timedelta(seconds=1), 40.0)   # the day before
    await put(midnight, 21.0)
    await put(midnight + timedelta(hours=3), None)       # no speed: still a car
    await put(T0, 30.0)                                   # the pass itself
    await put(T0 + timedelta(seconds=1), 50.0)          # after it
    assert await db.day_so_far(iso(T0), "America/Los_Angeles") == [21.0, None, 30.0]


async def test_fastest_ever_is_the_fastest_pass_with_a_speed(db):
    """For the fastest card's all-time comparison (2026-11-08)."""
    assert await db.fastest_ever() is None
    for i, mph in enumerate((30.0, 41.2, None, 25.0)):
        await db.add_pass(event_id=f"f{i}", ts=iso(T0 + timedelta(minutes=i)), cls="car", direction=1, mph=mph,
                          plus_minus=1.0, metres=10, seconds=1.2, n=20, flags="", calibration_id=1, clip_t=1.5)
    r = await db.fastest_ever()
    assert (r["mph"], r["ts"]) == (41.2, iso(T0 + timedelta(minutes=1)))


# Two-wheelers: counts are cars, bikes counted apart (plan 2026-11-09-two-wheelers, Task 3)

async def _mixed_day(db):
    """A car at 30, a YOLO bicycle at 40, a motorcycle the sidecar called a
    bike at 35, a bicycle it called an e-bike at 27, a night pass at 24 and an
    unsure motorcycle at 26, a minute apart."""
    spec = [("car", None, "", 30.0), ("bicycle", None, "", 40.0), ("motorcycle", "bike", "", 35.0),
            ("bicycle", "e-bike", "", 27.0), ("car", None, "headlights", 24.0), ("motorcycle", None, "", 26.0)]
    for i, (cls, body, flags, mph) in enumerate(spec):
        await db.add_pass(event_id=f"m{i}", ts=iso(T0 + timedelta(minutes=i)), cls=cls, body=body, flags=flags,
                          direction=1, mph=mph, length_m=4.0 if cls == "car" else 1.8)
    return iso(T0), iso(T0 + timedelta(hours=1))


async def test_counts_leave_out_bikes(db):
    start, end = await _mixed_day(db)
    assert [r["mph"] for r in await db.speeds_between(start, end)] == [30.0, 27.0, 24.0, 26.0]
    assert [r["mph"] for r in await db.report_rows(start, end)] == [30.0, 27.0, 24.0, 26.0]
    assert (await db.fastest_ever())["mph"] == 30.0
    assert (await db.fastest_before(end))["mph"] == 30.0
    assert (await db.totals_before(end, 20))["n"] == 4
    last = iso(T0 + timedelta(minutes=5))
    assert await db.day_so_far(last, "UTC") == [30.0, 27.0, 24.0, 26.0]
    # (the night pass is hidden: headlights is a rough flag)
    assert sorted(await db.recent_lengths(20)) == [1.8, 1.8, 4.0]


async def test_bikes_between(db):
    start, end = await _mixed_day(db)
    assert await db.bikes_between(start, end) == 2
    assert await db.bikes_between(end, iso(T0 + timedelta(hours=2))) == 0


