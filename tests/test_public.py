"""The public side: Tier A data only, from one module (public_view.py).

Times to the quarter hour, published after a delay, a quarter hour of fewer
than PUBLIC_MIN_GROUP passes only as a total, the kind only as car or
two-wheeler, speeds rounded; no camera pixels, pass ids, exact times or
makes and models in any response; and public pages never read who's
signed in."""

import json
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from safe_streets.public_view import PublicPass, public_day
from safe_streets.web import create_app

TZ = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=TZ)  # the delay (24 h) puts the cutoff at 11-03 12:00
DAY = date(2026, 11, 3)
SECRET_IDS = range(987650, 987700)
FORBIDDEN = [
    (re.compile(r"\b9876\d\d\b"), "a pass id"),
    (re.compile(r"evt-secret"), "an event id"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b"), "a time to the second"),
    (re.compile(r"\.(jpe?g|mp4|png|webp)\b", re.I), "an image or clip URL"),
    (re.compile(r"/shot|/clip|/hero|/thumb", re.I), "a picture route"),
    (re.compile(r"Saab|Sonett", re.I), "a make or model"),
]


async def seed(db):
    ids = iter(SECRET_IDS)

    async def add(local: datetime, mph: float, **kw):
        kw.setdefault("cls", "car")
        await db.add_pass(id=next(ids), event_id=f"evt-secret-{local:%H%M%S}", ts=local.astimezone(timezone.utc).isoformat(),
                          direction=kw.pop("direction", 1), mph=mph, plus_minus=1.0, shot_path=f"shots/{local:%H%M%S}.jpg",
                          vehicle_json=json.dumps({"make": "Saab", "model": "Sonett"}), **kw)

    for i, mph in enumerate([18.2, 21.6, 24.4, 19.9, 30.1, 22.5]):  # 08:00-08:15: six, listed
        await add(datetime(2026, 11, 3, 8, 1 + 2 * i, 7, tzinfo=TZ), mph)
    for i in range(4):  # 09:15-09:30: four, counted only
        await add(datetime(2026, 11, 3, 9, 16 + i, 33, tzinfo=TZ), 26.0 + i)
    for i in range(5):  # 11:45-12:00: five, listed; one a bicycle going the wrong way
        await add(datetime(2026, 11, 3, 11, 50 + i, 41, tzinfo=TZ), 15.4 + i,
                  **({"cls": "bicycle", "direction": -1} if i == 0 else {}))
    for i in range(5):  # 12:30: after the cutoff, not shown yet
        await add(datetime(2026, 11, 3, 12, 30 + i, 5, tzinfo=TZ), 40.0)


@pytest.fixture
async def seeded(db):
    await seed(db)
    return db


async def test_nothing_newer_than_the_delay_is_shown(seeded, settings):
    day = await public_day(seeded, DAY, settings, NOW)
    assert day.total == 15
    cutoff = NOW - timedelta(hours=settings.public_delay_hours)
    assert all(p.bucket < cutoff for p in day.passes)
    later = await public_day(seeded, DAY, settings, NOW + timedelta(hours=2))
    assert later.total == 20


async def test_times_are_floored_to_the_quarter_hour(seeded, settings):
    day = await public_day(seeded, DAY, settings, NOW)
    assert {p.bucket for p in day.passes} == {datetime(2026, 11, 3, 8, 0, tzinfo=TZ),
                                               datetime(2026, 11, 3, 11, 45, tzinfo=TZ)}
    assert all(isinstance(p, PublicPass) and p.bucket.tzinfo is not None for p in day.passes)


async def test_a_quarter_hour_of_four_is_counted_but_not_listed(seeded, settings):
    day = await public_day(seeded, DAY, settings, NOW)
    assert len(day.passes) == 11 and day.unlisted == 4
    assert not any(p.bucket.hour == 9 for p in day.passes)
    nine = day.hours[9]
    assert nine.n == 4 and nine.fastest is None and nine.over is None  # a total only


async def test_kinds_speeds_and_directions_are_coarse(seeded, settings):
    day = await public_day(seeded, DAY, settings, NOW)
    eight = [p for p in day.passes if p.bucket.hour == 8]
    assert sorted(p.speed for p in eight) == [18, 20, 22, 22, 24, 30]
    assert {p.kind for p in day.passes} == {"car", "two-wheeler"}
    bike = next(p for p in day.passes if p.kind == "two-wheeler")
    assert bike.direction == -1 and bike.speed == 15


async def test_kmh(seeded, settings):
    import dataclasses
    day = await public_day(seeded, DAY, dataclasses.replace(settings, units="kmh", speed_limit=30.0), NOW)
    assert max(p.speed for p in day.passes) == round(30.1 * 1.609344)
    assert day.unit == "km/h" and day.limit == 30.0


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                             base_url="http://test")


PUBLIC = ["/", f"/?day={DAY}", "/?day=2026-11-02", f"/zoo?day={DAY}", f"/dial?day={DAY}", f"/day.json?day={DAY}",
          "/day.json", "/healthz"]


def app_for(settings, db):
    return create_app(settings, db, None, None, None, now=lambda: NOW)


async def test_every_public_response_carries_only_public_fields(seeded, settings):
    from safe_streets import web
    app = app_for(settings, seeded)
    static = [f"/static/{p.name}" for p in web.STATIC.iterdir() if p.is_file()]
    async with client(app) as c:
        for url in PUBLIC + static:
            r = await c.get(url)
            assert r.status_code == 200, url
            for pat, what in FORBIDDEN:
                m = pat.search(r.text)
                assert not m, f"{url} has {what}: {m.group(0)!r}"
    # The page did show the day.
    async with client(app) as c:
        body = (await c.get(f"/?day={DAY}")).text
    assert "<b>15</b>" in body and "8–9 am: 6 vehicles" in body


async def test_public_responses_are_cacheable_and_unindexed(seeded, settings):
    async with client(app_for(settings, seeded)) as c:
        for url in PUBLIC:
            r = await c.get(url)
            assert r.headers["cache-control"] == "public, max-age=60", url
            assert r.headers["x-robots-tag"] == "noindex", url


async def test_public_errors_leak_nothing(seeded, settings, monkeypatch):
    """Review Focus 4: a 404 and a forced 500 are plain pages."""
    from safe_streets import web

    async def boom(*a, **kw):
        raise RuntimeError("evt-secret 987650 shots/x.jpg Saab")

    async with client(app_for(settings, seeded)) as c:
        r = await c.get("/no-such-page")
        assert r.status_code == 404
        assert "Traceback" not in r.text and "evt-secret" not in r.text
        r = await c.get("/?day=not-a-day")
        assert r.status_code == 404
        monkeypatch.setattr(web, "public_day", boom)
        r = await c.get(f"/?day={DAY}")
    assert r.status_code == 500
    for bad in ("Traceback", "RuntimeError", "evt-secret", "987650", ".jpg", "Saab"):
        assert bad not in r.text
    assert "text/html" in r.headers["content-type"]


async def test_public_pages_never_read_who_is_signed_in(seeded, settings):
    signed_in = {"Cookie": "safe_streets_session=valid; other=1", "Authorization": "Bearer x",
                 "Remote-User": "operator", "Remote-Groups": "admins"}
    async with client(app_for(settings, seeded)) as c:
        for url in PUBLIC:
            a, b = await c.get(url), await c.get(url, headers=signed_in)
            assert a.content == b.content, url
            assert "set-cookie" not in b.headers, url


async def test_the_default_day_is_the_latest_one_published(seeded, settings):
    async with client(app_for(settings, seeded)) as c:
        d = (await c.get("/day.json")).json()
    assert d["date"] == str(DAY)  # 11-04 has nothing published yet at noon
    assert d["total"] == 15 and len(d["passes"]) == 11
    assert set(d["passes"][0]) == {"q", "speed", "rough", "dir", "kind"}


# -- the review's findings (2026-11-10) -------------------------------------

async def lone_speeder(db):
    """Five cars at 10:01-10:05 doing 18-22, and one alone at 10:40 doing 47."""
    for i, mph in enumerate([18, 19, 20, 21, 22]):
        await db.add_pass(event_id=f"e{i}", ts=datetime(2026, 11, 3, 10, 1 + i, tzinfo=TZ).astimezone(timezone.utc).isoformat(),
                          cls="car", direction=1, mph=mph, plus_minus=1.0)
    await db.add_pass(event_id="lone", ts=datetime(2026, 11, 3, 10, 40, tzinfo=TZ).astimezone(timezone.utc).isoformat(),
                      cls="car", direction=1, mph=47.0, plus_minus=1.0)


async def test_an_unlisted_car_shows_in_no_speed_figure(db, settings):
    await lone_speeder(db)
    day = await public_day(db, DAY, settings, NOW)
    assert day.unlisted == 1 and day.total == 6
    ten = day.hours[10]
    assert ten.n == 6 and ten.fastest == 22 and ten.over == 2
    assert day.over == 2
    blob = json.dumps(day.as_json())
    assert "47" not in blob


async def test_the_cutoff_moves_a_whole_bucket_at_a_time(db, settings):
    """Polling can't time a pass to the minute: nothing appears until its
    quarter hour is wholly past the delay."""
    await lone_speeder(db)
    base = datetime(2026, 11, 4, 10, 0, tzinfo=TZ)  # cutoff 11-03 10:00
    seen = {m: (await public_day(db, DAY, settings, base + timedelta(minutes=m))).total for m in range(0, 15)}
    assert set(seen.values()) == {0}  # 10:01-10:05 are in the 10:00 bucket, published at 10:15
    at15 = await public_day(db, DAY, settings, base + timedelta(minutes=15))
    assert at15.total == 5 and at15.published_until == datetime(2026, 11, 3, 10, 15, tzinfo=TZ)


async def test_an_hours_speeds_wait_for_the_whole_hour(db, settings):
    await lone_speeder(db)
    half = await public_day(db, DAY, settings, datetime(2026, 11, 4, 10, 30, tzinfo=TZ))
    assert half.hours[10].n == 5 and half.hours[10].fastest is None and half.hours[10].over is None
    full = await public_day(db, DAY, settings, datetime(2026, 11, 4, 11, 0, tzinfo=TZ))
    assert full.hours[10].fastest == 22


async def test_night_and_rough_passes_are_public_as_rough(db, settings):
    for i, flags in enumerate(["", "headlights", "partial", "short", "outside_trusted"]):
        await db.add_pass(event_id=f"n{i}", ts=datetime(2026, 11, 3, 6, 1 + i, tzinfo=TZ).astimezone(timezone.utc).isoformat(),
                          cls="car", direction=1, mph=25.0, plus_minus=1.0, flags=flags)
    await db.add_pass(event_id="ns", ts=datetime(2026, 11, 3, 6, 7, tzinfo=TZ).astimezone(timezone.utc).isoformat(),
                      cls="car", direction=1, mph=None, plus_minus=None, flags="short")
    day = await public_day(db, DAY, settings, NOW)
    assert day.total == 6 and len(day.passes) == 5
    assert [p.rough for p in day.passes] == [False, True, False, True, True]


async def test_the_share_over_is_of_the_listed_passes(db, settings):
    await lone_speeder(db)
    app = create_app(settings, db, None, None, None, now=lambda: NOW)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        html = (await c.get("/", params={"day": DAY.isoformat()})).text
    assert "40%" in html  # 2 of the 5 listed; not 2 of 6, nor the lone 47 (3 of 6)
    assert "of listed vehicles over the limit" in html
