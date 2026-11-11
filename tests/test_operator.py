"""The operator's pages, and calibration by drive-bys from them: draw the
road (a draft: passes are tracked but get no speed), drive past twice at
known speeds, pick the passes, and the scale is fitted."""

import dataclasses
import json
import re
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest

from safe_streets.operator import auth
from safe_streets.track import Track
from safe_streets.web import create_app
from test_auth import PASSWORD, csrf_of, sign_in

T0 = datetime(2026, 11, 10, 20, 0, tzinfo=timezone.utc)
MPH = 0.44704
PX_PER_M = 60.0  # the scene's true scale along the road


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("192.168.1.20", 5000),  # prepublish: allow
                                                           raise_app_exceptions=False),
                             base_url="https://cam.example")


@pytest.fixture
async def op(settings, db):
    await auth.set_password(db, PASSWORD)
    app = create_app(settings, db, None, None, None, now=lambda: T0)
    async with client(app) as c:
        await sign_in(c)
        yield c


def trail(mph: float, direction=1, y=600.0, n=15, fps=12.0):
    v = mph * MPH * PX_PER_M * direction
    x0 = 300 if direction > 0 else 1600
    return [(x0 + v * i / fps, y + 0.5 * i, i / fps) for i in range(n)]


async def add_drive(db, mph: float, ts: datetime, **kw):
    t = Track(box=(0, 0, 1, 1), trail=trail(mph, **kw), track_id=1, cls="car", conf=0.9, vote="car", mph=None, n=15,
              fps=12.0, seconds=14 / 12, metres=None, length_m=None, direction=kw.get("direction", 1), road=[], crop=None,
              limit=20.0)
    return await db.add_pass(event_id=f"e{ts:%H%M%S}", ts=ts.isoformat(), cls="car", direction=kw.get("direction", 1),
                             mph=None, n=15, track_json=json.dumps(t.to_json()))


ROAD = [[0, 500], [1920, 500], [1920, 800], [0, 800]]


async def test_the_passes_table_shows_shots_and_exact_times(op, db, settings):
    shots = settings.data_dir / "shots"
    shots.mkdir(parents=True)
    (shots / "a.jpg").write_bytes(b"\xff\xd8\xff")
    pid = await db.add_pass(event_id="e1", ts="2026-11-10T19:04:05+00:00", cls="car", direction=1, mph=27.3,
                            plus_minus=1.2, shot_path=str(shots / "a.jpg"))
    body = (await op.get("/operator/?day=2026-11-10")).text
    assert "11:04:05" in body and "27.3" in body and f"/operator/shot/{pid}.jpg" in body


async def test_a_shot_outside_the_data_folder_is_never_served(op, db, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "x.jpg"
    outside.write_bytes(b"\xff\xd8\xff")
    pid = await db.add_pass(event_id="e1", ts="2026-11-10T19:00:00+00:00", cls="car", direction=1, mph=20.0,
                            shot_path=str(outside))
    assert (await op.get(f"/operator/shot/{pid}.jpg")).status_code == 404


async def test_settings_hide_secrets(op, settings):
    body = (await op.get("/operator/settings")).text
    assert "SPEED_LIMIT" in body and "PROTECT_PASSWORD" in body
    assert "<td>p</td>" not in body  # conftest's PROTECT_PASSWORD


async def test_the_log_viewer_reads_the_log_file(settings, db, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "app.log").write_text("line one\nbatch processed=2\nline <three>\n")
    await auth.set_password(db, PASSWORD)
    app = create_app(dataclasses.replace(settings, log_dir=str(logs)), db, None, None, None, now=lambda: T0)
    async with client(app) as c:
        await sign_in(c)
        body = (await c.get("/operator/logs")).text
        assert "batch processed=2" in body and "line &lt;three&gt;" in body
        assert "line one" not in (await c.get("/operator/logs?q=batch")).text


async def test_drive_bys_calibrate_the_scale(op, db):
    page = await op.get("/operator/calibrate")
    csrf = re.search(r'CSRF = "([^"]+)"', page.text).group(1)
    h = {"X-CSRF-Token": csrf}
    r = await op.post("/operator/calibrate/scene", json={"road": ROAD, "night_lane": [425, 440], "occluders": [1235],
                                                         "mask": [[[0, 0], [100, 0], [100, 100]]]}, headers=h)
    assert r.status_code == 200 and r.json()["status"] == "draft"
    a = await add_drive(db, 15, T0)
    b = await add_drive(db, 25, T0 + timedelta(minutes=3), direction=-1)
    page = await op.get("/operator/calibrate")
    assert f'data-id="{a}"' in page.text and f'data-id="{b}"' in page.text

    one = await op.post("/operator/calibrate/drives", json={"drives": [{"id": a, "speed": 15}]}, headers=h)
    assert one.status_code == 400 and "two" in one.text
    r = await op.post("/operator/calibrate/drives", json={"drives": [{"id": a, "speed": 15}, {"id": b, "speed": 25}]},
                      headers=h)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["px_per_m"] == pytest.approx(PX_PER_M, rel=0.01)
    assert j["readings"] == pytest.approx([15, 25], abs=0.2)
    row = await db.active_calibration()
    assert row["status"] == "active" and json.loads(row["road"]) == ROAD
    assert json.loads(row["night_lane"]) == [425, 440] and json.loads(row["occluders"]) == [1235]
    trusted = json.loads(row["trusted"])
    assert min(p[1] for p in trusted) < 600 < max(p[1] for p in trusted)  # the drives' wheel line ± 30 px
    assert np.allclose(json.loads(row["H"]), np.diag([1 / PX_PER_M, 1 / PX_PER_M, 1]), rtol=0.01)


async def test_drives_in_kmh(settings, db):
    await auth.set_password(db, PASSWORD)
    app = create_app(dataclasses.replace(settings, units="kmh", speed_limit=30.0), db, None, None, None,
                     now=lambda: T0)
    async with client(app) as c:
        await sign_in(c)
        csrf = re.search(r'CSRF = "([^"]+)"', (await c.get("/operator/calibrate")).text).group(1)
        await c.post("/operator/calibrate/scene", json={"road": ROAD}, headers={"X-CSRF-Token": csrf})
        a, b = await add_drive(db, 15, T0), await add_drive(db, 25, T0 + timedelta(minutes=1))
        r = await c.post("/operator/calibrate/drives", json={"drives": [{"id": a, "speed": 15 * 1.609344},
                                                                        {"id": b, "speed": 25 * 1.609344}]},
                         headers={"X-CSRF-Token": csrf})
    assert r.json()["px_per_m"] == pytest.approx(PX_PER_M, rel=0.01) and r.json()["unit"] == "km/h"


async def test_calibration_posts_need_the_csrf_token(op):
    r = await op.post("/operator/calibrate/scene", json={"road": ROAD})
    assert r.status_code == 403


async def test_a_draft_calibration_tracks_without_speeds(db, settings, tmp_path):
    """Under a draft (the road, no scale yet) passes are kept with their
    track, for picking drives, and get no speed."""
    from safe_streets.geometry import Calibration
    from safe_streets.pipeline import process_event
    from safe_streets.synth import render_clip, uneven_times
    from test_synth_pipeline import T0 as S0, SynthProtect, TruthVision
    clip = tmp_path / "src.mp4"
    sc = render_clip(clip, speed_mps=8.9408, frame_times=uneven_times(72, 12, 0.3, seed=2))
    draft = Calibration(1, [], [], np.eye(3), sc.road, "", "draft")
    await db.upsert_event("w1", S0.isoformat(), (S0 + timedelta(seconds=10)).isoformat())
    await process_event(await db.get_event("w1"), source=SynthProtect(clip), vision=TruthVision(sc, clip), cal=draft,
                        settings=settings, db=db)
    rows = await db.passes_list()
    assert len(rows) == 1 and rows[0]["mph"] is None and rows[0]["mps"] is None
    assert len(json.loads(rows[0]["track_json"])["trail"]) > 10


async def test_reset_password_from_the_command_line(monkeypatch, tmp_path, capsys):
    from safe_streets import main
    from safe_streets.db import Database
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = Database(tmp_path / "safe_streets.db")
    await db.init()
    await auth.set_password(db, PASSWORD)
    epoch = (await db.get_operator())["session_epoch"]
    link = await main.reset_password()
    op = await db.get_operator()
    assert op["password_hash"] is None and op["session_epoch"] == epoch + 1
    assert re.search(r"/operator/setup\?token=[\w-]{40,}$", link)
    await db.close()
