"""Defaults that run anywhere: the speed limit and time zone are the
operator's to say, speeds show in mph or km/h, and nothing paces itself by
CPU pressure unless asked."""

import asyncio
from datetime import date, datetime, timezone

import pytest

from safe_streets.config import Settings

MPH = 0.44704  # m/s


def test_start_without_a_speed_limit_fails_naming_it(monkeypatch, tmp_path):
    monkeypatch.delenv("SPEED_LIMIT", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from safe_streets import main
    with pytest.raises(SystemExit, match="SPEED_LIMIT"):
        asyncio.run(main.serve())


def test_start_without_a_time_zone_fails_naming_it(monkeypatch):
    monkeypatch.setenv("SPEED_LIMIT", "20")
    monkeypatch.delenv("TZ", raising=False)
    with pytest.raises(SystemExit, match="TZ"):
        Settings().require()


def test_a_speed_limit_and_time_zone_are_enough(monkeypatch):
    monkeypatch.setenv("SPEED_LIMIT", "20")
    s = Settings()
    s.require()
    assert s.speed_limit == 20.0 and s.units == "mph" and s.unit_label == "mph"


def test_kmh_shows_a_20_mph_car_as_32():
    s = Settings(units="kmh", speed_limit=30.0)
    assert s.unit_label == "km/h"
    assert round(s.speed(20 * MPH)) == 32
    assert Settings(units="mph", speed_limit=20.0).speed(20 * MPH) == pytest.approx(20.0)
    assert s.speed(None) is None


def test_kmh_bands_by_the_kmh_limit():
    from safe_streets.charts import cars_by_hour
    s = Settings(units="kmh", speed_limit=30.0, tz="America/Los_Angeles")
    rows = [{"ts": "2026-11-02T19:21:00+00:00", "mps": 20 * MPH, "mph": 20.0, "plus_minus": 1.0, "flags": ""}]
    out = cars_by_hour(rows, tz=s.tz, limit=s.speed_limit, units=s.units, day=date(2026, 11, 2),
                       now=datetime(2026, 11, 3, tzinfo=timezone.utc))
    col = next(c for c in out["chart"]["hours"] if c["h"] == 11)
    assert "1 over 30 km/h" in col["title"]


def test_an_unknown_unit_is_refused(monkeypatch):
    monkeypatch.setenv("SPEED_LIMIT", "20")
    with pytest.raises(SystemExit, match="UNITS"):
        Settings(units="furlongs").require()


def test_defaults_need_no_gpu_and_no_pacing(monkeypatch):
    for k in ("YOLO_DEVICE", "MAX_CPU_PRESSURE"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.yolo_device == "cpu"
    assert s.max_cpu_pressure is None


async def test_without_max_cpu_pressure_the_batcher_never_reads_it(settings, db):
    from safe_streets.batcher import Batcher

    def never():
        raise AssertionError("read the CPU pressure or load")

    b = Batcher(settings, db, None, None, load_fn=never, pressure_fn=never)
    assert settings.max_cpu_pressure is None
    assert b._busy() is False


async def test_a_pass_is_stored_in_metres_per_second(db):
    pid = await db.add_pass(event_id="e", ts="2026-11-02T19:21:00+00:00", cls="car", direction=1, mph=20.0)
    row = await db._one("SELECT mps FROM passes WHERE id = ?", (pid,))
    assert row["mps"] == pytest.approx(20 * MPH)


def test_env_example_names_every_setting():
    import re
    from pathlib import Path
    root = Path(__file__).parent.parent
    names = set(re.findall(r'(?:_env|_opt_float|os\.environ(?:\.get)?)\(?\[?"([A-Z0-9_]+)"', (root / "src/safe_streets/config.py").read_text()))
    example = (root / ".env.example").read_text()
    missing = sorted(n for n in names if not re.search(rf"^#? ?{n}=", example, re.M))
    assert not missing, missing


def test_no_night_lane_line_unless_given(monkeypatch):
    """The reference install's line (425/440 px) means nothing on another camera."""
    for k in ("NIGHT_LANE_Y0", "NIGHT_LANE_Y1"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.night_lane_y0 is None and s.night_lane_y1 is None


def test_the_detector_keeps_boxes_from_0_25(monkeypatch):
    """2026-11-10: RTMDet-tiny at 0.25 missed 20 of the labelled real
    cars against 28 at 0.35 (most of the gain at dusk, none lost by day),
    and lines up with ByteTrack's high threshold (0.25)."""
    monkeypatch.delenv("YOLO_CONFIDENCE", raising=False)
    from safe_streets.bytetrack import Params
    assert Settings().yolo_confidence == 0.25 == Params().track_thresh
