import json
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from safe_streets import charts

TZ = "America/Los_Angeles"
DAY = date(2026, 11, 7)


def utc(hour, minute=0):
    """An ISO UTC time for DAY at this local hour (PST is UTC-8)."""
    return datetime(2026, 11, 7, hour, minute, tzinfo=ZoneInfo(TZ)).astimezone(ZoneInfo("UTC")).isoformat()


def test_hour_labels():
    assert charts.hour_label(0) == "12–1 am"
    assert charts.hour_label(11) == "11 am–12 pm"
    assert charts.hour_label(14) == "2–3 pm"
    assert charts.hour_label(23) == "11 pm–12 am"


def test_nice_max():
    assert [charts.nice_max(v) for v in (0, 1, 3, 7, 12, 46, 51)] == [1, 1, 5, 10, 20, 50, 100]


def test_cars_by_hour_splits_over_and_under_by_local_hour():
    rows = [{"ts": utc(8, 5), "mph": 18.0}, {"ts": utc(8, 50), "mph": 26.0}, {"ts": utc(8, 59), "mph": None},
            {"ts": utc(15), "mph": 31.0},
            {"ts": datetime(2026, 11, 8, 1, tzinfo=ZoneInfo(TZ)).isoformat(), "mph": 40.0}]  # the next day
    out = charts.cars_by_hour(rows, tz=TZ, limit=20, day=DAY, now=datetime(2026, 11, 7, 15, 30, tzinfo=ZoneInfo(TZ)))
    assert out["total"] == 4 and out["over"] == 2
    eight, three = out["chart"]["hours"]
    assert (eight["h"], eight["under"], eight["over"]) == (8, 2, 1)  # no speed counts with "at or under"
    assert [s["key"] for s in eight["segs"]] == ["under", "over"]
    assert three["title"] == "3–4 pm (so far): 1 car, 1 over 20 mph"
    assert out["peak"]["total"] == 3 and out["peak"]["label"] == "8–9 am"
    assert [g["label"] for g in out["chart"]["grid"]] == ["0", "2.5", "5"]


def test_cars_by_hour_empty_day():
    assert charts.cars_by_hour([], tz=TZ, limit=20, day=DAY)["chart"] is None


def test_shares_by_hour_splits_each_hour_by_speed_band():
    rows = [{"ts": utc(8, m), "mph": mph} for m, mph in enumerate((18.0, 18.0, None, None, 22.0, 22.0, 26.0, 26.0,
                                                                  33.0, 33.0))]
    rows.append({"ts": utc(15), "mph": 40.0})
    out = charts.shares_by_hour(rows, tz=TZ, limit=20, day=DAY, now=datetime(2026, 11, 7, 15, 30, tzinfo=ZoneInfo(TZ)))
    eight, three = out["chart"]["hours"]
    # Bottom up, slowest first; no speed counts with "at or under", like the count view.
    assert [s["key"] for s in eight["segs"]] == ["under", "over-0", "over-5", "over-10"]
    assert eight["title"] == ("8–9 am: 10 cars, 60% over 20 mph\nat or under 20 (or no speed): 4 (40%)\n"
                              "20–25: 2 (20%)\n25–30: 2 (20%)\n30–35: 2 (20%)")
    assert three["title"] == "3–4 pm (so far): 1 car, 100% over 20 mph\n35+: 1 (100%)"
    # An hour of under 10 cars is drawn lighter: one car moves its shares more than 10%.
    assert (eight["few"], three["few"]) == (False, True)
    assert [g["label"] for g in out["chart"]["grid"]] == ["0", "50%", "100%"]
    # The key: the day's share in each band.
    assert [(k["key"], k["label"], k["pct"]) for k in out["legend"]] == [
        ("under", "at or under 20 (or no speed)", "36%"), ("over-0", "20–25", "18%"),
        ("over-5", "25–30", "18%"), ("over-10", "30–35", "18%"), ("over-15", "35+", "9%")]
    # The hour with the most over the limit, among hours with enough cars to say.
    assert (out["most"]["label"], out["most"]["pct"]) == ("8–9 am", "60%")


def test_every_share_column_fills_the_plot():
    rows = [{"ts": utc(9, m), "mph": mph} for m, mph in enumerate((12.0, 19.0, 21.0, 24.0, 28.0, 36.0, 50.0))]
    col = charts.shares_by_hour(rows, tz=TZ, limit=20, day=DAY)["chart"]["hours"][0]
    ys = [float(y) for s in col["segs"] for y in re.findall(r"[,V](-?[\d.]+)", s["d"])]
    assert min(ys) == charts.TOP  # the top segment reaches 100%


def test_no_hour_stands_out_when_none_has_enough_cars():
    out = charts.shares_by_hour([{"ts": utc(2, m), "mph": 45.0} for m in range(9)], tz=TZ, limit=20, day=DAY)
    assert out["chart"]["hours"][0]["few"] and out["most"] is None
    assert out["chart"]["few"] == 10  # the key's "lighter: under 10 cars"


def test_shares_by_hour_empty_day():
    assert charts.shares_by_hour([], tz=TZ, limit=20, day=DAY)["chart"] is None


def test_fastest_by_hour_shows_each_hours_fastest_car():
    """The tile in the 7-day strip's place (2026-11-08): each hour's
    fastest car, coloured by its band; a rough estimate says so."""
    rows = [{"ts": utc(8, 5), "mph": 18.0, "plus_minus": 1.0, "flags": ""},
            {"ts": utc(8, 40), "mph": 31.2, "plus_minus": 1.0, "flags": ""},
            {"ts": utc(8, 50), "mph": None, "plus_minus": None, "flags": "short"},
            {"ts": utc(9, 10), "mph": 19.0, "plus_minus": 1.0, "flags": ""},
            {"ts": utc(12), "mph": None, "plus_minus": None, "flags": "short"},  # no speed: no column
            {"ts": utc(15, 7), "mph": 44.6, "plus_minus": 4.0, "flags": "short"},
            {"ts": datetime(2026, 11, 8, 1, tzinfo=ZoneInfo(TZ)).isoformat(), "mph": 60.0,
             "plus_minus": 1.0, "flags": ""}]  # the next day
    out = charts.fastest_by_hour(rows, tz=TZ, limit=20, day=DAY,
                                 now=datetime(2026, 11, 7, 15, 30, tzinfo=ZoneInfo(TZ)))
    eight, nine, three = out["chart"]["hours"]
    assert [s["key"] for s in eight["segs"]] == ["over-10"]
    assert [s["key"] for s in nine["segs"]] == ["under"]
    assert [s["key"] for s in three["segs"]] == ["over-15"]
    assert (eight["rough"], nine["rough"], three["rough"]) == (False, False, True)
    assert eight["title"] == "8–9 am: fastest 31 mph at 8:40 am; 3 cars"
    assert three["title"] == "3–4 pm (so far): fastest 45 mph at 3:07 pm (rough estimate); 1 car"
    assert (out["fastest"]["mph"], out["fastest"]["label"]) == (45, "3–4 pm")
    # A clean top in tens, the limit drawn across.
    assert [g["label"] for g in out["chart"]["grid"]] == ["0", "25", "50"]
    assert out["chart"]["limit_y"] == 74.0  # 20 of 50 mph, up a 100-unit plot from y 114
    assert [k["key"] for k in out["keys"]] == ["under", "over-10", "over-15"]


def test_fastest_by_hour_top_covers_the_limit():
    rows = [{"ts": utc(8), "mph": 12.0, "plus_minus": 1.0, "flags": ""}]
    out = charts.fastest_by_hour(rows, tz=TZ, limit=20, day=DAY)
    assert out["chart"]["grid"][-1]["label"] == "20"


def test_fastest_by_hour_empty_day():
    rows = [{"ts": utc(8), "mph": None, "plus_minus": None, "flags": "short"}]
    assert charts.fastest_by_hour(rows, tz=TZ, limit=20, day=DAY)["chart"] is None
    assert charts.fastest_by_hour([], tz=TZ, limit=20, day=DAY)["chart"] is None


def test_pct():
    # Never "0%" or "100%" unless it's none or all.
    assert [charts.pct(n, 300) for n in (0, 1, 2, 150, 298, 299, 300)] == ["0%", "<1%", "1%", "50%", "99%", ">99%", "100%"]
    assert charts.pct(1, 8) == "13%"  # half up, not to even


def test_columns_are_square_at_the_base_and_rounded_on_top_only():
    d = charts._column(10, 8, 100, 50, True)
    assert d.startswith("M10,100V53") and "Q" in d and d.endswith("V100Z")
    assert "Q" not in charts._column(10, 8, 100, 50, False)
    assert charts._column(10, 8, 100, 100, True) == ""  # zero height draws nothing


def ev(hour, minute, ms, car, **stages):
    return {"status": "done" if car else "no_vehicle", "start_ts": utc(hour, minute), "ms": ms,
            "stages_json": json.dumps({**stages, "ms": ms}) if stages else None}


def test_processing_three_designs():
    rows = [ev(9, 0, 2000, False, export_ms=200, gate_ms=1500, night_ms=100),
            ev(9, 10, 8000, True, export_ms=300, gate_ms=1700, track_ms=5500),
            ev(9, 20, 1800, False, export_ms=150, gate_ms=1500),
            ev(14, 0, 12400, True),  # analysed before the breakdown: all "other"
            {"status": "failed", "start_ts": utc(10), "ms": 900, "stages_json": None}]
    out = charts.processing(rows, tz=TZ, day=DAY)
    assert out["count"] == 4 and out["n_car"] == 2 and out["n_empty"] == 2
    assert out["median_empty"] == 1.9 and out["median_car"] == 10.2
    # Trend: two lines, the car one broken across the empty hours, each with an end label.
    keys = {s["key"]: s for s in out["trend"]["series"]}
    assert keys["car"]["d"].count("M") == 2 and keys["car"]["end"]["label"] == "12 s"
    assert keys["empty"]["end"]["label"] == "1.9 s"
    hit9 = next(h for h in out["trend"]["hits"] if "9–10 am" in h["title"])
    assert hit9["title"] == "9–10 am: 3 events, empty median 1.9 s, with a car median 8 s, slowest 8 s"
    # Stages: per hour, stacked bottom up; old events land in "other".
    st = out["stages"]
    nine = next(c for c in st["cols"] if c["title"].startswith("9–10 am"))
    # Bottom up; night (0.1 s) is too thin to draw but stays in the title and legend.
    assert [s["key"] for s in nine["segs"]] == ["export", "gate", "track", "other"]
    assert "night 0.1 s" in nine["title"] and "export 0.7 s" in nine["title"]
    assert [g["label"] for g in st["grid"]] == ["0s", "10s", "20s"]
    assert {k["key"] for k in st["legend"]} == {"export", "gate", "track", "night", "other"}
    # Events: one dot path per kind.
    assert {p["key"] for p in out["events"]["paths"]} == {"empty", "car"}
    assert out["events"]["paths"][0]["d"].count("h0") + out["events"]["paths"][1]["d"].count("h0") == 4


def test_processing_caps_the_dots_and_counts_the_rest():
    rows = [ev(9, i, 2000, False) for i in range(60)] + [ev(10, 0, 95000, True)]
    out = charts.processing(rows, tz=TZ, day=DAY)
    assert out["events"]["above"] == {"n": 1, "max": "95"}
    assert out["stages"] is None  # no breakdown stored


def test_processing_without_any_stage_breakdown_has_no_stage_chart():
    out = charts.processing([ev(9, 0, 2000, False)], tz=TZ, day=DAY)
    assert out["stages"] is None and out["trend"] and out["events"]


def test_processing_empty_day():
    out = charts.processing([], tz=TZ, day=DAY)
    assert out["count"] == 0 and out["trend"] is None


def test_processing_by_cpu_and_gpu_per_stage():
    rows = [ev(7, 0, 9000, True),  # before CPU was recorded: left out of CPU and GPU
            ev(9, 0, 2000, False, export_ms=200, gate_ms=1500, gate_cpu_ms=1400, gate_gpu_ms=30),
            ev(9, 10, 8000, True, export_ms=300, gate_ms=1700, gate_cpu_ms=1600, gate_gpu_ms=40,
               track_ms=5500, track_cpu_ms=2600, track_gpu_ms=600),
            ev(10, 0, 2400, False, export_ms=200, gate_ms=1500, gate_cpu_ms=1500, night_ms=600,
               night_cpu_ms=550)]  # no GPU open: no gpu keys
    out = charts.processing(rows, tz=TZ, day=DAY)
    assert out["count"] == 4  # the wall time has every event
    cpu, gpu = out["cpu"], out["gpu"]
    assert cpu["count"] == 3 and cpu["coverage"] == "CPU recorded for 3 of 4 events, from 9:00 am"
    assert round(cpu["median_empty"], 3) == 1.725 and cpu["median_car"] == 4.2
    # Only the stages CPU was measured in: no export, no "other".
    assert {k["key"] for k in cpu["stages"]["legend"]} == {"gate", "track", "night"}
    nine = next(c for c in cpu["stages"]["cols"] if c["title"].startswith("9–10 am"))
    assert nine["title"] == "9–10 am: 5.6 s of CPU — gate 3 s, track 2.6 s"
    hit9 = next(h for h in cpu["trend"]["hits"] if "9–10 am" in h["title"])
    assert hit9["title"] == "9–10 am: 2 events, empty median 1.4 s CPU, with a car median 4.2 s CPU, most 4.2 s CPU"
    assert gpu["count"] == 2 and gpu["coverage"] == "GPU recorded for 2 of 4 events, from 9:00 am"
    assert {k["key"] for k in gpu["stages"]["legend"]} == {"gate", "track"}
    assert gpu["busy"] == "0.7 s"


def test_processing_without_cpu_or_gpu_has_neither():
    out = charts.processing([ev(9, 0, 2000, False, export_ms=200, gate_ms=1500)], tz=TZ, day=DAY)
    assert out["cpu"] is None and out["gpu"] is None and out["coverage"] is None


