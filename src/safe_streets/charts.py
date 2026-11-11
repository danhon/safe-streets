"""The dashboard's day charts beside the speed histogram: each hour's fastest
car, cars per hour, and the processing time of that day's events in three designs (2026-11-07:
"options for sparklines or graphs"; the tile switches between them), each in
three measures (2026-11-07: "break down cpu/gpu where it makes sense"):
time (wall clock, every stage), CPU (the app's CPU time, every thread's, so
it can beat the wall clock) and GPU (the GPU engines' busy time). CPU and GPU
are per stage where they were measured (gate, track, night; not the export,
which waits on Protect), so they stack by stage but not on top of the wall
time: they overlap it, and each other. Each has its own axis.

Everything is laid out here, in SVG units, and drawn by index.html, like
histogram.py. Columns have a 2-unit gap and a rounded top, square at the
baseline; one axis each; hairline gridlines at clean values. Colours are
CSS classes (index.html), checked with the dataviz validator: at or under /
over is blue / red; the stages are yellow, blue, orange, aqua in stack order;
car / empty events are orange / blue.
"""

import json
import math
import statistics
from datetime import date, datetime
from zoneinfo import ZoneInfo

from .config import UNIT_MPS
from .speed import band, confidence, whole

WIDTH, TOP, PLOT_H = 320, 14, 100  # like histogram.py
LEFT = 20  # room for the y-axis labels
BOTTOM = 16  # the x-axis labels
GAP = 2
RADIUS = 3
HOURS = 24
STAGES = ("export", "gate", "track", "night", "classify")
# Each measure's word in titles and keys ("" for time). CPU per stage since
# 2026-11-07 07:09 (9a2dc72); GPU per stage since it was deployed that day.
MEASURES = {"time": "", "cpu": "CPU", "gpu": "GPU"}
HOUR_TICKS = {0: "12a", 6: "6a", 12: "12p", 18: "6p"}
# The share view's stack, bottom up: at or under the limit, then the bands over it.
SHARE_BANDS = ("under", "over-0", "over-5", "over-10", "over-15")
FEW_CARS = 10  # an hour with fewer is drawn lighter in the share view: one car moves it >10%


def band_labels(limit: float) -> dict[str, str]:
    """The over-limit bands' names (`speed.band`): "20–25" … "35+"."""
    n = int(limit)
    return {"over-0": f"{n}–{n + 5}", "over-5": f"{n + 5}–{n + 10}", "over-10": f"{n + 10}–{n + 15}",
            "over-15": f"{n + 15}+"}


def shown(r, units: str = "mph") -> float | None:
    """A row's speed in UNITS: from its m/s, or its mph in rows that lack it."""
    if units == "mph":
        return r["mph"]
    mps = r["mps"] if "mps" in r.keys() else None
    if mps is None:
        mps = None if r["mph"] is None else r["mph"] * UNIT_MPS["mph"]
    return None if mps is None else mps / UNIT_MPS[units]


def unit_label(units: str) -> str:
    return "km/h" if units == "kmh" else "mph"


def _local(ts: str, tz: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo(tz))


def hour_label(h: int) -> str:
    """"2–3 pm", "11 am–12 pm"; midnight and noon as 12."""
    a, b = h % 24, (h + 1) % 24
    num = lambda x: f"{(x % 12) or 12}"  # noqa: E731
    half = lambda x: "am" if x < 12 else "pm"  # noqa: E731
    if half(a) == half(b):
        return f"{num(a)}–{num(b)} {half(b)}"
    return f"{num(a)} {half(a)}–{num(b)} {half(b)}"


def nice_max(v: float) -> float:
    """The smallest 1, 2 or 5 × 10^k at or above v (at least 1)."""
    if v <= 1:
        return 1
    k = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        if v <= m * k:
            return m * k
    return 10 * k


def _grid(top_value: float, fmt) -> list[dict]:
    """Gridlines at 0, half and the top (clean values: top is nice_max)."""
    out = []
    for v in (0, top_value / 2, top_value):
        y = TOP + PLOT_H - v / top_value * PLOT_H
        out.append({"y": round(y, 2), "label": fmt(v)})
    return out


def _column(x: float, w: float, y0: float, y1: float, rounded: bool) -> str:
    """A path from y0 (the bottom) up to y1, with a rounded top if asked."""
    h = y0 - y1
    if h <= 0:
        return ""
    r = min(RADIUS, w / 2, h) if rounded else 0
    x, w, y0, y1 = (round(v, 2) for v in (x, w, y0, y1))
    if not r:
        return f"M{x},{y0}V{y1}H{round(x + w, 2)}V{y0}Z"
    r = round(r, 2)
    return (f"M{x},{y0}V{round(y1 + r, 2)}Q{x},{y1} {round(x + r, 2)},{y1}"
            f"H{round(x + w - r, 2)}Q{round(x + w, 2)},{y1} {round(x + w, 2)},{round(y1 + r, 2)}V{y0}Z")


def _slots():
    slot = (WIDTH - LEFT) / HOURS
    return slot, min(24, slot - GAP)


def _x_ticks() -> list[dict]:
    slot, _ = _slots()
    return [{"x": round(LEFT + h * slot, 2), "label": lab} for h, lab in HOUR_TICKS.items()]


def _stack(h: int, parts: list[tuple[str, float]], scale: float) -> list[dict]:
    """Segments (bottom first) of hour h's column: a 2-unit gap between them,
    only the top one rounded."""
    slot, w = _slots()
    x = LEFT + h * slot + (slot - w) / 2
    base = TOP + PLOT_H
    shown = [(k, v) for k, v in parts if v > 0]
    segs, y = [], base
    for i, (key, v) in enumerate(shown):
        y_top = y - v * scale
        bottom = y if i == 0 else y - GAP / 2
        top = y_top + (GAP / 2 if i < len(shown) - 1 else 0)
        if bottom - top > 0.3:
            segs.append({"key": key, "d": _column(x, w, bottom, top, i == len(shown) - 1)})
        y = y_top
    return segs


def _hit(h: int) -> dict:
    slot, _ = _slots()
    return {"x": round(LEFT + h * slot, 2), "w": round(slot, 2), "y": TOP, "h": PLOT_H}


def _secs(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}".rstrip("0").rstrip(".")


def _dur(seconds: float) -> str:
    """Seconds as "45 s", "12.5 min" or "1.2 h"."""
    if seconds < 60:
        return f"{_secs(seconds)} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def cars_by_hour(rows, *, tz: str, limit: float, day: date, now: datetime | None = None, units: str = "mph") -> dict:
    """rows: the day's speeds_between rows. Each hour's cars, at or under the
    limit (or without a speed) below, over it on top."""
    under, over = [0] * HOURS, [0] * HOURS
    for r in rows:
        d = _local(r["ts"], tz)
        if d.date() != day:
            continue
        v = shown(r, units)
        if v is not None and v > limit:
            over[d.hour] += 1
        else:
            under[d.hour] += 1
    totals = [u + o for u, o in zip(under, over)]
    out = {"total": sum(totals), "over": sum(over), "chart": None}
    if not out["total"]:
        return out
    top = nice_max(max(totals))
    scale = PLOT_H / top
    current = now.hour if now is not None and now.date() == day else None
    hours = []
    for h in range(HOURS):
        if not totals[h]:
            continue
        hours.append({
            "h": h, "under": under[h], "over": over[h], "total": totals[h],
            "segs": _stack(h, [("under", under[h]), ("over", over[h])], scale),
            "hit": _hit(h),
            "title": (f"{hour_label(h)}{' (so far)' if h == current else ''}: {totals[h]} "
                      f"car{'s' if totals[h] != 1 else ''}, {over[h]} over {limit:g} {unit_label(units)}"),
        })
    peak = max(hours, key=lambda x: (x["total"], -x["h"]))
    slot, _ = _slots()
    out["peak"] = {"h": peak["h"], "label": hour_label(peak["h"]), "total": peak["total"],
                   "x": round(LEFT + (peak["h"] + 0.5) * slot, 2),
                   "y": round(TOP + PLOT_H - peak["total"] * scale - 3, 2)}
    out["chart"] = {"width": WIDTH, "height": TOP + PLOT_H + BOTTOM, "base": TOP + PLOT_H, "left": LEFT,
                    "hours": hours, "grid": _grid(top, lambda v: f"{v:g}"), "ticks": _x_ticks()}
    return out


def _ampm(d: datetime) -> str:
    return d.strftime("%-I:%M ") + d.strftime("%p").lower()


def fastest_by_hour(rows, *, tz: str, limit: float, day: date, now: datetime | None = None, units: str = "mph") -> dict:
    """rows: the day's speeds_between rows. Each hour's fastest car
    (2026-11-08: in the 7-day strip's place), drawn like cars by hour: one
    column an hour, coloured by its band (`speed.band`, unrounded) as in the
    share view, blue at or under the limit, with the limit drawn across. A
    rough estimate says so in its title but isn't drawn lighter: at night
    nearly every column is rough, and a lighter dark red reads as a slower
    band. Speeds read as the posts round them (`speed.whole`). An hour whose
    cars have no speed has no column."""
    best: list[dict | None] = [None] * HOURS
    cars = [0] * HOURS
    for r in rows:
        d = _local(r["ts"], tz)
        if d.date() != day:
            continue
        cars[d.hour] += 1
        v = shown(r, units)  # "mph" below is the speed in UNITS
        if v is not None and (best[d.hour] is None or v > best[d.hour]["mph"]):
            flags = [f for f in (r["flags"] or "").split(",") if f]
            best[d.hour] = {"mph": v, "at": d,
                            "rough": confidence(flags, r["plus_minus"])[0] == "rough"}
    names = {"under": f"at or under {limit:g}", **band_labels(limit)}
    out = {"chart": None, "fastest": None, "keys": []}
    known = [(h, b) for h, b in enumerate(best) if b is not None]
    if not known:
        return out
    top = 10 * math.ceil(max(max(whole(b["mph"]) for _, b in known), limit) / 10)
    scale = PLOT_H / top
    current = now.hour if now is not None and now.date() == day else None
    hours, used = [], set()
    for h, b in known:
        key = band(b["mph"], limit) or "under"
        used.add(key)
        hours.append({
            "h": h, "mph": whole(b["mph"]), "rough": b["rough"],
            "segs": _stack(h, [(key, b["mph"])], scale), "hit": _hit(h),
            "title": (f"{hour_label(h)}{' (so far)' if h == current else ''}: fastest {whole(b['mph'])} {unit_label(units)} "
                      f"at {_ampm(b['at'])}{' (rough estimate)' if b['rough'] else ''}; "
                      f"{cars[h]} car{'s' if cars[h] != 1 else ''}"),
        })
    h, b = max(known, key=lambda x: (x[1]["mph"], -x[0]))
    slot, _ = _slots()
    out["fastest"] = {"h": h, "label": hour_label(h), "mph": whole(b["mph"]), "x": round(LEFT + (h + 0.5) * slot, 2),
                      "y": round(TOP + PLOT_H - b["mph"] * scale - 3, 2)}
    out["keys"] = [{"key": k, "label": names[k]} for k in SHARE_BANDS if k in used]
    out["chart"] = {"width": WIDTH, "height": TOP + PLOT_H + BOTTOM, "base": TOP + PLOT_H, "left": LEFT,
                    "hours": hours, "grid": _grid(top, lambda v: f"{v:g}"), "ticks": _x_ticks(),
                    "limit_y": round(TOP + PLOT_H - limit * scale, 2)}
    return out


def pct(n: int, total: int) -> str:
    """n of total as a whole percent, half up. Never "0%" or "100%" unless
    it's none or all: "<1%" and ">99%" instead."""
    if not n or n == total:
        return "0%" if not n else "100%"
    p = math.floor(100 * n / total + 0.5)
    return "<1%" if p < 1 else ">99%" if p > 99 else f"{p}%"


def shares_by_hour(rows, *, tz: str, limit: float, day: date, now: datetime | None = None, units: str = "mph") -> dict:
    """rows: the day's speeds_between rows. The cars-by-hour tile's other view
    (2026-11-07): each hour's cars as shares, by speed band, slowest at
    the bottom: at or under the limit (or without a speed, as in cars_by_hour),
    then the four over-limit bands (`speed.band`, unrounded). Every
    column fills the plot, so an hour of fewer than FEW_CARS, whose shares
    swing on one car, is drawn lighter and can't be the hour with the most
    over the limit."""
    counts = [dict.fromkeys(SHARE_BANDS, 0) for _ in range(HOURS)]
    for r in rows:
        d = _local(r["ts"], tz)
        if d.date() == day:
            counts[d.hour][band(shown(r, units), limit) or "under"] += 1
    names = {"under": f"at or under {limit:g} (or no speed)", **band_labels(limit)}
    day_n = {k: sum(c[k] for c in counts) for k in SHARE_BANDS}
    total = sum(day_n.values())
    out = {"total": total, "chart": None, "most": None,
           "legend": [{"key": k, "label": names[k], "n": day_n[k], "pct": pct(day_n[k], total)}
                      for k in SHARE_BANDS]}
    if not total:
        return out
    current = now.hour if now is not None and now.date() == day else None
    hours = []
    for h, c in enumerate(counts):
        n = sum(c.values())
        if not n:
            continue
        over = n - c["under"]
        lines = [f"{hour_label(h)}{' (so far)' if h == current else ''}: {n} car{'s' if n != 1 else ''}, "
                 f"{pct(over, n)} over {limit:g} {unit_label(units)}"]
        lines += [f"{names[k]}: {c[k]} ({pct(c[k], n)})" for k in SHARE_BANDS if c[k]]
        hours.append({"h": h, "n": n, "over": over, "few": n < FEW_CARS,
                      "segs": _stack(h, [(k, c[k]) for k in SHARE_BANDS], PLOT_H / n),
                      "hit": _hit(h), "title": "\n".join(lines)})
    enough = [x for x in hours if not x["few"] and x["over"]]
    if enough:
        m = max(enough, key=lambda x: (x["over"] / x["n"], x["n"], -x["h"]))
        slot, _ = _slots()
        out["most"] = {"h": m["h"], "label": hour_label(m["h"]), "pct": pct(m["over"], m["n"]),
                       "x": round(LEFT + (m["h"] + 0.5) * slot, 2), "y": TOP - 3}
    out["chart"] = {"width": WIDTH, "height": TOP + PLOT_H + BOTTOM, "base": TOP + PLOT_H, "left": LEFT,
                    "hours": hours, "few": FEW_CARS if any(x["few"] for x in hours) else 0,
                    "grid": _grid(100, lambda v: f"{v:g}%" if v else "0"), "ticks": _x_ticks()}
    return out


def _stages(row) -> dict:
    try:
        return json.loads(row["stages_json"]) if row["stages_json"] else {}
    except ValueError:
        return {}


def _median(xs):
    return statistics.median(xs) if xs else None


def _clock(minutes: float) -> str:
    """Minutes since midnight as "7:09 am"."""
    h, m = divmod(int(minutes), 60)
    return f"{(h % 12) or 12}:{m:02d} {'am' if h < 12 else 'pm'}"


def _amount(e, measure: str) -> float | None:
    """The event's seconds of this measure: its wall time, or the sum of its
    stages' CPU or GPU time; None when none was recorded."""
    if measure == "time":
        return e["s"]
    vals = [e["st"][f"{k}_{measure}_ms"] for k in STAGES if f"{k}_{measure}_ms" in e["st"]]
    return sum(vals) / 1000 if vals else None


def processing(rows, *, tz: str, day: date, now: datetime | None = None) -> dict:
    """rows: the day's analysed events (status, start_ts, ms, stages_json).
    Three designs of the same data: `trend` (each hour's median time, empty
    events and events with a car), `stages` (each hour's processing time by
    stage, stacked) and `events` (every event as a dot). The top level is the
    wall time; `cpu` and `gpu` hold the same for those measures (None when
    the day has none recorded), with `coverage` when only some events have it."""
    evs = []
    for r in rows:
        if r["ms"] is None or r["status"] not in ("done", "no_vehicle"):
            continue
        d = _local(r["start_ts"], tz)
        if d.date() != day:
            continue
        evs.append({"h": d.hour, "min": d.hour * 60 + d.minute + d.second / 60, "s": r["ms"] / 1000,
                    "car": r["status"] == "done", "st": _stages(r)})
    out = {"count": len(evs), "trend": None, "stages": None, "events": None, "cpu": None, "gpu": None}
    if not evs:
        return out
    current = now.hour if now is not None and now.date() == day else None
    out.update(_views(evs, "time", current))
    for m in ("cpu", "gpu"):
        out[m] = _views(evs, m, current)
    return out


def _views(evs, measure: str, current) -> dict | None:
    """The three designs for one measure, over the events that have it."""
    word = MEASURES[measure]
    unit = f" {word}" if word else ""
    have = [{**e, "s": v} for e in evs if (v := _amount(e, measure)) is not None]
    if not have:
        return None
    out = {"count": len(have), "word": word, "coverage": None}
    if len(have) < len(evs):
        out["coverage"] = (f"{word} recorded for {len(have)} of {len(evs)} events, "
                           f"from {_clock(min(e['min'] for e in have))}")
    by_hour = [[e for e in have if e["h"] == h] for h in range(HOURS)]
    car_all = [e["s"] for e in have if e["car"]]
    empty_all = [e["s"] for e in have if not e["car"]]
    out["median_car"], out["median_empty"] = _median(car_all), _median(empty_all)
    out["n_car"], out["n_empty"] = len(car_all), len(empty_all)
    out["busy"] = _dur(sum(e["s"] for e in have))
    titles = []
    for h, hs in enumerate(by_hour):
        if not hs:
            titles.append(None)
            continue
        c, e = [x["s"] for x in hs if x["car"]], [x["s"] for x in hs if not x["car"]]
        parts = [f"{hour_label(h)}{' (so far)' if h == current else ''}: {len(hs)} event{'s' if len(hs) != 1 else ''}"]
        if e:
            parts.append(f"empty median {_secs(_median(e))} s{unit}")
        if c:
            parts.append(f"with a car median {_secs(_median(c))} s{unit}")
        parts.append(f"{'most' if word else 'slowest'} {_secs(max(x['s'] for x in hs))} s{unit}")
        titles.append(", ".join(parts))
    hits = [{**_hit(h), "title": t} for h, t in enumerate(titles) if t]
    frame = {"width": WIDTH, "height": TOP + PLOT_H + BOTTOM, "base": TOP + PLOT_H, "left": LEFT,
             "ticks": _x_ticks(), "hits": hits}
    out["trend"] = _trend(by_hour, frame)
    out["stages"] = _stage_columns(by_hour, frame, current, measure)
    out["events"] = _dots(have, frame)
    return out


def _y(v: float, top: float) -> float:
    return round(TOP + PLOT_H - min(v, top) / top * PLOT_H, 2)


def _trend(by_hour, frame) -> dict:
    """Two lines of hourly medians, broken where an hour had none."""
    slot, _ = _slots()
    meds = {"car": [], "empty": []}
    for h, hs in enumerate(by_hour):
        for key, want in (("car", True), ("empty", False)):
            xs = [x["s"] for x in hs if x["car"] is want]
            meds[key].append(_median(xs))
    top = nice_max(max(v for vs in meds.values() for v in vs if v is not None))
    series = []
    for key in ("empty", "car"):
        d, pen, last = [], False, None
        for h, v in enumerate(meds[key]):
            if v is None:
                pen = False
                continue
            x = round(LEFT + (h + 0.5) * slot, 2)
            d.append(f"{'L' if pen else 'M'}{x},{_y(v, top)}")
            pen, last = True, {"x": x, "y": _y(v, top), "label": f"{_secs(v)} s"}
        if last:
            # A lone point (no neighbour) still shows: its end dot is drawn.
            series.append({"key": key, "d": "".join(d), "end": last})
    _separate(series)
    return {**frame, "series": series, "grid": _grid(top, lambda v: f"{v:g} s")}


def _separate(series) -> None:
    """End labels that would overlap move apart; each keeps its dot."""
    if len(series) == 2:
        a, b = sorted(series, key=lambda s: s["end"]["y"])
        a["end"]["ly"], b["end"]["ly"] = a["end"]["y"], b["end"]["y"]
        if b["end"]["ly"] - a["end"]["ly"] < 9:
            mid = (a["end"]["y"] + b["end"]["y"]) / 2
            a["end"]["ly"], b["end"]["ly"] = round(mid - 4.5, 2), round(mid + 4.5, 2)
    for s in series:
        s["end"].setdefault("ly", s["end"]["y"])


def _stage_columns(by_hour, frame, current, measure: str = "time") -> dict | None:
    """Each hour's processing time, stacked by stage. Wall time: events from
    before the breakdown was stored count as `other`. CPU and GPU: only the
    stages they were measured in, so no `other`."""
    suffix = "_ms" if measure == "time" else f"_{measure}_ms"
    sums = []
    totals = {k: 0.0 for k in (*STAGES, "other")}
    for hs in by_hour:
        s = {k: 0.0 for k in (*STAGES, "other")}
        for e in hs:
            known = 0.0
            for k in STAGES:
                v = e["st"].get(f"{k}{suffix}", 0) / 1000
                s[k] += v
                known += v
            if measure == "time":
                s["other"] += max(0.0, e["s"] - known)
        for k in s:
            totals[k] += s[k]
        sums.append(s)
    if not any(totals[k] for k in STAGES):
        return None
    peak = max(sum(s.values()) for s in sums)
    if peak >= 60:  # minutes on the axis
        top_label = nice_max(peak / 60)
        top, fmt = top_label * 60, (lambda v: f"{v / 60:g}m")
    else:
        top, fmt = nice_max(peak), (lambda v: f"{v:g}s")
    scale = PLOT_H / top
    cols = []
    for h, s in enumerate(sums):
        if not sum(s.values()):
            continue
        parts = [(k, s[k]) for k in (*STAGES, "other")]
        what = "processing" if measure == "time" else f"of {MEASURES[measure]}"
        title = (f"{hour_label(h)}{' (so far)' if h == current else ''}: {_dur(sum(s.values()))} {what} — "
                 + ", ".join(f"{k} {_dur(v)}" for k, v in parts if v >= 0.05))
        cols.append({"segs": _stack(h, parts, scale), "title": title, "hit": _hit(h)})
    legend = [{"key": k, "label": k, "value": _dur(totals[k])} for k in (*STAGES, "other") if totals[k] > 0]
    return {**frame, "cols": cols, "legend": legend, "grid": _grid(top, fmt)}


def _dots(evs, frame) -> dict:
    """Every event as a dot (time of day, seconds), as one path per kind:
    zero-length segments with round caps, so a day of ~1,000 events stays
    small. Events past the cap sit on the top edge and are counted."""
    xs = sorted(e["s"] for e in evs)
    p98 = xs[min(len(xs) - 1, int(0.98 * len(xs)))]
    top = nice_max(max(p98, 1))
    w = WIDTH - LEFT
    paths = {"empty": [], "car": []}
    for e in evs:
        x = round(LEFT + e["min"] / (HOURS * 60) * w, 1)
        paths["car" if e["car"] else "empty"].append(f"M{x},{round(_y(e['s'], top), 1)}h0")
    above = [e["s"] for e in evs if e["s"] > top]
    return {**frame, "paths": [{"key": k, "d": "".join(v)} for k, v in paths.items() if v],
            "grid": _grid(top, lambda v: f"{v:g} s"),
            "above": {"n": len(above), "max": _secs(max(above))} if above else None}
