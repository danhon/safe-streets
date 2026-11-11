"""The main page's daily speed histogram: counts per bin, and where to draw them.

Bins are BIN_MPH wide with an edge exactly at the limit, so no bar straddles
it. At 2.5 mph the edges also fall on the row colour bands (limit +5, +10,
+15), so each bar over the limit takes one band's colour. Bins are closed on
the right, like `band`: a car at exactly the limit is "at or under".
"""

import math

from .speed import band

BIN_MPH = 2.5
WIDTH, TOP, PLOT_H = 320, 20, 100  # SVG units: label room above, bar area
AXIS_EVERY_MPH = 10


def _bin_index(mph: float, lo: float) -> int:
    # Right-closed: (lo + k·w, lo + (k+1)·w]. Rounding first keeps 20.0 in (17.5, 20].
    return max(0, math.ceil(round((mph - lo) / BIN_MPH, 9)) - 1)


def histogram(speeds: list[tuple[float | None, bool]], limit: float) -> dict:
    """speeds: (mph or None, rough) per pass. Rough estimates count in the
    totals and are kept apart only to be drawn lighter (2026-11-02)."""
    known = [(m, r) for m, r in speeds if m is not None]
    out = {"bins": [], "under": 0, "over": 0, "no_speed": len(speeds) - len(known), "limit": limit,
           "tallest": 0, "chart": None}
    if not known:
        return out
    lo = limit - math.ceil(limit / BIN_MPH) * BIN_MPH  # the bin edge at or below 0
    top = max(max(m for m, _ in known), limit + BIN_MPH)
    n = _bin_index(top, lo) + 1
    bins = [{"lo": round(lo + i * BIN_MPH, 6), "hi": round(lo + (i + 1) * BIN_MPH, 6), "solid": 0, "rough": 0}
            for i in range(n)]
    for mph, rough in known:
        bins[_bin_index(mph, lo)]["rough" if rough else "solid"] += 1
        out["over" if mph > limit else "under"] += 1
    for b in bins:
        b["band"] = band(b["hi"], limit)
    out["bins"] = bins
    out["tallest"] = max(b["solid"] + b["rough"] for b in bins)
    out["chart"] = _chart(bins, limit, lo, out["tallest"])
    return out


def _chart(bins: list[dict], limit: float, lo: float, tallest: int) -> dict:
    w = WIDTH / len(bins)
    scale = PLOT_H / tallest
    base = TOP + PLOT_H
    bars = []
    for i, b in enumerate(bins):
        rough_h = b["rough"] * scale
        solid_h = b["solid"] * scale
        bars.append({**b, "x": i * w, "w": w, "rough_y": base - rough_h, "rough_h": rough_h,
                     "solid_y": base - rough_h - solid_h, "solid_h": solid_h})
    ticks = []
    for i in range(len(bins) + 1):
        v = lo + i * BIN_MPH
        if v >= 0 and abs(v / AXIS_EVERY_MPH - round(v / AXIS_EVERY_MPH)) < 1e-6:
            ticks.append({"x": i * w, "label": f"{v:.0f}"})
    return {"width": WIDTH, "height": base + 16, "base": base, "top": TOP, "bars": bars, "ticks": ticks,
            "limit_x": round((limit - lo) / BIN_MPH) * w}
