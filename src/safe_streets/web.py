"""The web app: the public views at `/`, and the operator's dashboard at
`/operator` (operator/, its own sub-app behind a password).

Public routes take no Request: they can't read cookies, Authorization or
Remote-User, so the same URL gives every visitor the same bytes, and any
cache in front of them is safe. Their data comes only from public_view.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.exceptions import HTTPException as StarletteHTTPException

from .charts import hour_label, pct
from .public_view import PublicDay, latest_day, public_day

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
STATIC = HERE / "static"
TEMPLATES = HERE / "templates"
PUBLIC_HEADERS = {"Cache-Control": "public, max-age=60", "X-Robots-Tag": "noindex"}
ERROR_HEADERS = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex"}

_env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))

# Chart frame, in SVG units.
W, H, LEFT, TOP, PLOT_H = 320, 130, 22, 8, 106


def _error_page(status: int) -> HTMLResponse:
    words = {404: "Not found", 500: "Something went wrong"}.get(status, "Error")
    body = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"robots\" content=\"noindex\">"
            f"<title>{words}</title></head><body><main><h1>{words}</h1><p><a href=\"/\">The day's traffic</a></p>"
            f"</main></body></html>")
    return HTMLResponse(body, status_code=status, headers=ERROR_HEADERS)


def _nice(v: float) -> float:
    if v <= 0:
        return 1
    mag = 10 ** math.floor(math.log10(v))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= v)


def _grid(top: float, scale: float) -> list[dict]:
    step = top / 2
    return [{"v": f"{step * i:g}", "y": round(TOP + PLOT_H - step * i * scale, 2)} for i in range(1, 3)]


def _charts(d: PublicDay, k: int) -> tuple[dict, dict]:
    slot = (W - LEFT) / 24
    ticks = [{"x": round(LEFT + (h + .5) * slot, 2), "label": label}
             for h, label in ((0, "12a"), (6, "6a"), (12, "12p"), (18, "6p"))]
    top = _nice(max([h.n for h in d.hours] + [1]))
    scale = PLOT_H / top
    cols = []
    for h in d.hours:
        if not h.n:
            continue
        x, w = round(LEFT + h.hour * slot + 1, 2), round(slot - 2, 2)
        hgt = h.n * scale
        few = h.over is None
        cols.append({"x": x, "w": w, "y": round(TOP + PLOT_H - hgt, 2), "h": round(hgt, 2), "few": few,
                     "over_h": 0 if few else round(h.over * scale, 2),
                     "title": (f"{hour_label(h.hour)}: {h.n} vehicle{'s' if h.n != 1 else ''}"
                               + ("" if few else f", {h.over} over {d.limit:g} {d.unit}"))})
    busiest = max(d.hours, key=lambda h: h.n)
    chart = {"w": W, "h": H, "left": LEFT, "cols": cols, "ticks": ticks, "grid": _grid(top, scale),
             "label": (f"Vehicles by hour: {d.total} in all, the most {busiest.n} at {hour_label(busiest.hour)}"
                       if d.total else "No vehicles")}
    known = [h for h in d.hours if h.fastest is not None]
    ftop = 10 * math.ceil(max([h.fastest for h in known] + [d.limit]) / 10) or 10
    fscale = PLOT_H / ftop
    fcols = [{"x": round(LEFT + h.hour * slot + 1, 2), "w": round(slot - 2, 2),
              "y": round(TOP + PLOT_H - h.fastest * fscale, 2), "h": round(h.fastest * fscale, 2),
              "over": h.fastest > d.limit,
              "title": f"{hour_label(h.hour)}: fastest about {h.fastest} {d.unit}"} for h in known]
    fast = {"cols": fcols, "grid": _grid(ftop, fscale), "limit_y": round(TOP + PLOT_H - d.limit * fscale, 2),
            "label": (f"The fastest vehicle each hour, of hours with {k} or more vehicles; the limit is "
                      f"{d.limit:g} {d.unit}")}
    return chart, fast


def _day_label(day: date) -> str:
    return day.strftime("%A, %B ") + str(day.day)


def _delay_label(hours: float) -> str:
    if hours % 24 == 0 and hours >= 24:
        n = int(hours // 24)
        return f"{n} day{'s' if n != 1 else ''}"
    return f"{hours:g} hours"


def _clock(d: datetime) -> str:
    h = d.hour % 12 or 12
    return f"{h}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'}" if d.minute else f"{h} {'am' if d.hour < 12 else 'pm'}"


def create_app(settings, db, batcher, source, vision, *,
               now: Callable[[], datetime] | None = None) -> FastAPI:
    now = now or (lambda: datetime.now(timezone.utc))
    @asynccontextmanager
    async def lifespan(_app):
        # The batches run in the app's process, from its start to its stop.
        if batcher is not None:
            batcher.start()
        yield
        if batcher is not None:
            await batcher.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request, exc):  # the request is never read
        return _error_page(404 if exc.status_code == 404 else exc.status_code)

    @app.exception_handler(Exception)
    async def server_error(_request, exc):
        log.error("public page failed: %s", type(exc).__name__)
        return _error_page(500)

    def which_day(day: str | None) -> date:
        latest = latest_day(settings, now())
        if day is None:
            return latest
        try:
            d = date.fromisoformat(day)
        except ValueError:
            raise StarletteHTTPException(404)
        if d > latest:
            raise StarletteHTTPException(404)
        return d

    async def get_day(day: str | None) -> PublicDay:
        return await public_day(db, which_day(day), settings, now())

    @app.get("/", response_class=HTMLResponse)
    async def index(day: str | None = None):
        d = await get_day(day)
        latest = latest_day(settings, now())
        prev, nxt = d.day - timedelta(days=1), d.day + timedelta(days=1)
        chart, fast = _charts(d, settings.public_min_group)
        fastest = max((h.fastest for h in d.hours if h.fastest is not None), default=None)
        html = _env.get_template("public/day.html").render(
            title="Traffic on the street", d=d, chart=chart, fast=fast, fastest=fastest,
            k=settings.public_min_group, day_label=_day_label(d.day), limit_label=f"{d.limit:g}",
            over_pct=pct(d.over, len(d.passes)) if d.passes else "–",
            until_label=_clock(d.published_until) if d.published_until else "",
            delay_label=_delay_label(settings.public_delay_hours),
            prev=prev.isoformat(), prev_label=prev.strftime("%a"),
            next=nxt.isoformat() if nxt <= latest else None, next_label=nxt.strftime("%a"))
        return HTMLResponse(html, headers=PUBLIC_HEADERS)

    def _proto(name: str, d: PublicDay) -> HTMLResponse:
        data = json.dumps(d.as_json(), separators=(",", ":")).replace("</", "<\\/")
        page = (TEMPLATES / "public" / name).read_text()
        script = f'<script>const DATA = {data};</script><script src="/static/common.js"></script>'
        return HTMLResponse(page.replace("<!--COMMON-->", script), headers=PUBLIC_HEADERS)

    @app.get("/zoo", response_class=HTMLResponse)
    async def zoo(day: str | None = None):
        return _proto("zoo.html", await get_day(day))

    @app.get("/dial", response_class=HTMLResponse)
    async def dial(day: str | None = None):
        return _proto("dial.html", await get_day(day))

    @app.get("/day.json")
    async def day_json(day: str | None = None):
        return JSONResponse((await get_day(day)).as_json(), headers=PUBLIC_HEADERS)

    @app.get("/healthz")
    async def healthz():
        """Always 200 while the app answers; batch_stale says the batcher stalled."""
        out = {"ok": True}
        if batcher is not None:
            age = batcher.age()
            out |= {"batch_age_s": round(age), "batch_stale": age > settings.batch_stale_seconds}
        return JSONResponse(out, headers=PUBLIC_HEADERS)

    class _Static(StaticFiles):
        async def get_response(self, path, scope):
            r = await super().get_response(path, scope)
            if r.status_code == 200:
                r.headers.update(PUBLIC_HEADERS)
            return r

    app.mount("/static", _Static(directory=STATIC), name="static")
    from .operator import create_operator_app
    app.mount("/operator", create_operator_app(settings, db, batcher, source, vision, now=now))
    return app
