"""The operator's pages: the passes table with shots and clips, the log,
calibration, health and the settings (read only). Everything here is behind
`require` (a session) and every POST behind `csrf`."""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..calibrate import SNAP_RE, MAX_REPROJ_M, scale_from_drives
from ..config import UNIT_MPS
from ..geometry import WORK, fit_homography
from ..speed import Sample, confidence
from ..track import Track
from . import page
from .auth import Session

LOG_LINES = 400
SECRET_WORDS = ("PASSWORD", "SECRET", "TOKEN")
TRUSTED_BAND_PX = 30  # the trusted region from drives: their wheel line ± this
PAGE_PASSES = 500


class SceneIn(BaseModel):
    snapshot: str | None = None
    road: list[list[float]]
    lane: list[list[float]] | None = None
    mask: list[list[list[float]]] = []
    night_lane: list[float] | None = None
    occluders: list[float] = []


class DriveIn(BaseModel):
    id: int
    speed: float  # true speed, in UNITS


class DrivesIn(BaseModel):
    drives: list[DriveIn]


class PointIn(BaseModel):
    x: float
    y: float
    X: float
    Y: float


class PointsIn(BaseModel):
    snapshot: str
    points: list[PointIn]


def _inside(base: Path, p: str | None) -> Path | None:
    """p as a file inside base, or None (no escaping the data folder)."""
    if not p:
        return None
    try:
        f = Path(p).resolve()
        f.relative_to(base.resolve())
    except (ValueError, OSError):
        return None
    return f if f.is_file() else None


def _scene_of(row) -> dict:
    keys = row.keys()

    def opt(k):
        return json.loads(row[k]) if k in keys and row[k] else None

    return {"id": row["id"], "status": row["status"], "created_at": row["created_at"],
            "road": json.loads(row["road"]), "lane": opt("lane"), "trusted": opt("trusted"),
            "night_lane": opt("night_lane"), "occluders": opt("occluders") or [], "mask": opt("mask") or [],
            "size": opt("size") or list(WORK), "H": json.loads(row["H"]),
            "image_pts": json.loads(row["image_pts"]), "world_pts": json.loads(row["world_pts"]),
            "reference_path": row["reference_path"]}


def add_views(app, settings, db, batcher, source, vision, *, now, require, csrf) -> None:
    tz = ZoneInfo(settings.tz or "UTC")
    cal_dir = Path(settings.data_dir) / "calibration"
    unit = settings.unit_label
    per_unit = UNIT_MPS.get(settings.units, UNIT_MPS["mph"])

    def shown(mps):
        v = settings.speed(mps)
        return None if v is None else round(v, 1)

    def pm_shown(pm_mph):
        return None if pm_mph is None else round(pm_mph * UNIT_MPS["mph"] / per_unit, 1)

    # -- passes ----------------------------------------------------------

    @app.get("/")
    async def passes(day: str | None = None, s: Session = Depends(require)):
        today = now().astimezone(tz).date()
        try:
            d = date.fromisoformat(day) if day else today
        except ValueError:
            raise HTTPException(404, "Not a day")
        start = datetime.combine(d, datetime.min.time(), tz).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        rows = await db.passes_list("p.ts >= ? AND p.ts < ?", (start.isoformat(), end.isoformat()),
                                    limit=PAGE_PASSES)
        items = []
        for r in rows:
            flags = [f for f in (r["flags"] or "").split(",") if f]
            items.append({
                "id": r["id"], "time": datetime.fromisoformat(r["ts"]).astimezone(tz).strftime("%H:%M:%S"),
                "speed": shown(r["mps"]), "pm": pm_shown(r["plus_minus"]), "flags": ", ".join(flags),
                "confidence": confidence(flags, r["plus_minus"])[1], "cls": r["body"] or r["cls"],
                "direction": "→" if r["direction"] >= 0 else "←", "hidden": bool(r["hidden"]),
                "shot": bool(r["shot_path"]), "clip": bool(r["clip_path"])})
        return page("passes.html", s=s, settings=settings, unit=unit, items=items, day=d,
                    prev=(d - timedelta(days=1)).isoformat(), next=(d + timedelta(days=1)).isoformat()
                    if d < today else None, health=await _health())

    @app.get("/shot/{pid}.jpg")
    async def shot(pid: int, _s: Session = Depends(require)):
        row = await db.get_pass(pid)
        f = _inside(settings.data_dir, row["shot_path"] if row else None)
        if f is None:
            raise HTTPException(404, "No picture")
        return FileResponse(f, media_type="image/jpeg")

    @app.get("/clip/{pid}.mp4")
    async def clip(pid: int, _s: Session = Depends(require)):
        row = await db.get_pass(pid)
        ev = await db.get_event(row["event_id"]) if row else None
        f = _inside(settings.data_dir, ev["clip_path"] if ev else None)
        if f is None:
            raise HTTPException(404, "No clip (clips are kept for CLIP_RETENTION_DAYS)")
        return FileResponse(f, media_type="video/mp4")

    # -- the log, settings, health ---------------------------------------

    @app.get("/logs")
    async def logs(q: str = "", s: Session = Depends(require)):
        lines, where = [], None
        if settings.log_dir:
            where = Path(settings.log_dir) / "app.log"
            if where.is_file():
                with where.open(encoding="utf-8", errors="replace") as f:
                    lines = list(deque((ln.rstrip("\n") for ln in f if q.lower() in ln.lower()), maxlen=LOG_LINES))
        return page("logs.html", s=s, lines=lines, where=where, q=q, n=LOG_LINES)

    @app.get("/settings")
    async def settings_page(s: Session = Depends(require)):
        rows = []
        for f in dataclasses.fields(settings):
            v = getattr(settings, f.name)
            secret = any(w in f.name.upper() for w in SECRET_WORDS)
            rows.append({"name": f.name.upper(), "value": ("(set)" if v else "(unset)") if secret else v})
        return page("settings.html", s=s, rows=rows)

    async def _health() -> dict:
        row = await db.active_calibration()
        out = {"calibration": f"{row['status']} (#{row['id']})" if row else "none: nothing is analysed yet",
               "source": None, "batch": None}
        if source is not None:
            out["source"] = "connected" if source.connected else f"not connected: {source.last_error or 'not tried yet'}"
        if batcher is not None:
            out["batch"] = {"age_s": round(batcher.age()), "last": batcher.last_run}
        return out

    @app.get("/health")
    async def health(s: Session = Depends(require)):
        return page("health.html", s=s, h=await _health())

    # -- calibration -----------------------------------------------------

    def snap_path(name: str | None) -> Path | None:
        if not name or not SNAP_RE.match(name) or not (cal_dir / name).is_file():
            return None
        return cal_dir / name

    @app.get("/calibrate")
    async def calibrate(s: Session = Depends(require)):
        cal_dir.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            if source is not None and (source.connected or await source.ensure_connected()):
                name = f"snap-{int(time.time())}.jpg"
                (cal_dir / name).write_bytes(await source.snapshot())
        except Exception:  # noqa: BLE001  (fall back to the newest saved snapshot)
            name = None
        if name is None:
            snaps = sorted(cal_dir.glob("snap-*.jpg"))
            name = snaps[-1].name if snaps else None
        row = await db.active_calibration()
        rows = await db.passes_list("p.track_json IS NOT NULL AND instr(p.flags, 'headlights') = 0", (), limit=60)
        drives = [{"id": r["id"], "time": datetime.fromisoformat(r["ts"]).astimezone(tz).strftime("%a %H:%M:%S"),
                   "speed": shown(r["mps"]), "direction": "→" if r["direction"] >= 0 else "←", "n": r["n"],
                   "shot": bool(r["shot_path"])} for r in rows]
        lengths = await db.recent_lengths(20)
        return page("calibrate.html", s=s, snapshot=name, scene=json.dumps(_scene_of(row) if row else None),
                    cal_status=row["status"] if row else None, drives=drives, unit=unit,
                    median_length=round(statistics.median(lengths), 2) if lengths else None, n_lengths=len(lengths))

    @app.get("/calibrate/img/{name}")
    async def snapshot_img(name: str, _s: Session = Depends(require)):
        p = snap_path(name)
        if p is None:
            raise HTTPException(404, "No snapshot")
        return FileResponse(p, media_type="image/jpeg")

    @app.post("/calibrate/scene")
    async def save_scene(body: SceneIn, _s: Session = Depends(csrf)):
        """The scene: road, lane, masks, the night lane line, occluders. On the
        current calibration's scale, or as a draft (no scale yet) to collect
        drive-bys under."""
        if len(body.road) < 3:
            raise HTTPException(400, "Draw the road outline (3 corners or more).")
        row = await db.active_calibration()
        cur = _scene_of(row) if row and row["status"] != "stale" else None
        ref = snap_path(body.snapshot)
        scaled = cur is not None and cur["status"] != "draft"
        cid = await db.save_calibration(
            image_pts=cur["image_pts"] if scaled else [], world_pts=cur["world_pts"] if scaled else [],
            H=cur["H"] if scaled else [[1, 0, 0], [0, 1, 0], [0, 0, 1]], road=body.road,
            reference_path=str(ref) if ref else (cur or {}).get("reference_path", ""),
            status=cur["status"] if scaled else "draft", lane=body.lane or None,
            trusted=cur["trusted"] if scaled else None,
            night_lane=body.night_lane if body.night_lane and len(body.night_lane) == 2 else None,
            occluders=body.occluders, mask=body.mask, size=list(WORK))
        return {"id": cid, "status": "active" if scaled else "draft"}

    @app.post("/calibrate/drives")
    async def save_drives(body: DrivesIn, _s: Session = Depends(csrf)):
        """The scale from two drive-bys or more at known speeds."""
        row = await db.active_calibration()
        if row is None:
            raise HTTPException(400, "Draw the road and save it first.")
        cur = _scene_of(row)
        drives, ys = [], []
        for d in body.drives:
            p = await db.get_pass(d.id)
            if p is None or not p["track_json"] or d.speed <= 0:
                raise HTTPException(400, f"Pass {d.id} has no track to calibrate from.")
            trail = Track.from_json(json.loads(p["track_json"])).trail
            samples = [Sample(t=t, x=x, y=y, box=(x, y, x, y)) for x, y, t in trail]
            if len(samples) < 3:
                raise HTTPException(400, f"Pass {d.id}'s track is too short.")
            drives.append((samples, d.speed * per_unit))
            ys += [s.y for s in samples]
        try:
            fit = scale_from_drives(drives)
        except ValueError:
            raise HTTPException(400, "A calibration needs two drive-bys or more.")
        y = statistics.median(ys)
        trusted = [[0, y - TRUSTED_BAND_PX], [WORK[0], y - TRUSTED_BAND_PX], [WORK[0], y + TRUSTED_BAND_PX],
                   [0, y + TRUSTED_BAND_PX]]
        cid = await db.save_calibration(
            image_pts=[], world_pts=[], H=fit.H.tolist(), road=cur["road"], reference_path=cur["reference_path"],
            status="active", lane=cur["lane"], trusted=trusted, night_lane=cur["night_lane"],
            occluders=cur["occluders"], mask=cur["mask"], size=cur["size"])
        to_unit = UNIT_MPS["mph"] / per_unit
        return {"id": cid, "px_per_m": round(fit.px_per_m, 3),
                "readings": [round(r * to_unit, 1) for r in fit.readings_mph],
                "spread": round(fit.spread_mph * to_unit, 2), "unit": unit}

    @app.post("/calibrate/points")
    async def save_points(body: PointsIn, _s: Session = Depends(csrf)):
        """Experimental, untested: a homography from measured ground points."""
        row = await db.active_calibration()
        if row is None:
            raise HTTPException(400, "Draw the road and save it first.")
        cur = _scene_of(row)
        image = [(p.x, p.y) for p in body.points]
        world = [(p.X, p.Y) for p in body.points]
        try:
            H, err = fit_homography(image, world)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if err > MAX_REPROJ_M:
            raise HTTPException(400, f"Re-projection error {err:.2f} m is over {MAX_REPROJ_M} m: "
                                     "check the points and their distances.")
        ref = snap_path(body.snapshot)
        cid = await db.save_calibration(
            image_pts=image, world_pts=world, H=H.tolist(), road=cur["road"],
            reference_path=str(ref) if ref else cur["reference_path"], status="active", lane=cur["lane"],
            trusted=cur["trusted"], night_lane=cur["night_lane"], occluders=cur["occluders"], mask=cur["mask"],
            size=cur["size"])
        return {"id": cid, "reproj_err_m": err, "H": H.tolist()}
