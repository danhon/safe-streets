"""The tracker's facts for one pass, kept as passes.track_json."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Track:
    """cls, conf and track_id are the detector's for the frame kept (None at
    night); vote is the class across the track's frames. Image px."""
    box: tuple[float, float, float, float]
    trail: list[tuple[float, float, float]]  # x, y (box bottom-centre), t (pts, s)
    track_id: int | None
    cls: str
    conf: float | None
    vote: str
    mph: float | None
    n: int
    fps: float | None
    seconds: float
    metres: float | None
    length_m: float | None
    direction: int
    road: list[tuple[float, float]]
    crop: tuple[int, int, int, int] | None
    limit: float

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Track":
        d = dict(d)
        d["box"] = tuple(d["box"])
        d["trail"] = [tuple(p) for p in d["trail"]]
        d["road"] = [tuple(p) for p in d["road"]]
        d["crop"] = tuple(d["crop"]) if d["crop"] is not None else None
        return cls(**d)
