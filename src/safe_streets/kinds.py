"""A pass's two-wheeler kind: the one place, in Python and in SQL, that says
what a daytime two-wheeler was, and so whether it's a bike (not a car) or a powered vehicle (a car).

YOLO says `motorcycle` or `bicycle`; the sidecar's kind, once
TWO_WHEELER_TYPE is on, is in `passes.body`. Without one (off, shadow,
unsure, the sidecar down) YOLO's class decides: `bicycle` is a bike,
`motorcycle` an unsure two-wheeler. Night passes are never two-wheelers:
one light can't tell (they're `car`, a "Vehicle").

A bike is every pedal or cargo bike, electric or not, and any bike with no
visible motor; an e-bike is visibly electric and not a cargo bike
(2026-11-09). E-bikes are powered: cars.
"""

TWO_WHEEL_CLS = frozenset({"motorcycle", "bicycle"})
KINDS = ("motorcycle", "scooter", "e-bike", "bike")
POST_LABELS = {"motorcycle": "Motorcycle", "scooter": "Scooter", "e-bike": "E-bike", "bike": "Bike",
               "two-wheeler": "Motorcycle or e-bike"}


def _flags(row) -> list[str]:
    return [f for f in (row["flags"] or "").split(",") if f]


def kind(row) -> str | None:
    """motorcycle, scooter, e-bike, bike, two-wheeler (unsure), or None for
    anything else (a four-wheeler, a night pass)."""
    if row["cls"] not in TWO_WHEEL_CLS or "headlights" in _flags(row):
        return None
    body = row["body"] if "body" in row.keys() else None
    if body in KINDS:
        return body
    return "bike" if row["cls"] == "bicycle" else "two-wheeler"


def is_bike(row) -> bool:
    return kind(row) == "bike"


_IN_CLS = "cls IN ('motorcycle', 'bicycle')"
_KINDS_SQL = ", ".join(f"'{k}'" for k in KINDS)
_DAY = "instr(COALESCE(flags, ''), 'headlights') = 0"
_KIND_SQL = (f"COALESCE(CASE WHEN body IN ({_KINDS_SQL}) THEN body END, "
             "CASE WHEN cls = 'bicycle' THEN 'bike' ELSE 'two-wheeler' END)")
# Never NULL (IS, not =), so NOT of it never drops a row.
BIKE_SQL = f"({_IN_CLS} AND {_DAY} AND {_KIND_SQL} IS 'bike')"
CAR_SQL = f"(NOT {BIKE_SQL})"


def type_sql(body: bool) -> str:
    """The table's Type value in SQL: the kind for a daytime two-wheeler,
    else the sidecar's body where it answered (with `body`) or YOLO's class."""
    other = "COALESCE(body, cls)" if body else "cls"
    return f"(CASE WHEN {_IN_CLS} AND {_DAY} THEN {_KIND_SQL} ELSE {other} END)"


def type_of(row, body: bool) -> str:
    """type_sql, in Python."""
    if k := kind(row):
        return k
    return (row["body"] or row["cls"]) if body and "body" in row.keys() else row["cls"]
