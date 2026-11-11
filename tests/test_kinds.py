import sqlite3

import pytest

from safe_streets import kinds

ROWS = [  # (cls, body, flags) -> kind
    (("car", None, ""), None), (("car", "SUV", ""), None), (("car", None, "headlights"), None),
    (("truck", "pickup", ""), None),
    (("bicycle", None, ""), "bike"), (("motorcycle", None, ""), "two-wheeler"),
    (("motorcycle", "bike", ""), "bike"), (("bicycle", "e-bike", ""), "e-bike"),
    (("motorcycle", "scooter", "short"), "scooter"), (("motorcycle", "motorcycle", ""), "motorcycle"),
    (("bicycle", None, None), "bike"),
]


def row(cls, body, flags):
    return {"cls": cls, "body": body, "flags": flags}


@pytest.mark.parametrize("cols, want", ROWS)
def test_kind(cols, want):
    assert kinds.kind(row(*cols)) == want
    assert kinds.is_bike(row(*cols)) == (want == "bike")


def test_kind_without_a_body_column():
    assert kinds.kind({"cls": "bicycle", "flags": ""}) == "bike"
    assert kinds.kind({"cls": "motorcycle", "flags": ""}) == "two-wheeler"


def test_post_labels():
    assert kinds.POST_LABELS == {"motorcycle": "Motorcycle", "scooter": "Scooter", "e-bike": "E-bike",
                                 "bike": "Bike", "two-wheeler": "Motorcycle or e-bike"}


@pytest.fixture
def table():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE passes (cls TEXT, body TEXT, flags TEXT)")
    c.executemany("INSERT INTO passes VALUES (?, ?, ?)", [cols for cols, _ in ROWS])
    return c


def test_bike_sql_matches_python(table):
    bikes = {r for (r,) in table.execute(f"SELECT rowid FROM passes WHERE {kinds.BIKE_SQL}")}
    cars = {r for (r,) in table.execute(f"SELECT rowid FROM passes p WHERE {kinds.CAR_SQL}")}
    want = {i + 1 for i, (cols, _) in enumerate(ROWS) if kinds.is_bike(row(*cols))}
    assert bikes == want
    assert cars == set(range(1, len(ROWS) + 1)) - want


@pytest.mark.parametrize("body", [False, True])
def test_type_sql_matches_type_of(table, body):
    got = [t for (t,) in table.execute(f"SELECT {kinds.type_sql(body)} FROM passes ORDER BY rowid")]
    assert got == [kinds.type_of(row(*cols), body) for cols, _ in ROWS]


def test_type_of():
    assert kinds.type_of(row("car", "SUV", ""), True) == "SUV"
    assert kinds.type_of(row("car", "SUV", ""), False) == "car"
    assert kinds.type_of(row("bicycle", None, ""), False) == "bike"
    assert kinds.type_of(row("motorcycle", "e-bike", ""), True) == "e-bike"
    assert kinds.type_of(row("motorcycle", None, ""), True) == "two-wheeler"
