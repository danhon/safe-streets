"""The playful public pages' scripts run on the days a new install has: none
at all, or only a few passes, counted but not listed. Each page's inline
scripts run in node against a small stand-in for the DOM."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import pytest

from test_public import DAY, TZ, app_for, client, seeded  # noqa: F401 (a fixture)

STATIC = Path(__file__).resolve().parent.parent / "src" / "safe_streets" / "static"

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="needs node")

DOM = r"""
const els = {};
const el = id => els[id] || (els[id] = {id, innerHTML: "", textContent: "", style: {}, dataset: {},
  classList: {add() {}, remove() {}, toggle() {}, contains() { return false; }},
  insertAdjacentHTML(_, h) { this.innerHTML += h; }, addEventListener() {}, appendChild() {},
  querySelector() { return el("_q"); }, querySelectorAll() { return []; }, setAttribute() {},
  getBoundingClientRect() { return {width: 800, height: 600, left: 0, top: 0}; },
  cloneNode() { return el("_clone" + Math.random()); }, removeAttribute() {}, remove() {}, append() {},
  offsetWidth: 800, offsetHeight: 600, scrollWidth: 800, clientWidth: 800,
  get parentNode() { return el("_parent"); }, children: [], removeChild() {}, insertBefore() {}});
globalThis.document = {getElementById: el, querySelector: () => el("_q"), querySelectorAll: () => [],
  createElement: () => el("_new" + Math.random()), createElementNS: () => el("_ns" + Math.random()), head: el("head"), body: el("body"),
  addEventListener() {}, documentElement: el("html")};
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.location = {search: ""};
globalThis.localStorage = {getItem: () => null, setItem() {}};
globalThis.requestAnimationFrame = () => 0;
globalThis.matchMedia = () => ({matches: false, addEventListener() {}});
globalThis.ResizeObserver = class { observe() {} };
globalThis.setTimeout = () => 0; globalThis.setInterval = () => 0;
"""
REPORT = "\nconsole.log(JSON.stringify(Object.fromEntries(Object.entries(els).map(([k, v]) => [k, v.innerHTML || v.textContent]))));"


def run_page(html: str, tmp_path) -> dict:
    """Every script in page order: inline ones, and /static/ ones read from disk."""
    scripts = []
    for attrs, body in re.findall(r"<script([^>]*)>(.*?)</script>", html, re.S):
        src = re.search(r'src="/static/([^"]+)"', attrs)
        scripts.append((STATIC / src[1]).read_text() if src else body)
    js = tmp_path / "page.js"
    js.write_text(DOM + "\n".join(scripts) + REPORT)
    r = subprocess.run(["node", str(js)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


async def four_cars(db):
    """Four cars in one quarter hour: counted, never listed."""
    for i in range(4):
        await db.add_pass(event_id=f"f{i}", ts=datetime(2026, 11, 3, 10, 1 + i, tzinfo=TZ).astimezone(timezone.utc).isoformat(),
                          cls="car", direction=1, mph=24.0, plus_minus=1.0)


@pytest.mark.parametrize("page", ["zoo", "dial"])
@pytest.mark.parametrize("cars", [0, 4])
async def test_a_day_with_nothing_listed_draws_without_errors(db, settings, tmp_path, page, cars):
    if cars:
        await four_cars(db)
    async with client(app_for(settings, db)) as c:
        r = await c.get(f"/{page}?day={DAY}")
    assert r.status_code == 200
    out = run_page(r.text, tmp_path)
    if page == "zoo":
        assert "mostly" not in out["quip"] and "parked" not in out["quip"]
        assert (f"{cars} " if cars else "No ") in out["quip"]


@pytest.mark.parametrize("page", ["zoo", "dial"])
async def test_a_full_day_still_draws(seeded, settings, tmp_path, page):
    async with client(app_for(settings, seeded)) as c:
        r = await c.get(f"/{page}?day={DAY}")
    out = run_page(r.text, tmp_path)
    if page == "zoo":
        assert "mostly" in out["quip"] and "Wildest hour" in out["awards"]
