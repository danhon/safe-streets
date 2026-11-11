"""The operator's dashboard sits behind a built-in password (AUTH_MODE=password,
the default): a one-time setup link, argon2id, a signed session cookie,
lockout, CSRF on every POST, and no sign-in over plain HTTP from the
internet. AUTH_MODE=none serves it to localhost only."""

import dataclasses
import re
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from safe_streets.operator import auth
from safe_streets.web import create_app

T0 = datetime(2026, 11, 10, 20, 0, tzinfo=timezone.utc)
LAN = ("192.168.1.20", 50000)  # prepublish: allow
PUBLIC_IP = ("203.0.113.5", 50000)
PASSWORD = "correct horse battery"


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


def client(app, *, peer=LAN, https=True):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=peer, raise_app_exceptions=False),
                             base_url=("https" if https else "http") + "://cam.example")


def app_for(settings, db, clock, **kw):
    return create_app(dataclasses.replace(settings, **kw), db, None, None, None, now=clock)


def csrf_of(html: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


async def sign_in(c, password=PASSWORD):
    form = await c.get("/operator/login")
    return await c.post("/operator/login", data={"password": password, "csrf": csrf_of(form.text)})


async def test_the_setup_link_works_once(settings, db, clock):
    token = await auth.new_setup_token(db, clock())
    async with client(app_for(settings, db, clock)) as c:
        page = await c.get(f"/operator/setup?token={token}")
        assert page.status_code == 200
        r = await c.post("/operator/setup", data={"token": token, "password": PASSWORD, "password2": PASSWORD,
                                                  "csrf": csrf_of(page.text)})
        assert r.status_code == 303 and r.headers["location"] == "/operator/"
        assert (await c.get("/operator/")).status_code == 200  # signed in by setting it
    async with client(app_for(settings, db, clock)) as c:
        assert (await c.get(f"/operator/setup?token={token}")).status_code == 403
    assert (await db.get_operator())["password_hash"].startswith("$argon2id$")


async def test_the_setup_link_expires_after_24_hours(settings, db, clock):
    token = await auth.new_setup_token(db, clock())
    clock.t = T0 + timedelta(hours=24, minutes=1)
    async with client(app_for(settings, db, clock)) as c:
        assert (await c.get(f"/operator/setup?token={token}")).status_code == 403


async def test_a_short_password_is_refused(settings, db, clock):
    token = await auth.new_setup_token(db, clock())
    async with client(app_for(settings, db, clock)) as c:
        page = await c.get(f"/operator/setup?token={token}")
        r = await c.post("/operator/setup", data={"token": token, "password": "elevenchars", "password2": "elevenchars",
                                                  "csrf": csrf_of(page.text)})
    assert r.status_code == 400 and "12" in r.text
    assert (await db.get_operator() or {}).get("password_hash") is None


async def test_six_wrong_passwords_lock_sign_in(settings, db, clock):
    await auth.set_password(db, PASSWORD)
    async with client(app_for(settings, db, clock)) as c:
        for _ in range(5):
            assert (await sign_in(c, "not the password")).status_code == 401
        r = await sign_in(c)  # the right one, but locked
        assert r.status_code == 429 and "ss_session" not in r.headers.get("set-cookie", "")
        clock.t = T0 + timedelta(minutes=16)
        assert (await sign_in(c)).status_code == 303


async def test_reset_password_ends_sessions(settings, db, clock):
    """Review Focus 5: a lost password is reset from the command line, and
    every session from before it ends."""
    await auth.set_password(db, PASSWORD)
    async with client(app_for(settings, db, clock)) as c:
        assert (await sign_in(c)).status_code == 303
        assert (await c.get("/operator/")).status_code == 200
        token = await auth.reset_password(db, clock())
        r = await c.get("/operator/")
        assert r.status_code == 303 and r.headers["location"] == "/operator/login"
        assert (await sign_in(c)).status_code == 401  # the old password is gone too
        assert (await c.get(f"/operator/setup?token={token}")).status_code == 200


async def test_a_post_without_the_csrf_token_is_403(settings, db, clock):
    await auth.set_password(db, PASSWORD)
    async with client(app_for(settings, db, clock)) as c:
        assert (await sign_in(c)).status_code == 303
        assert (await c.post("/operator/logout")).status_code == 403
        assert (await c.post("/operator/logout", data={"csrf": "forged"})).status_code == 403
        assert (await c.post("/operator/login", data={"password": PASSWORD})).status_code == 403
        assert (await c.get("/operator/")).status_code == 200  # still signed in


async def test_sign_out(settings, db, clock):
    await auth.set_password(db, PASSWORD)
    async with client(app_for(settings, db, clock)) as c:
        await sign_in(c)
        page = await c.get("/operator/")
        assert (await c.post("/operator/logout", data={"csrf": csrf_of(page.text)})).status_code == 303
        assert (await c.get("/operator/")).status_code == 303


@pytest.mark.parametrize("peer,https,ok", [(PUBLIC_IP, False, False), (PUBLIC_IP, True, True), (LAN, False, True),
                                           (("127.0.0.1", 1), False, True)])
async def test_no_sign_in_over_plain_http_from_the_internet(settings, db, clock, peer, https, ok):
    await auth.set_password(db, PASSWORD)
    async with client(app_for(settings, db, clock), peer=peer, https=https) as c:
        r = await c.get("/operator/login")
        assert (r.status_code == 200) is ok
        if ok:
            r = await sign_in(c)
            assert r.status_code == 303
            cookie = r.headers["set-cookie"]
            assert "HttpOnly" in cookie and "SameSite=strict" in cookie.replace("Strict", "strict")
            assert "Path=/operator" in cookie
            assert ("Secure" in cookie) is https
        else:
            assert "HTTPS" in r.text


@pytest.mark.parametrize("peer,status", [(("127.0.0.1", 1), 200), (("::1", 1), 200), (LAN, 403), (PUBLIC_IP, 403)])
async def test_auth_mode_none_is_localhost_only(settings, db, clock, peer, status):
    async with client(app_for(settings, db, clock, auth_mode="none"), peer=peer, https=False) as c:
        assert (await c.get("/operator/")).status_code == status


async def test_every_operator_page_needs_a_session(settings, db, clock):
    await auth.set_password(db, PASSWORD)
    app = app_for(settings, db, clock)
    op = next(r.app for r in app.routes if getattr(r, "path", "") == "/operator")
    open_ = {"/login", "/setup", "/logout"}
    paths = [r.path for r in op.routes if "GET" in getattr(r, "methods", ()) and r.path not in open_]
    assert len(paths) >= 6
    async with client(app) as c:
        for p in paths:
            url = "/operator" + re.sub(r"\{[^}]+\}", "1", p)
            r = await c.get(url)
            assert r.status_code == 303 and r.headers["location"] == "/operator/login", url


async def test_shots_and_clips_only_under_operator(settings, db, clock):
    shots = settings.data_dir / "shots"
    shots.mkdir(parents=True)
    (shots / "1.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
    pid = await db.add_pass(event_id="e1", ts="2026-11-10T19:00:00+00:00", cls="car", direction=1, mph=24.0,
                            shot_path=str(shots / "1.jpg"))
    await auth.set_password(db, PASSWORD)
    async with client(app_for(settings, db, clock)) as c:
        for url in (f"/shot/{pid}.jpg", f"/clip/{pid}.mp4", f"/shots/1.jpg", f"/static/../shots/1.jpg"):
            assert (await c.get(url)).status_code == 404, url
        assert (await c.get(f"/operator/shot/{pid}.jpg")).status_code == 303
        await sign_in(c)
        r = await c.get(f"/operator/shot/{pid}.jpg")
        assert r.status_code == 200 and r.content.startswith(b"\xff\xd8")


async def test_operator_responses_are_never_stored_or_indexed(settings, db, clock):
    await auth.set_password(db, PASSWORD)
    async with client(app_for(settings, db, clock)) as c:
        for r in (await c.get("/operator/login"), await sign_in(c), await c.get("/operator/")):
            assert r.headers["cache-control"] == "no-store"
            assert r.headers["x-robots-tag"] == "noindex"


def test_an_unknown_auth_mode_refuses_to_start(settings):
    with pytest.raises(SystemExit, match="AUTH_MODE"):
        dataclasses.replace(settings, auth_mode="open", tz="UTC").require()


# -- the review's findings (2026-11-10) -------------------------------------

async def test_the_cookie_is_secure_whenever_base_url_is_https(settings, db, clock):
    """Behind a TLS proxy the app may see plain HTTP; BASE_URL says it isn't."""
    await auth.set_password(db, PASSWORD)
    async with client(app_for(settings, db, clock, base_url="https://cam.example"), https=False) as c:
        r = await sign_in(c)
        assert r.status_code == 303 and "Secure" in r.headers["set-cookie"]


async def test_auth_mode_none_refuses_a_proxied_request(settings, db, clock):
    """A host proxy that forwards to 127.0.0.1 would make every visitor local."""
    async with client(app_for(settings, db, clock, auth_mode="none"), peer=("127.0.0.1", 1), https=False) as c:
        for h in ({"X-Forwarded-For": "203.0.113.5"}, {"Forwarded": "for=203.0.113.5"}, {"X-Real-IP": "203.0.113.5"}):
            assert (await c.get("/operator/", headers=h)).status_code == 403, h


def test_auth_mode_none_with_a_base_url_refuses_to_start(settings):
    with pytest.raises(SystemExit, match="AUTH_MODE=none"):
        dataclasses.replace(settings, auth_mode="none", base_url="https://cam.example", tz="UTC").require()


def test_forwarded_headers_are_trusted_only_from_the_named_proxies(settings, monkeypatch):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "198.51.100.7")
    from safe_streets.config import Settings
    assert Settings().forwarded_allow_ips == "198.51.100.7"
    monkeypatch.delenv("FORWARDED_ALLOW_IPS")
    assert Settings().forwarded_allow_ips == "127.0.0.1"
