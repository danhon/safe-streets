"""The operator's dashboard: a sub-app mounted at /operator, with its own
error handling and headers (never cached, never indexed), behind the
built-in password (auth.py) or, with AUTH_MODE=none, loopback only."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import auth
from .auth import COOKIE, SESSION_DAYS, Session, Signer

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent.parent / "templates"
HEADERS = [(b"cache-control", b"no-store"), (b"x-robots-tag", b"noindex")]

env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))


class NeedSignIn(Exception):
    pass


class _Headers:
    """Every operator response: no-store, noindex (errors and redirects too)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_(msg):
            if msg["type"] == "http.response.start":
                drop = {k for k, _ in HEADERS}
                msg["headers"] = [(k, v) for k, v in msg.get("headers", []) if k.lower() not in drop] + HEADERS
            await send(msg)

        await self.app(scope, receive, send_)


def page(name: str, status: int = 200, **ctx) -> HTMLResponse:
    return HTMLResponse(env.get_template(f"operator/{name}").render(**ctx), status_code=status)


def _plain(status: int, words: str) -> HTMLResponse:
    r = page("message.html", status, title=words, message=words)
    r.headers.update({k.decode(): v.decode() for k, v in HEADERS})  # a 500 skips the middleware
    return r


def create_operator_app(settings, db, batcher, source, vision, *, now: Callable[[], datetime]) -> FastAPI:
    """The /operator sub-app."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    signer = Signer(auth.load_secret(settings.data_dir))
    none_csrf = secrets.token_urlsafe(24)  # AUTH_MODE=none: the process's token
    app.state.signer = signer
    gateway = auth.default_gateway()
    warned_gateway = []

    @app.exception_handler(NeedSignIn)
    async def need_sign_in(_request, _exc):
        return RedirectResponse("/operator/login", status_code=303)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request, exc):
        return _plain(exc.status_code, exc.detail if isinstance(exc.detail, str) else "Error")

    @app.exception_handler(Exception)
    async def server_error(request, exc):
        log.error("operator page %s failed", request.url.path, exc_info=exc)
        return _plain(500, "Something went wrong: the log has the details.")

    async def require_operator(request: Request) -> Session:
        """The signed-in operator, or a redirect to sign in (403 off loopback
        with AUTH_MODE=none)."""
        if settings.auth_mode == "none":
            proxied = any(h in request.headers for h in ("x-forwarded-for", "forwarded", "x-real-ip"))
            if not proxied and auth.peer_is_loopback(request.client.host if request.client else None):
                return Session(csrf=none_csrf)
            raise StarletteHTTPException(403, "AUTH_MODE=none: the dashboard answers only on this machine")
        op = await db.get_operator()
        s = signer.read_session(request.cookies.get(COOKIE), (op or {}).get("session_epoch") or 0) \
            if op and op["password_hash"] else None
        if s is None:
            raise NeedSignIn()
        return s

    async def check_csrf(request: Request, session: Session = Depends(require_operator)) -> Session:
        """Every POST: the session's CSRF token, as the `csrf` form field or
        the X-CSRF-Token header."""
        sent = request.headers.get("x-csrf-token")
        if sent is None and request.headers.get("content-type", "").startswith(
                ("application/x-www-form-urlencoded", "multipart/form-data")):
            sent = (await request.form()).get("csrf")
        if not sent or not secrets.compare_digest(str(sent), session.csrf):
            raise StarletteHTTPException(403, "The form expired or came from elsewhere: reload and try again.")
        return session

    def plain_http_refused(request: Request) -> HTMLResponse | None:
        peer = request.client.host if request.client else None
        if request.url.scheme == "https":
            return None
        if auth.peer_is_local(peer):
            if gateway and peer == gateway and not warned_gateway:
                warned_gateway.append(peer)
                log.warning("plain-HTTP sign-in from %s, the gateway: this Docker setup hides visitors' addresses "
                            "(Docker Desktop, rootless Docker, Podman), so everyone looks local. Never expose this "
                            "port to the internet without HTTPS in front (see .env.example).", peer)
            return None
        return _plain(403, "Sign in over HTTPS: signing in over plain HTTP is refused from outside this network.")

    def signed_in(request: Request, epoch: int) -> RedirectResponse:
        value, _ = signer.new_session(epoch)
        r = RedirectResponse("/operator/", status_code=303)
        r.set_cookie(COOKIE, value, max_age=SESSION_DAYS * 86400, path="/operator", httponly=True,
                     samesite="strict",
                     secure=request.url.scheme == "https" or settings.base_url.startswith("https://"))
        return r

    @app.get("/login")
    async def login_form(request: Request):
        if settings.auth_mode == "none":
            return RedirectResponse("/operator/", status_code=303)
        if refused := plain_http_refused(request):
            return refused
        return page("login.html", csrf=signer.form_token(), error=None)

    @app.post("/login")
    async def login(request: Request, password: str = Form(""), csrf: str = Form("")):
        if settings.auth_mode == "none":
            return RedirectResponse("/operator/", status_code=303)
        if refused := plain_http_refused(request):
            return refused
        if not signer.form_token_ok(csrf):
            return _plain(403, "The form expired: reload and try again.")
        result = await auth.check_password(db, password, now())
        if result == "locked":
            return page("login.html", 429, csrf=signer.form_token(),
                        error="Too many wrong passwords: sign-in is locked for 15 minutes.")
        if result != "ok":
            return page("login.html", 401, csrf=signer.form_token(), error="That isn't the password.")
        op = await db.get_operator()
        return signed_in(request, op["session_epoch"] or 0)

    @app.get("/setup")
    async def setup_form(request: Request, token: str = ""):
        if refused := plain_http_refused(request):
            return refused
        if not await auth.setup_token_ok(db, token, now()):
            return _plain(403, "This setup link has been used or has expired. "
                               "Run `safe-streets reset-password` for a new one.")
        return page("setup.html", token=token, csrf=signer.form_token(), error=None)

    @app.post("/setup")
    async def setup(request: Request, token: str = Form(""), password: str = Form(""), password2: str = Form(""),
                    csrf: str = Form("")):
        if refused := plain_http_refused(request):
            return refused
        if not signer.form_token_ok(csrf):
            return _plain(403, "The form expired: reload and try again.")
        if not await auth.setup_token_ok(db, token, now()):
            return _plain(403, "This setup link has been used or has expired.")
        error = None
        if len(password) < auth.MIN_PASSWORD:
            error = f"Use {auth.MIN_PASSWORD} characters or more."
        elif password != password2:
            error = "The two passwords differ."
        if error:
            return page("setup.html", 400, token=token, csrf=signer.form_token(), error=error)
        await auth.set_password(db, password)
        op = await db.get_operator()
        return signed_in(request, op["session_epoch"] or 0)

    @app.post("/logout")
    async def logout(_s: Session = Depends(check_csrf)):
        r = RedirectResponse("/operator/login", status_code=303)
        r.delete_cookie(COOKIE, path="/operator")
        return r

    from .views import add_views
    add_views(app, settings, db, batcher, source, vision, now=now, require=require_operator, csrf=check_csrf)
    app.add_middleware(_Headers)
    return app
