"""The operator's sign-in (AUTH_MODE=password, the default).

- First start with no password: a one-time setup link (a random 32-byte
  token, stored as its sha256, single use, 24 h) is logged.
- Passwords: argon2id (argon2-cffi's defaults), 12 characters or more.
- Session: the `ss_session` cookie, signed with itsdangerous (the secret is
  made in the data volume on first start), HttpOnly, SameSite=Strict,
  Path=/operator, 30 days, carrying the session epoch and a CSRF token.
  `reset-password` bumps the epoch, so every earlier session ends.
- Five wrong passwords in a row lock sign-in for 15 minutes, counted one
  check at a time (guesses sent together each count). argon2 runs in a
  worker thread, off the event loop.
- Every POST carries a CSRF token: the session's, or before sign-in a
  short-lived signed one.
- No sign-in over plain HTTP unless the client is loopback or private.
  Docker Desktop, rootless Docker and Podman show every visitor as the
  container's gateway (a private address), and so does Linux Docker for a
  visitor on 127.0.0.1 or over some VPNs: there the rule can't tell a
  neighbour from the internet. The app allows it and warns once (M9).

AUTH_MODE=none serves the dashboard to loopback clients only.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from weakref import WeakKeyDictionary

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from itsdangerous import BadSignature, URLSafeTimedSerializer

COOKIE = "ss_session"
SESSION_DAYS = 30
MIN_PASSWORD = 12
MAX_FAILED = 5
LOCK = timedelta(minutes=15)
SETUP_TTL = timedelta(hours=24)
FORM_TTL_S = 3600

_ph = PasswordHasher()
# One check at a time per database: each reads the count of wrong passwords
# and writes it back, so guesses sent together must not all read the same one.
_checking: WeakKeyDictionary = WeakKeyDictionary()


@dataclass(frozen=True)
class Session:
    csrf: str


def load_secret(data_dir: Path) -> bytes:
    """The signing secret, made on first start (0600) in the data volume."""
    path = Path(data_dir) / "secret.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    return path.read_text().strip().encode()


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat()


async def new_setup_token(db, now: datetime) -> str:
    """A fresh one-time setup token (the previous one stops working)."""
    token = secrets.token_urlsafe(32)
    await db.update_operator(setup_token_hash=_sha(token), setup_expires=_iso(now + SETUP_TTL))
    return token


async def ensure_setup(db, now: datetime) -> str | None:
    """On start: a setup token to log if no password is set yet, else None."""
    op = await db.get_operator()
    if op and op["password_hash"]:
        return None
    return await new_setup_token(db, now)


async def setup_token_ok(db, token: str, now: datetime) -> bool:
    op = await db.get_operator()
    if not op or not op["setup_token_hash"] or not token or op["password_hash"]:
        return False
    if not hmac.compare_digest(op["setup_token_hash"], _sha(token)):
        return False
    return now < datetime.fromisoformat(op["setup_expires"])


async def set_password(db, password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"the password needs {MIN_PASSWORD} characters or more")
    await db.update_operator(password_hash=await asyncio.to_thread(_ph.hash, password), setup_token_hash=None, setup_expires=None,
                             failed=0, locked_until=None)


async def reset_password(db, now: datetime) -> str:
    """The CLI's reset: forget the password, end every session, and return a
    new setup token."""
    op = await db.get_operator() or {}
    await db.update_operator(password_hash=None, session_epoch=(op.get("session_epoch") or 0) + 1,
                             failed=0, locked_until=None)
    return await new_setup_token(db, now)


async def check_password(db, password: str, now: datetime) -> str:
    """"ok", "wrong" or "locked"."""
    lock = _checking.setdefault(db, asyncio.Lock())
    async with lock:
        return await _check_password(db, password, now)


async def _check_password(db, password: str, now: datetime) -> str:
    op = await db.get_operator()
    if not op or not op["password_hash"]:
        return "wrong"
    if op["locked_until"] and now < datetime.fromisoformat(op["locked_until"]):
        return "locked"
    try:
        await asyncio.to_thread(_ph.verify, op["password_hash"], password)
    except (VerificationError, InvalidHashError):
        failed = (op["failed"] or 0) + 1
        await db.update_operator(failed=0 if failed >= MAX_FAILED else failed,
                                 locked_until=_iso(now + LOCK) if failed >= MAX_FAILED else op["locked_until"])
        return "wrong"
    await db.update_operator(failed=0, locked_until=None)
    return "ok"


class Signer:
    def __init__(self, secret: bytes):
        self.session = URLSafeTimedSerializer(secret, salt="ss-session")
        self.form = URLSafeTimedSerializer(secret, salt="ss-form")

    def new_session(self, epoch: int) -> tuple[str, str]:
        csrf = secrets.token_urlsafe(24)
        return self.session.dumps({"e": epoch, "c": csrf}), csrf

    def read_session(self, value: str | None, epoch: int) -> Session | None:
        if not value:
            return None
        try:
            d = self.session.loads(value, max_age=SESSION_DAYS * 86400)
        except BadSignature:
            return None
        if not isinstance(d, dict) or d.get("e") != epoch or not d.get("c"):
            return None
        return Session(csrf=d["c"])

    def form_token(self) -> str:
        """A CSRF token for the sign-in and setup forms (no session yet)."""
        return self.form.dumps(secrets.token_urlsafe(8))

    def form_token_ok(self, value: str | None) -> bool:
        if not value:
            return False
        try:
            self.form.loads(value, max_age=FORM_TTL_S)
        except BadSignature:
            return False
        return True


# RFC 1918 and IPv6 unique local addresses. Not ipaddress's is_private,
# which also counts documentation and other reserved ranges.
PRIVATE = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")]  # prepublish: allow


def peer_is_local(host: str | None) -> bool:
    """Loopback or a private (RFC 1918 / ULA) address."""
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback or any(ip in n for n in PRIVATE)


def default_gateway(route: Path | str = "/proc/net/route") -> str | None:
    """This machine's (in Docker, the container's) default gateway, from
    Linux's routing table; None where it can't be read."""
    try:
        lines = Path(route).read_text().splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        f = line.split()
        if len(f) > 2 and f[1] == "00000000" and f[2] != "00000000":
            try:
                return str(ipaddress.IPv4Address(int(f[2], 16).to_bytes(4, "little")))
            except ValueError:
                return None
    return None


def peer_is_loopback(host: str | None) -> bool:
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False
