"""Entry point: one process serves the page and runs the batch loop."""

import asyncio
import logging
import logging.handlers
import os
from pathlib import Path

import uvicorn

from .batcher import Batcher, cpu_pressure
from .config import Settings
from .db import Database
from .sources import make_source
from .vision import Vision


def log_handlers(log_dir, keep_days: int) -> tuple[list[logging.Handler], str]:
    """stdout, plus app.log in log_dir rotated at midnight and kept keep_days,
    so the log outlives the container a deploy replaces. Returns the handlers
    and a problem to report (the file couldn't be opened), or ""."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if not log_dir:
        return handlers, ""
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.TimedRotatingFileHandler(
            Path(log_dir) / "app.log", when="midnight", backupCount=keep_days, encoding="utf-8"))
    except OSError as exc:
        return handlers, f"can't open {log_dir}/app.log ({exc.strerror})"
    return handlers, ""


async def serve() -> None:
    s = Settings()
    s.require()
    handlers, problem = log_handlers(s.log_dir, s.log_keep_days)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers)
    if problem:
        logging.getLogger(__name__).warning("logging to stdout only: %s", problem)
    logging.getLogger(__name__).info("safe_streets %s starting", s.build_sha)
    db = Database(s.db_path)
    await db.init()
    await db.requeue_interrupted()
    if s.auth_mode == "password":
        from datetime import datetime, timezone

        from .operator import auth
        token = await auth.ensure_setup(db, datetime.now(timezone.utc))
        if token:
            logging.getLogger(__name__).warning(
                "no operator password yet: set one at %s (works once, for 24 hours)", setup_link(s, token))
    source = make_source(s)
    vision = Vision(s.models_dir, s.yolo_device, s.yolo_confidence, s.yolo_gpu_throttle,
                    cache_dir=s.ov_cache_dir)
    batcher = Batcher(s, db, source, vision, pressure_fn=cpu_pressure)
    from .hooks import load_hooks
    batcher.hooks = load_hooks(s.hooks)
    from .web import create_app  # the public and operator apps (Tasks 6-7)
    app = create_app(s, db, batcher, source, vision)
    config = uvicorn.Config(app, host=os.environ.get("WEB_HOST", "0.0.0.0"), port=int(os.environ.get("WEB_PORT", 8080)),
                            log_level="info", access_log=False, proxy_headers=True,
                            forwarded_allow_ips=s.forwarded_allow_ips)
    try:
        await uvicorn.Server(config).serve()
    finally:
        await source.close()
        await db.close()


def setup_link(s: Settings, token: str) -> str:
    base = s.base_url or f"http://localhost:{os.environ.get('WEB_PORT', 8080)}"
    return f"{base}/operator/setup?token={token}"


async def reset_password() -> str:
    """Forget the operator's password, end every session, and return a new
    setup link."""
    from datetime import datetime, timezone

    from .operator import auth
    s = Settings()
    db = Database(s.db_path)
    await db.init()
    try:
        return setup_link(s, await auth.reset_password(db, datetime.now(timezone.utc)))
    finally:
        await db.close()


USAGE = """usage: safe-streets [serve | reset-password]

  serve           run the app (the default)
  reset-password  forget the operator's password, sign everyone out, and
                  print a new one-time setup link"""


def main(argv: list[str] | None = None) -> None:
    import sys
    args = sys.argv[1:] if argv is None else argv
    cmd = args[0] if args else "serve"
    if cmd == "serve":
        asyncio.run(serve())
    elif cmd == "reset-password":
        link = asyncio.run(reset_password())
        print(f"The password is reset and every session ended. Set a new one at:\n{link}\n(works once, for 24 hours)")
    else:
        raise SystemExit(USAGE)
