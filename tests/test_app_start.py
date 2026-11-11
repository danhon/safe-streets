"""The app runs its batches: serving the pages starts the batcher, and
stopping the app stops it (Task 9's image run found it never started)."""

from fastapi.testclient import TestClient

from safe_streets.web import create_app


class FakeBatcher:
    running = False

    def __init__(self):
        self.started = self.stopped = 0

    def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1

    def age(self):
        return 0.0


async def test_serving_starts_the_batcher_and_stopping_stops_it(db, settings):
    b = FakeBatcher()
    app = create_app(settings, db, b, None, None)
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
        assert b.started == 1 and b.stopped == 0
    assert b.stopped == 1


async def test_no_batcher_no_start(db, settings):
    with TestClient(create_app(settings, db, None, None, None)) as c:
        assert c.get("/healthz").status_code == 200
