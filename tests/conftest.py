import os

import pytest

# The suite never sees a .env. safe_streets.config loads one on import, searching
# up from its own directory, so from a worktree it finds the main checkout's,
# which is production's (MOTION_GATE=shadow failed the default tests). Undo
# whatever that load added.
_before = set(os.environ)
import safe_streets.config  # noqa: E402,F401
for _key in set(os.environ) - _before:
    del os.environ[_key]

os.environ.setdefault("PROTECT_HOST", "192.0.2.1")
os.environ.setdefault("PROTECT_USERNAME", "u")
os.environ.setdefault("PROTECT_PASSWORD", "p")
os.environ.setdefault("TZ", "America/Los_Angeles")  # the tests' times are written in Pacific time


@pytest.fixture
def settings(tmp_path):
    from safe_streets.config import Settings

    # The night fixtures are the reference install's, so its night lane line.
    return Settings(data_dir=tmp_path, models_dir=tmp_path / "models", speed_limit=20.0,
                    night_lane_y0=425.0, night_lane_y1=440.0)


@pytest.fixture
async def db(tmp_path):
    from safe_streets.db import Database

    d = Database(tmp_path / "safe_streets.db")
    await d.init()
    yield d
    await d.close()
