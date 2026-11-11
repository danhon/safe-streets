"""Times and text as the log shows them.

The DB keeps UTC. A log line's own time is local (the container's `TZ`),
which is what logdy's Time column and Protect show, so a time inside a
message is given in local time too, with its offset: `21:30` in a message
then reads as `21:30` on the line (2026-11-06).

Logdy reads one line per record: a line without its own timestamp shows
untimed, red, and sorted away from its record, so text that spans lines
(a post) is logged on one line, its breaks shown as ⏎.
"""

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def local(ts, tz: str) -> str:
    """`ts` (a datetime, or an ISO string from the DB; naive is UTC) in `tz`,
    to the millisecond, with its offset: `2026-11-06T21:30:40.892-08:00`.
    Anything that isn't a time is given as it is."""
    try:
        t = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(ZoneInfo(tz)).isoformat(timespec="milliseconds")
    except Exception:  # noqa: BLE001  (a log line must never raise)
        return str(ts)


def one_line(text: str) -> str:
    """`text` on one line, each run of line breaks shown as ⏎."""
    return re.sub(r"\s*\n\s*", " ⏎ ", text.strip())
