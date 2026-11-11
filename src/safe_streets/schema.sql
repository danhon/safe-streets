CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,           -- Protect event id
    start_ts    TEXT NOT NULL,              -- ISO 8601 UTC
    end_ts      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued|exporting|processing|done|no_vehicle|failed
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    clip_path   TEXT,                       -- NULL once pruned or deleted
    fps         REAL,
    ms          INTEGER,
    first_done_at TEXT,                     -- first live analysis finished; NULL until then
    stages_json TEXT,                       -- the event line's stages (pipeline._stages) and ms, as JSON
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS events_status ON events(status, start_ts);

CREATE TABLE IF NOT EXISTS passes (
    id              INTEGER PRIMARY KEY,
    event_id        TEXT NOT NULL,
    ts              TEXT NOT NULL,          -- UTC, car at mid-road
    cls             TEXT NOT NULL,
    direction       INTEGER NOT NULL,       -- +1 along road X, -1 against
    mph             REAL,                   -- NULL when uncalibrated or camera moved
    mps             REAL,                   -- the same speed in m/s, what displays convert
    plus_minus      REAL,
    metres          REAL,
    length_m        REAL,                   -- box bottom edge through H: a scale sanity check
    seconds         REAL,
    n               INTEGER,
    flags           TEXT NOT NULL DEFAULT '',
    hidden          INTEGER NOT NULL DEFAULT 0,
    shot_path       TEXT,
    hero_path       TEXT,
    card_path       TEXT,                   -- the post's image, with the tracker's overlay
    track_json      TEXT,                   -- overlay.Overlay as JSON, to redraw it
    body            TEXT,                   -- vehicle type: car, SUV, pickup, van, bus or truck; NULL unsure, skipped or not asked
    vehicle_json    TEXT,                   -- the classifier's reply (encoder, scores, guesses) or {encoder, skipped}
    colour          TEXT,                   -- light, dark, blue, red or other, from the sidecar; NULL not known
    calibration_id  INTEGER,
    clip_t          REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS passes_ts ON passes(ts);

CREATE TABLE IF NOT EXISTS calibrations (
    id              INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    image_pts       TEXT NOT NULL,          -- JSON
    world_pts       TEXT NOT NULL,          -- JSON, metres
    H               TEXT NOT NULL,          -- JSON 3x3, image -> world
    road            TEXT NOT NULL,          -- JSON image polygon
    reference_path  TEXT NOT NULL,
    status          TEXT NOT NULL,          -- provisional|active|stale
    active          INTEGER NOT NULL DEFAULT 1,
    lane            TEXT,                   -- JSON image polygon; NULL: the road
    trusted         TEXT,                   -- JSON image polygon; NULL: everywhere
    night_lane      TEXT,                   -- JSON [y_left, y_right]
    occluders       TEXT,                   -- JSON [x, ...]
    mask            TEXT,                   -- JSON [[polygon], ...]
    size            TEXT                    -- JSON [w, h] of the snapshot; NULL: 1920x1080
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL                     -- JSON
);

CREATE TABLE IF NOT EXISTS posts (
    id          INTEGER PRIMARY KEY,
    event_id    TEXT NOT NULL,
    ts          TEXT NOT NULL,              -- the pass's ts
    direction   INTEGER NOT NULL,
    mph         REAL,
    text        TEXT NOT NULL,
    uri         TEXT,
    status      TEXT NOT NULL,              -- posted|unknown|failed|dry-run
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    yell_key    TEXT                        -- which yell the post used
);
CREATE INDEX IF NOT EXISTS posts_ts ON posts(ts);

-- Open-Meteo's 15-minute rain at the house (rain.py): mm in the 15 minutes
-- ending at slot. Only slots that had ended when fetched; refetches replace.
CREATE TABLE IF NOT EXISTS rain (
    slot        TEXT PRIMARY KEY,           -- UTC ISO, the slot's end
    mm          REAL NOT NULL,
    fetched_at  TEXT NOT NULL
);

-- The hourly traffic report (reporter.py): one row per hour at most, written
-- before the post is sent (`unknown`) and never retried, like `posts`.
CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY,
    hour        TEXT NOT NULL UNIQUE,       -- the hour it covers: its local start, ISO with offset
    cards       TEXT NOT NULL,              -- JSON: [{"id", "title", "alt", "path"}], in post order
    text        TEXT NOT NULL,
    uri         TEXT,
    status      TEXT NOT NULL,              -- posted|unknown|failed|dry-run
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- A single card posted by hand from /report (reporter.post_card), as the
-- bot. Written `unknown` before it is sent and never retried, like `reports`;
-- a card goes out at most once for an hour.
CREATE TABLE IF NOT EXISTS card_posts (
    id          INTEGER PRIMARY KEY,
    card        TEXT NOT NULL,              -- the card's id (cards.SPECS)
    hour        TEXT NOT NULL,              -- the hour it was drawn for: its local start, ISO with offset
    title       TEXT NOT NULL,
    alt         TEXT NOT NULL,
    text        TEXT NOT NULL,
    path        TEXT,                       -- the PNG kept in data/reports/<hour>/
    uri         TEXT,
    status      TEXT NOT NULL,              -- posted|unknown|failed|dry-run
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (card, hour)
);

-- The operator's sign-in (operator/auth.py): one row.
CREATE TABLE IF NOT EXISTS operator (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    password_hash    TEXT,                   -- argon2id; NULL until set up
    session_epoch    INTEGER NOT NULL DEFAULT 0,  -- sessions carry it; a reset bumps it
    failed           INTEGER NOT NULL DEFAULT 0,  -- wrong passwords in a row
    locked_until     TEXT,                   -- UTC ISO
    setup_token_hash TEXT,                   -- sha256 of the one-time setup token
    setup_expires    TEXT                    -- UTC ISO
);
