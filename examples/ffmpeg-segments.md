# Recording an RTSP camera into segments

safe-streets never watches a live stream. It analyses recorded footage in
batches, and it needs clips whose frames carry their real timestamps. If
your camera has an NVR that keeps continuous recordings as files, point
`SOURCE_DIR` (or `FOOTAGE_DIR`, with compose) at them. If it has only an
RTSP stream, something has to record it. The bundled `recorder` service
does that, and so does the ffmpeg command below.

## Check the stream's frame rate first

A camera's RTSP stream can run at a lower frame rate than its own
recordings. The camera safe-streets was developed on streamed 3–7 fps over
RTSP against ~25 fps in its recordings. Speeds come from each frame's time,
so a low rate still measures, but cars cross in fewer frames: below
`MIN_FPS` (5) a pass is flagged rough, and a fast car can cross in too few
frames to measure at all.

Count the frames in 20 seconds of the stream:

```sh
ffmpeg -rtsp_transport tcp -i "$RTSP_URL" -t 20 -map 0:v:0 -c copy -f null - 2>&1 | grep -o 'frame= *[0-9]*' | tail -1
```

Divide by 20. Then ask what the stream says it is:

```sh
ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,r_frame_rate \
  -of default=nw=1 "$RTSP_URL"
```

If the counted rate is well below the advertised one, look in the camera's
settings for the stream's frame rate, or use its main (highest quality)
stream rather than a sub-stream.

## With compose

In `.env`:

```sh
COMPOSE_PROFILES=recorder
RTSP_URL=rtsp://user:password@camera.local:554/stream1
TZ=Europe/London          # segment names are local time in TZ
# FOOTAGE_DIR=./footage   # where segments go; the app reads it read-only
# RECORDER_KEEP_HOURS=48  # older segments are deleted
```

then `docker compose up -d --build`.

## Without compose

```sh
mkdir -p footage
TZ=Europe/London ffmpeg -hide_banner -loglevel warning \
  -rtsp_transport tcp -i "$RTSP_URL" \
  -map 0:v:0 -c copy \
  -f segment -segment_time 60 -segment_format mp4 -reset_timestamps 0 -strftime 1 \
  "footage/%Y%m%d-%H%M%S%z.mp4"
```

- `-c copy` keeps the camera's own frames and timestamps. Never re-encode
  (no `-r`, no `-vf`): a re-encode at a fixed rate replaces each frame's
  real time with an even grid, and the speeds go wrong.
- `-reset_timestamps 0` carries each segment's timestamps on from the last,
  so the app joins segments on them, not only on their names.
- The names are each segment's start in local time with its UTC offset
  (`20261101-013000-0700.mp4`), safe-streets' default `SOURCE_PATTERN`
  (`%Y%m%d-%H%M%S%z.mp4`). Keep the `%z`: without it, the hour the clocks
  go back gets the same names as the hour before, overwriting it. Name
  them any other way and set `SOURCE_PATTERN` to match (a pattern without
  `%z` is read in `TZ`).
- A segment still being written (modified within `SOURCE_SETTLE_S`, 10 s)
  is left alone, and one that ffmpeg never finished (the stream dropped)
  is skipped.
- Delete old segments yourself, e.g. hourly from cron:
  `find footage -name '*.mp4' -mmin +2880 -delete`.

Run it under a process supervisor (systemd, compose's restart policy): when
the stream drops, ffmpeg exits.
