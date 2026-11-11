#!/bin/sh
# Copy RTSP_URL's video into /footage as segments of 60 s named by their
# start time in TZ (safe-streets' default SOURCE_PATTERN, %Y%m%d-%H%M%S.mp4),
# and delete segments older than RECORDER_KEEP_HOURS (48).
#
# -c copy: the camera's own frames and timestamps, never re-encoded.
# -reset_timestamps 0: each segment's pts carry on from the last, so the
#   app joins segments on their pts, not only on their names.
# If the stream drops, ffmpeg exits and compose restarts the container; the
# segment it was writing is left unfinished and the app skips it.
set -eu
: "${RTSP_URL:?set RTSP_URL in .env}"
KEEP_HOURS="${RECORDER_KEEP_HOURS:-48}"
OUT=/footage

(
  while true; do
    find "$OUT" -maxdepth 1 -type f -name '*.mp4' -mmin "+$((KEEP_HOURS * 60))" -delete
    sleep 600
  done
) &

exec ffmpeg -hide_banner -loglevel warning \
  -rtsp_transport tcp -i "$RTSP_URL" \
  -map 0:v:0 -c copy \
  -f segment -segment_time 60 -segment_format mp4 -reset_timestamps 0 -strftime 1 \
  "$OUT/%Y%m%d-%H%M%S.mp4"
