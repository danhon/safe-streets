# With Frigate

[Frigate](https://frigate.video) records cameras continuously, but in nested
folders by UTC date and hour (`recordings/YYYY-MM-DD/HH/<camera>/MM.SS.mp4`).
safe-streets' folder source reads one folder of segments named by their
start, so it can't read Frigate's recordings directly yet.

Record a second copy from Frigate's restream instead. Frigate's go2rtc
serves each camera's stream without re-encoding at
`rtsp://<frigate-host>:8554/<camera>`. Point the bundled recorder at it, so
the camera has one connection (Frigate's):

```sh
COMPOSE_PROFILES=recorder
RTSP_URL=rtsp://frigate.local:8554/front
```

The restream is the camera's stream, so check its frame rate as
[ffmpeg-segments.md](ffmpeg-segments.md) says before relying on it: Frigate
may restream a lower-rate sub-stream than it records.

This keeps a second copy of the footage (`RECORDER_KEEP_HOURS`, 48 by
default). Reading Frigate's own folders is a possible later source.
