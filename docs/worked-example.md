# Worked example: one house, one camera

Every value the reference install uses, and why, so you can map yours. It's
one side-on camera over a residential street with a 20 mph limit, analysed
on a small Linux box in the house, from 2026-11. Where a value is about this
scene and no other, it says so: yours will differ.

## The camera and the recorder

| | Here | Why it matters |
|---|---|---|
| Camera | UniFi G3 Flex, 1920×1080, fixed, side-on to the road | Side-on keeps one scale along the road good enough (calibration, step 4). |
| Recorder | UniFi Protect, continuous recording; clips exported through its private API | The app analyses recordings, never a live stream (decision 0002). |
| Low-res channel | 640×360, used only to find motion | 5 min of footage: 14 MB, exported in 1.4 s, scanned in 4–7 s of CPU. |
| Recorded frame rate | 8.7–20 fps in the default low-light mode, ~25 in the faster-shutter mode | Speeds use each frame's own time (0001). The live RTSP stream ran at only 3–7 fps, which is one reason not to use it. |
| Video mode | Slow shutter (low-light) by default; a faster-shutter mode was trialled | In low sun, the faster shutter let the vehicle-type model name 80% of passes against 44% (0014). Switching mode rebooted the camera. |
| On-screen clock | The top 90/1080 of the frame | Masked, so it never counts as motion and never appears in a crop. |

## The scene

| | Here | Note |
|---|---|---|
| Road outline (wheels allowed) | x 870–1920, y about 457–615 | Scene-specific. Starts where a tree hides the left of the road. |
| Lanes | Near lane in scope, both directions; far lane out (its wheels sit at y≈443, above the outline) | Lane, not direction (0005): a wrong-way car in the near lane is kept. |
| Night lane line | y 425 at the left edge to 440 at the right | Headlights sit higher than wheels; a tall vehicle's lights can cross the line. |
| Occluder | A street tree's trunk at x≈1235 | Splits a headlight's track; the night stitch joins the pieces (0010). |
| Parked cars | Often along the near kerb | They hide the back of passing cars, hence following the leading edge (0006). |

## The calibration

| | Here |
|---|---|
| Method | Drive-bys at known speeds only; nothing measured on the ground (0015). |
| Scale | 90.4 px/m along the road, from two 15 mph drives (they read 15.7 and 14.5). |
| Checks | 20 mph read 21.9; ~15 read 15.8; 19 read 19.3. |
| Trusted region | Near the drives' wheel line (image y about 512–561); passes lower in the frame (nearer the camera, y over 560) are marked rough. |
| Calibration car | About 5 m long; its box measured 5.9 m, so boxes read about 1.2× long. Watch the median car length for drift, not its value. |

## The thresholds

| Setting | Value | Why |
|---|---|---|
| `MIN_TRACK_M` / `MIN_TRACK_S` | 4 m / 0.5 s | Shorter tracks read slow; they're marked rough. |
| `MAX_UNCERTAINTY_MPH` | 3 | A wider ± is rough. "Good" needs ± ≤ 2 and no caveat. |
| `PARKED_MPH` | 3 | Slower is parked or parking; dropped. |
| `MIN_FPS` | 5 | Below this a clip's speeds are rough. |
| Detector | YOLO11n, threshold 0.35 | `safe-streets` ships RTMDet-tiny at 0.25 instead (YOLO11n is AGPL). Replayed here, it missed fewer confirmed cars, most of the gain at dusk, for about 2.4× the CPU a call on this box. |
| `TRACK_PAD_BEFORE_S` / `_AFTER_S` | 1.0 / 1.0 | 0.5 s after lost a car (0007). |
| `MOTION_GATE`, `EPISODES` | on, on | Two thirds less detector work, same results (0007). |
| Night car / speed cover | 60% / 80% of the outline's width | Precision over recall at night (0009). |
| Night stitch | gap 0.3 s, 1.0 m at the handover | Only kept when it changes no shown speed (0010). |

## Running it

| Setting | Value | Why |
|---|---|---|
| Batch interval | 60 s | A batch costs ~0.1 s beyond its events, so the interval only sets how soon results appear. |
| Events per batch | 10 | Lets a backlog's results out as it goes. |
| CPU pressure gate | 60% (`some avg60`); a batch runs anyway after ~30 busy minutes | The box is shared; normal days ran 6–40% (0008). |
| Recorder timeouts | 60 s a call, 120 s an export try, 3 tries | One export once hung for 36 hours (0012). |
| Clip retention | 30 days | Pass rows are kept for good. |
| Process priority | nice 10; tests at nice 19 | So the box's other services come first. |

## What it found

Near-lane cars in both directions, by day and by night, each with a speed
and its ±, or counted without a speed when the track was too short to say.
The numbers it publishes are counts, shares over the limit and speeds
rounded, with no pictures (0013).
