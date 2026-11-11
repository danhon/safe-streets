# Setting up and calibrating

**All the reference install has ever needed is a drive-by at a known
speed.** Nobody measured anything on the ground. Its scale comes from two
drives past the camera at 15 mph, and later drives check it. This guide
leads with that method. Calibration by measured ground points also exists
in the code, but **it has never been tested on a real street**: treat it as
experimental.

Accuracy depends on calibration more than on anything else. Here is the
checklist, with why each step matters.

## 1. Place the camera

- **Side-on to the road is best.** The car crosses the frame, so it covers
  many pixels per metre and stays in view for a second or more. The
  reference camera sees the near lane across about two thirds of a 1080p
  frame. Side-on also keeps a single scale along the road good enough
  (step 4).
- **Mount it firmly.** It will still get bumped; the app checks (decision
  0011), but every move means calibrating again.
- **Check the real frame rate** of what it records, not the setting:
  `ffprobe -show_entries frame=pts_time -of csv clip.mp4`. Under about
  5 fps there's too little to fit; frame rates that vary are fine (0001).
- **Prefer a fast shutter** to more pixels (0014). Motion blur softens the
  box edges the speed depends on.
- **Mind the law** where you live on recording a public street, and on
  publishing what you record.

## 2. Check the source

**You need a recorder (an NVR), not just a camera.** The app never analyses
a live stream. If your camera only offers RTSP, use an NVR (Frigate, Blue
Iris, UniFi Protect) or the bundled ffmpeg recorder, and check the recorded
frame rate: some cameras send a slower stream over RTSP than they record
internally.

The app reads the continuous recording, not the camera's motion events,
which can miss cars (0002): one of the reference install's two calibration
drives produced no motion event at all. It needs clips with real per-frame
timestamps.

## 3. Draw the road and the lane

Sign in at `/operator` (the first time, through the one-time setup link the
app logs at start; `safe-streets reset-password` makes a new one), open
**Calibrate**, and on a snapshot draw:

- **The road outline** is where wheels may be. Wheels outside it are
  ignored.
- **Choose the lane in scope.** A far lane is usually too small and too
  oblique for good speeds; leave it out. Filter by lane, never by
  direction (0005): a wrong-way car in your lane is worth keeping.
- **Masks:** the camera's on-screen clock or logo, so it isn't read as
  motion.
- **Night only:** a lane line in image space (headlights sit higher than
  wheels, and a truck's sit higher still), and any occluders, such as a
  tree trunk that splits a light's track (0010).

Save it. The calibration is now a **draft**: the app tracks passes in the
lane but gives them no speed yet, so your drive-bys can be found.

## 4. Calibrate: drive past at a known speed

This is the whole calibration.

1. **Drive past in the lane you care about at a steady, known speed.** Use
   cruise control, or a GPS speedometer. A car's own dial usually reads a
   little high; the reference install's 19 mph check used the car's digital
   speedometer.
   Hold the speed through the whole view, so set it well before the camera.
2. **Note the time** you passed, roughly.
3. On **Calibrate**, pick the pass the app found near that time and enter
   the true speed (in `UNITS`). The app works out the scale (pixels per
   metre along the road) that makes that track read the true speed.
4. **Do it at least twice.** The reference install's scale came from two
   15 mph drives, which read 15.7 and 14.5 mph against it: about the spread
   to expect from one drive to the next. The app averages them, needs two
   or more, and shows what each drive reads against the result.
5. **Then check with other drives**, at another speed and in the other
   direction if you can, not used for the fit. On the reference install:

   | Drive | True | Read |
   |---|---|---|
   | used for the fit | 15 | 15.7 |
   | used for the fit | 15 | 14.5 |
   | check | 20 | 21.9 |
   | check (another car, an analogue dial) | ~15 | 15.8 |
   | check | 19 | 19.3 |

   Within ±1.5 mph is good; the 20 mph check (+1.9) is just outside it.
   If every check is off the same way, add those drives to the fit.

**What the scale covers.** A single scale is right near the wheel line of
the drives that set it. The reference drives' wheels sat around image
y 512–561; passes much nearer the camera are marked rough, since nobody
has driven there at a known speed. On the reference camera, a single scale
held well along the road: cars' own speeds were flat across the frame to
within about 3% (x 1200–1800). A camera that looks along the road rather
than across it will need more than one scale.

**Why it works.** Speed only needs distance along the road, so one number
(pixels per metre along the road, at the lane's distance) is enough for a
side-on view. Driving at a known speed measures exactly that number, with
your camera, your lens and your clip timing, and checks the whole pipeline
end to end at the same time.

### Experimental: measured ground points (untested)

**Calibrate** can also fit a full homography from four or more points on the
road whose positions you've measured in metres (with a tape, or from a
survey or satellite map). It would handle perspective across the lane
properly, and a camera that doesn't look side-on. **This method has never
been used on a real street.** If you try it, check it with known-speed
drives (step 4.5) before trusting it, and please report how it went. The
page rejects a fit whose re-projection error is over 0.5 m.

## 5. Keep checking

- **Drive past again** after anything changes: the camera is moved or
  re-aimed, its mode or resolution changes, or the app reports that it has
  moved (0011). A drive is the quickest check there is.
- **The median length of recent cars**, shown on **Calibrate**: a typical
  car is about 4.5 m. The reference install's boxes read about 1.2× long
  (its calibration car, about 5 m long, measured 5.9 m), so watch for drift,
  not the absolute value.
- **The ±** on each speed: a sudden rise in wide ± flags suggests a
  problem.

## 6. Size the box

On the reference install, with the motion gate and episodes on (0007), a
car event cost about 14 s of one CPU core with YOLO11n on the CPU, or a few
GPU-seconds on an Intel iGPU; `safe-streets`' detector, RTMDet-tiny, costs
about 2.4 times as much a call on the same CPU ([hardware.md](hardware.md)). A street with a few hundred cars a day
fits on a small machine. Night analysis is cheaper.
