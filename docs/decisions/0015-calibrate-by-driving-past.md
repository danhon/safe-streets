# 0015: Calibrate by driving past at a known speed

Status: accepted (reference install, 2026-11).

## Context

The textbook calibration is a homography from measured points on the road:
tape-measure four or more marks, click them, type their positions. It takes
a tape, chalk, time on the road and care, and a mistake in one point skews
every speed. A side-on camera only needs the distance along the road.

## Decision

Calibrate by driving past at a known, steady speed. The app takes the
drive's track and works out the scale (pixels per metre along the road)
that makes it read the true speed. Two drives set it; more drives, at other
speeds and directions, check it. Passes far from the drives' wheel line are
marked rough until someone drives there.

The reference install has used only this: its scale came from two 15 mph
drives (they read 15.7 and 14.5 against it), and three later drives (20,
~15 and 19 mph) read 21.9, 15.8 and 19.3. Nobody measured anything on the
ground.

## Consequences

- Anyone with a car and a steady speed can calibrate; it also tests the
  whole pipeline end to end (camera, clip timing, detector, tracker).
- One scale suits a side-on view. On the reference camera, cars' own speeds
  were flat across the frame to within about 3%. A camera looking along the
  road needs a fuller model.
- The ground-point homography remains in the code as an experimental
  option. **It has never been tested on a real street.**
