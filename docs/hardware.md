# Hardware

What it takes to run, measured on the reference install: a 4-core Intel
box with an HD 530 iGPU and about 5 GB of free memory, shared with other
home services. A street with a few hundred cars a day fits comfortably.

**The reference install ran YOLO11n; `safe-streets` runs RTMDet-tiny**
(YOLO11n's licence is AGPL). On this box's CPU (an Intel i5-6500, one
thread) an RTMDet-tiny call took about 345 ms of CPU against YOLO11n's
145, about 2.4 times as much (1.8 times on an ARM laptop). Scale the
detector rows below by that until someone measures them with RTMDet on an
install like this one.

## What costs what

| Work | Cost | Note |
|---|---|---|
| Finding motion in the low-res recording | 4–7 s of CPU per 5 minutes of footage | About 2% of one core, all day. |
| A car event, the detector on the CPU (one thread) | about 79 detector calls, 13.8 s of CPU | With the motion gate, episodes and 1.0 s pads (decision 0007). Without them: 243 calls, 43.3 s. |
| A car event, the detector on the iGPU | a few GPU-seconds, and far less CPU | Before the gate and episodes: about 2.6 GPU-s an event. They remove about two thirds of the detector calls. |
| A night event | about 87% less than decoding the clip twice | The headlight tracker reuses the gate's frames. |
| A batch beyond its events | about 0.1 s | So batching every minute is cheap. |

## CPU or GPU

**CPU is the default** in `safe-streets`: it runs anywhere. With the motion
gate and episodes a car cost the reference install about 14 s of one core
with YOLO11n; replaying the same events, RTMDet-tiny made about as many
calls and cost about 36 s against YOLO11n's 16 on this box's CPU. A few hundred cars a day
is two or three hours of one core, spread across the day; a busy street on
a small CPU may want the iGPU.

**An Intel iGPU** (through OpenVINO) cuts the CPU per car further. To use
it:

- run with `compose.gpu.yml`: it builds the image with Intel's OpenCL
  runtime (`intel-opencl-icd`), passes `/dev/dri` and the host's render
  group (`RENDER_GID`), and sets `YOLO_DEVICE=intel:gpu`;
- check the operator's health page says the GPU, because without the
  runtime OpenVINO falls back to the CPU without saying so;
- keep `YOLO_GPU_THROTTLE=LOW` (the default): it stopped the waiting
  thread busy-spinning, from about 45 to about 8 ms of CPU per inference;
- keep the compile cache (`OV_CACHE_DIR`): after a restart, the first
  inference took about 16 s with it and 41 s without.

**Sharing the iGPU** with another heavy GPU user (another detector, a video
transcoder, an audio model) needs care. On the reference box, every hard
freeze over its first days happened with two GPU tenants busy at once.
If yours freezes, move one of them to the CPU first.

## Sharing the box

- **Pace by CPU pressure, not the load average** (0008): set
  `MAX_CPU_PRESSURE` (60 worked) if the box runs anything that stutters
  under load, such as live video. Off by default.
- **Run at a lower priority**: the app at nice 10, tests at nice 19.
- **Memory**: the app is modest; clips are the bulk of the disk. At
  roughly 3–5 MB per 1080p clip, keep `CLIP_RETENTION_DAYS` in proportion
  to your traffic.
