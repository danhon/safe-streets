"""The detector: anything that finds vehicles in an image (the `Detector`
protocol), and the default, RTMDet-tiny (mmdetection 3.3.0, Apache-2.0)
as an OpenVINO IR.

The IR (scripts/export_rtmdet.py) is the network and its box decode. This
does the rest of mmdetection's test pipeline:

- pre: Resize(scale=(640, 640), keep_ratio=True) (cv2 bilinear, as mmcv's
  imrescale), Pad to 640x640 bottom-right with 114, (img - mean) / std in
  BGR order;
- post: each prior keeps its best vehicle class (as YOLO's NMS did), then
  per-class NMS at IoU 0.65 on candidates above the threshold, at most 300,
  boxes back to the image and clipped. Filtering at the threshold before
  NMS keeps the same boxes above it as NMS on everything would: a
  lower-scoring box never suppresses a higher one.

Classes are COCO's indices, so VEHICLES in vision.py names them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

MEAN = np.array([103.53, 116.28, 123.675], np.float32)
STD = np.array([57.375, 57.12, 58.395], np.float32)
SIZE, PAD, IOU, MAX_DET = 640, 114, 0.65, 300
# The default model's folder under MODELS_DIR, and its files' sha256
# (scripts/export_rtmdet.py writes them; checked on load, which refuses any
# other files).
RTMDET_DIR = "rtmdet-tiny"
RTMDET_SHA256 = {
    "model.xml": "4df141df993a78941c88af58fd1408dc5c35cb86dbe42f160dc6070445157d50",
    "model.bin": "c213c011b860126069bc20dff471c4ec739bc61641278e88e6cacd06a6b1d400",
}


@dataclass(frozen=True)
class Detections:
    """One image's detections: xyxy (N, 4) in the image's pixels, conf (N,)
    and cls (N,) as COCO class indices."""
    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    conf: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    cls: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, keep) -> "Detections":
        return Detections(self.xyxy[keep], self.conf[keep], self.cls[keep])


class Detector(Protocol):
    names: dict[int, str]

    def detect(self, image: np.ndarray) -> Detections: ...


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        r = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[r, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[r, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        o = inter / (area(boxes[i]) + area(boxes[r]) - inter)
        order = r[o <= iou]
    return keep


def preprocess(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The network's input and the resize's x and y scales."""
    h, w = img.shape[:2]
    s = min(SIZE / w, SIZE / h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    x = np.full((SIZE, SIZE, 3), PAD, np.float32)
    x[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x), nw / w, nh / h


def postprocess(boxes: np.ndarray, scores: np.ndarray, shape, sx: float, sy: float, classes, conf: float) -> Detections:
    """boxes (P, 4) in the input's pixels and scores (P, 80) to Detections."""
    h, w = shape[:2]
    ok = np.zeros(scores.shape[1], bool)
    ok[list(classes)] = True
    scores = np.where(ok[None], scores, 0.0)
    lab = scores.argmax(1)
    pi = np.flatnonzero(scores[np.arange(len(lab)), lab] > conf)
    li = lab[pi]
    sc = scores[pi, li]
    b = boxes[pi] / np.array([sx, sy, sx, sy], np.float32)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    keep: list[int] = []
    for c in np.unique(li):
        idx = np.flatnonzero(li == c)
        keep += [int(idx[k]) for k in nms(b[idx], sc[idx], IOU)]
    keep = sorted(keep, key=lambda k: -sc[k])[:MAX_DET]
    return Detections(b[keep].astype(np.float32), sc[keep].astype(np.float32), li[keep].astype(np.int64))


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RTMDet:
    """RTMDet-tiny through OpenVINO. device: "cpu" or "intel:gpu" (OpenVINO's
    CPU or GPU)."""

    names = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, model_dir: Path, device: str = "cpu", conf: float = 0.35, classes=None, check: bool = True):
        import openvino as ov

        folder = Path(model_dir) / RTMDET_DIR
        xml = folder / "model.xml"
        if not xml.is_file():
            raise FileNotFoundError(f"no detector at {xml}: run scripts/export_rtmdet.py, or the Docker build")
        if check:
            for name, want in RTMDET_SHA256.items():
                got = _sha256(folder / name)
                if got != want:
                    raise ValueError(f"detector {folder / name}'s sha256 is {got}, not the pinned {want}: "
                                     "rebuild the image, or run scripts/fetch_detector.py")
        dev = "GPU" if device.lower().endswith("gpu") else "CPU"
        self.compiled = ov.Core().compile_model(str(xml), dev)
        self.req = self.compiled.create_infer_request()
        self.conf = conf
        self.classes = tuple(classes) if classes is not None else tuple(self.names)

    def detect(self, image: np.ndarray) -> Detections:
        x, sx, sy = preprocess(image)
        out = self.req.infer({"images": x})
        boxes = np.asarray(out[self.compiled.output("boxes")])[0]
        scores = np.asarray(out[self.compiled.output("scores")])[0]
        return postprocess(boxes, scores, image.shape, sx, sy, self.classes, self.conf)
