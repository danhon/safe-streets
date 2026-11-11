"""Export the default detector, RTMDet-tiny (mmdetection 3.3.0, Apache-2.0),
to an OpenVINO IR at FP32: the network and its box decode, without NMS
(safe_streets.detector does that). Then check safe_streets.detector's
detections against mmdetection's own inference_detector on a frame of your
own (any photo with cars in it).

Pinned inputs:
- mmdetection 3.3.0's configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py
- weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth from
  https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/
  sha256 78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb
- the environment: Python 3.11, torch 2.1.2, mmcv 2.1.0 (built from source),
  mmdet 3.3.0, openvino 2026.4.1. None of these is a safe-streets dependency.

  python scripts/export_rtmdet.py CONFIG WEIGHTS.pth models/rtmdet-tiny FRAME.jpg LICENSE

The IR's sha256 is pinned in safe_streets/detector.py (RTMDET_SHA256).
Inputs: images (1,3,640,640), BGR, the keep-ratio resize padded with 114 and
normalised by RTMDet's mean/std. Outputs: boxes (1,8400,4) xyxy in the 640
input's pixels, scores (1,8400,80) after sigmoid.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from mmdet.apis import inference_detector, init_detector

config, weights, outdir, frame, licence = map(Path, sys.argv[1:6])
WEIGHTS_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
if hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHTS_SHA256:
    sys.exit(f"{weights} isn't the pinned RTMDet-tiny weights (sha256 {WEIGHTS_SHA256})")
outdir.mkdir(parents=True, exist_ok=True)
model = init_detector(str(config), str(weights), device="cpu").eval()
head = model.bbox_head


class Deploy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = model
        sizes = [(640 // s, 640 // s) for s, _ in head.prior_generator.strides]
        self.register_buffer("priors", torch.cat(head.prior_generator.grid_priors(sizes, device="cpu")))

    def forward(self, images):
        cls_scores, bbox_preds = self.m.bbox_head(self.m.extract_feat(images))
        scores = torch.cat([c.permute(0, 2, 3, 1).reshape(1, -1, head.cls_out_channels) for c in cls_scores], 1).sigmoid()
        dist = torch.cat([b.permute(0, 2, 3, 1).reshape(1, -1, 4) for b in bbox_preds], 1)
        p = self.priors[None, :, :2]
        boxes = torch.cat([p - dist[..., :2], p + dist[..., 2:]], -1)  # DistancePointBBoxCoder.decode
        return boxes, scores


d = Deploy().eval()
with torch.no_grad():
    ir = ov.convert_model(d, example_input=torch.zeros(1, 3, 640, 640),
                          input=[("images", [1, 3, 640, 640], ov.Type.f32)])
for o, n in zip(ir.outputs, ("boxes", "scores")):
    o.get_tensor().set_names({n})
ov.save_model(ir, str(outdir / "model.xml"), compress_to_fp16=False)
shutil.copy(licence, outdir / "LICENSE")

# The check: mmdetection's own pipeline and NMS vs the IR through the bench adapter
import cv2  # noqa: E402

from safe_streets.detector import RTMDET_DIR, RTMDet  # noqa: E402

img = cv2.imread(str(frame))
ref = inference_detector(model, img).pred_instances
keep = ref.scores > 0.35
rb, rl, rs = ref.bboxes[keep].numpy(), ref.labels[keep].numpy(), ref.scores[keep].numpy()
r = RTMDet(outdir.parent if outdir.name == RTMDET_DIR else outdir, "cpu", 0.35, classes=range(80), check=False).detect(img)
ob, ol, os_ = r.xyxy, r.cls.astype(int), r.conf
print("mmdet dets >0.35:", len(rs), "adapter:", len(os_))
for b, lab, s in sorted(zip(rb, rl, rs), key=lambda x: -x[2])[:8]:
    print("  mmdet  ", int(lab), np.round(b, 2), round(float(s), 4))
for b, lab, s in sorted(zip(ob, ol, os_), key=lambda x: -x[2])[:8]:
    print("  adapter", int(lab), np.round(b, 2), round(float(s), 4))
diff = -1.0
if len(rs) == len(os_) and len(rs):
    a = sorted(zip(rl, rs, map(tuple, rb)), key=lambda x: (-x[1], x[0]))
    b = sorted(zip(ol, os_, map(tuple, ob)), key=lambda x: (-x[1], x[0]))
    diff = float(max(np.abs(np.array(x[2]) - np.array(y[2])).max() for x, y in zip(a, b)))
print("max box diff (px):", diff)
meta = {"config": str(config), "weights": weights.name,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "mmdet": "3.3.0 (44ebd17)",
        "ir_from": "pytorch (ov.convert_model): network + DistancePointBBoxCoder decode, no NMS",
        "check": {"frame": frame.name, "mmdet_dets": int(len(rs)), "adapter_dets": int(len(os_)), "max_box_diff_px": diff}}
(outdir / "export.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
