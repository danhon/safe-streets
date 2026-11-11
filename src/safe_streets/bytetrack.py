"""ByteTrack, vendored from https://github.com/ifzhang/ByteTrack (commit
d1bf0191, yolox/tracker/: byte_tracker.py, kalman_filter.py, matching.py,
basetrack.py) and ported to numpy alone:

- detections come in as arrays (xyxy, conf, cls), and each output row keeps
  the index of the detection it was matched to this frame, so a sample can
  keep its frame's detection box rather than the Kalman state (which lags a
  car still coming into view);
- IoU in numpy instead of cython_bbox, without its +1 a side (see ious),
  the Kalman update by numpy.linalg instead of scipy, ids counted per
  tracker instead of globally. Replayed over 11,462 frames of real
  detections it matches the tracker speedcam ran frame for frame, and
  every stored speed bit for bit (2026-11-10);
- the thresholds are the ones safe_streets used through its earlier tracker:
  high 0.25, low 0.1, new track 0.25, buffer 30 frames, match 0.8, with the
  score fused into the first association.

The original's licence:

MIT License

Copyright (c) 2021 Yifu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass

import lap
import numpy as np


@dataclass(frozen=True)
class Params:
    track_thresh: float = 0.25  # high-score detections
    low_thresh: float = 0.1  # the second association's floor
    new_track_thresh: float = 0.25  # to start a track
    track_buffer: int = 30  # frames a lost track is kept
    match_thresh: float = 0.8
    fuse_score: bool = True


# -- Kalman filter (kalman_filter.py) ----------------------------------------

class KalmanFilter:
    """Constant velocity in (x, y, a, h): the box's centre, aspect ratio
    (w / h) and height, and their velocities."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros_like(measurement)]
        h = measurement[3]
        std = [2 * self._std_weight_position * h, 2 * self._std_weight_position * h, 1e-2,
               2 * self._std_weight_position * h, 10 * self._std_weight_velocity * h,
               10 * self._std_weight_velocity * h, 1e-5, 10 * self._std_weight_velocity * h]
        return mean, np.diag(np.square(std))

    def _motion_cov(self, h):
        std_pos = [self._std_weight_position * h, self._std_weight_position * h, 1e-2, self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5, self._std_weight_velocity * h]
        return np.diag(np.square(np.r_[std_pos, std_vel]))

    def multi_predict(self, mean, covariance):
        # The noise comes from each track's height before the step, as in the original.
        out = np.empty_like(covariance)
        for i in range(len(mean)):
            out[i] = self._motion_mat @ covariance[i] @ self._motion_mat.T + self._motion_cov(mean[i][3])
        return mean @ self._motion_mat.T, out

    def project(self, mean, covariance):
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h, 1e-1, self._std_weight_position * h]
        m = self._update_mat @ mean
        c = self._update_mat @ covariance @ self._update_mat.T
        return m, c + np.diag(np.square(std))

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        # K = P H^T S^-1 (the original solves it by Cholesky; S is symmetric positive definite)
        gain = np.linalg.solve(projected_cov, (covariance @ self._update_mat.T).T).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ gain.T
        new_cov = covariance - gain @ projected_cov @ gain.T
        return new_mean, new_cov


# -- tracks (basetrack.py, byte_tracker.py's STrack) ---------------------------

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class STrack:
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score: float, cls: float, idx: int):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter: KalmanFilter | None = None
        self.mean = self.covariance = None
        self.is_activated = False
        self.state = NEW
        self.score, self.cls, self.idx = score, cls, idx
        self.tracklet_len = 0
        self.track_id = 0
        self.frame_id = self.start_frame = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def multi_predict(stracks: list["STrack"]) -> None:
        if not stracks:
            return
        mean = np.asarray([st.mean.copy() for st in stracks])
        cov = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TRACKED:
                mean[i][7] = 0
        mean, cov = STrack.shared_kalman.multi_predict(mean, cov)
        for st, m, c in zip(stracks, mean, cov):
            st.mean, st.covariance = m, c

    def activate(self, kalman_filter: KalmanFilter, frame_id: int, track_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TRACKED
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, new: "STrack", frame_id: int) -> None:
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.tlwh_to_xyah(new.tlwh))
        self.tracklet_len = 0
        self.state = TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score, self.cls, self.idx = new.score, new.cls, new.idx

    def update(self, new: "STrack", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.tlwh_to_xyah(new.tlwh))
        self.state = TRACKED
        self.is_activated = True
        self.score, self.cls, self.idx = new.score, new.cls, new.idx

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


# -- matching (matching.py) ----------------------------------------------------

def ious(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N,4 tlbr) with every box in b (M,4), edges as
    coordinates (no +1 a side, unlike the original's cython_bbox), in
    float32 with 1e-7 in the union: the tracker speedcam ran computes it so,
    and at low detection scores the difference flips matches (2026-11-10)."""
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    iw = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    ih = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = iw * ih
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-7)


def iou_distance(atracks: list[STrack], btracks: list[STrack]) -> np.ndarray:
    return 1 - ious(np.array([t.tlbr for t in atracks]), np.array([t.tlbr for t in btracks]))


def fuse_score(cost: np.ndarray, detections: list[STrack]) -> np.ndarray:
    if cost.size == 0:
        return cost
    scores = np.array([d.score for d in detections])[None, :]
    return 1 - (1 - cost) * scores


def linear_assignment(cost: np.ndarray, thresh: float):
    if cost.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost.shape[0])), tuple(range(cost.shape[1]))
    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
    matches = np.asarray([[i, m] for i, m in enumerate(x) if m >= 0], dtype=int).reshape(-1, 2)
    return matches, np.where(x < 0)[0], np.where(y < 0)[0]


def _joint(a: list[STrack], b: list[STrack]) -> list[STrack]:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen and not seen.add(t.track_id)]


def _sub(a: list[STrack], b: list[STrack]) -> list[STrack]:
    drop = {t.track_id for t in b}
    return [t for t in a if t.track_id not in drop]


def _remove_duplicates(a: list[STrack], b: list[STrack]):
    pdist = iou_distance(a, b)
    dupa, dupb = set(), set()
    for p, q in zip(*np.where(pdist < 0.15)):
        if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
            dupb.add(q)
        else:
            dupa.add(p)
    return [t for i, t in enumerate(a) if i not in dupa], [t for i, t in enumerate(b) if i not in dupb]


# -- the tracker (byte_tracker.py's BYTETracker) -------------------------------

class BYTETracker:
    def __init__(self, params: Params = Params(), frame_rate: float = 30.0):
        self.p = params
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.removed: list[STrack] = []
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * params.track_buffer)
        self.kalman_filter = KalmanFilter()
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> np.ndarray:
        """One frame's detections. Returns (K, 8) rows for the tracks matched
        this frame: x0, y0, x1, y1 (the Kalman state), track id, score,
        class, and the index of the detection it was matched to."""
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []
        xyxy = np.asarray(xyxy, np.float64).reshape(-1, 4)
        conf, cls = np.asarray(conf, np.float64).reshape(-1), np.asarray(cls, np.float64).reshape(-1)
        idx = np.arange(len(conf))

        high = conf > self.p.track_thresh
        second = (conf > self.p.low_thresh) & (conf < self.p.track_thresh)

        def strs(mask):
            return [STrack(np.r_[b[:2], b[2:] - b[:2]], s, c, i)
                    for b, s, c, i in zip(xyxy[mask], conf[mask], cls[mask], idx[mask])]

        detections = strs(high)
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        # First association, with the high-score detections.
        pool = _joint(tracked, self.lost)
        STrack.multi_predict(pool)
        dists = iou_distance(pool, detections)
        if self.p.fuse_score:
            dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, self.p.match_thresh)
        for it, idet in matches:
            track, det = pool[it], detections[idet]
            if track.state == TRACKED:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)

        # Second association: the tracks left, with the low-score detections.
        detections_second = strs(second)
        r_tracked = [pool[i] for i in u_track if pool[i].state == TRACKED]
        dists = iou_distance(r_tracked, detections_second)
        matches, u_track, _ = linear_assignment(dists, 0.5)
        for it, idet in matches:
            track, det = r_tracked[it], detections_second[idet]
            if track.state == TRACKED:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)
        for it in u_track:
            track = r_tracked[it]
            if track.state != LOST:
                track.state = LOST
                lost.append(track)

        # Unconfirmed tracks (usually one frame old).
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        if self.p.fuse_score:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, 0.7)
        for it, idet in matches:
            unconfirmed[it].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconfirmed:
            unconfirmed[it].state = REMOVED
            removed.append(unconfirmed[it])

        # New tracks.
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.p.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, self._new_id())
            activated.append(track)

        for track in self.lost:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.state = REMOVED
                removed.append(track)

        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _joint(self.tracked, activated)
        self.tracked = _joint(self.tracked, refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, self.removed)
        self.removed.extend(removed)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)
        out = [t for t in self.tracked if t.is_activated and t.frame_id == self.frame_id]
        return np.array([[*t.tlbr, t.track_id, t.score, t.cls, t.idx] for t in out], np.float64).reshape(-1, 8)
