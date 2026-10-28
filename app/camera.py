"""Camera motion: per-frame homographies from each clip frame to frame 0, from background features.

ORB features on a downscaled grey frame, with the player and ball boxes masked out, matched to the current
keyframe (ratio test, RANSAC homography). The keyframe moves forward every `CAMERA_KEYFRAME_EVERY` frames or
when the overlap gets thin, so error only accumulates at keyframe changes, not at every frame. A frame whose
match is poor gets no homography (None): the caller falls back to the static calibration for it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import settings

PixelBox = tuple[float, float, float, float]
Features = tuple[np.ndarray, np.ndarray | None]  # (points Nx2 in work pixels, ORB descriptors)


@dataclass
class CameraMotion:
    # to_frame0[i] maps clip.mp4 pixels of frame i to those of frame 0; None = poor match (fallback).
    to_frame0: list[np.ndarray | None]
    image_size: tuple[int, int]

    @property
    def frames_tracked(self) -> int:
        return sum(1 for h in self.to_frame0 if h is not None)

    @property
    def frames_fallback(self) -> int:
        return len(self.to_frame0) - self.frames_tracked

    def summary(self) -> dict[str, int]:
        return {"framesTracked": self.frames_tracked, "framesFallback": self.frames_fallback}

    def to_json(self) -> dict[str, Any]:
        return {
            "imageSize": list(self.image_size),
            "frames": len(self.to_frame0),
            **self.summary(),
            "toFrame0": [
                None if h is None else [float(f"{v:.9g}") for v in h.ravel()] for h in self.to_frame0
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CameraMotion:
        mats = [None if h is None else np.asarray(h, dtype=float).reshape(3, 3) for h in data["toFrame0"]]
        return cls(mats, (int(data["imageSize"][0]), int(data["imageSize"][1])))


def read_gray_frames(clip: Path) -> Iterator[np.ndarray]:
    """Decode clip.mp4 frame by frame as greyscale (OpenCV's bundled ffmpeg)."""
    cap = cv2.VideoCapture(str(clip))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


def _plausible(h: np.ndarray, size: tuple[int, int]) -> bool:
    """Reject degenerate fits: scale change beyond 2x, or image corners thrown more than half a frame."""
    det = float(np.linalg.det(h[:2, :2]))
    if not 0.5 < det < 2.0:
        return False
    w, h_ = size
    corners = np.float32([[0, 0], [w, 0], [w, h_], [0, h_]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
    if not np.all(np.isfinite(moved)):
        return False
    return float(np.max(np.linalg.norm(moved - corners.reshape(-1, 2), axis=1))) < 0.5 * max(w, h_)


class _Matcher:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, a: Features, b: Features) -> tuple[np.ndarray | None, int]:
        """Homography mapping frame a's work pixels to frame b's, and its RANSAC inlier count."""
        (pa, da), (pb, db) = a, b
        need = settings.CAMERA_MIN_INLIERS
        if da is None or db is None or len(da) < need or len(db) < need:
            return None, 0
        pairs = self.bf.knnMatch(da, db, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < settings.CAMERA_RATIO * p[1].distance]
        if len(good) < need:
            return None, len(good)
        src = pa[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = pb[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, settings.CAMERA_RANSAC_PX)
        if h is None or inliers is None:
            return None, 0
        n = int(inliers.sum())
        if n < need or not _plausible(h, self.size):
            return None, n
        return h, n


def detect_features(frames: Iterable[np.ndarray], scale: float = settings.CAMERA_SCALE) -> list[Features]:
    """ORB keypoints and descriptors of every frame, on a downscaled copy. No mask yet: this runs while SAM
    is still working, and `estimate_motion` drops the keypoints that fall inside player or ball boxes."""
    orb = cv2.ORB_create(nfeatures=settings.CAMERA_FEATURES)
    out: list[Features] = []
    for gray in frames:
        h, w = gray.shape[:2]
        work = (round(w * scale), round(h * scale))
        small = gray if scale == 1 else cv2.resize(gray, work, interpolation=cv2.INTER_AREA)
        kps, des = orb.detectAndCompute(small, None)
        out.append((np.float32([k.pt for k in kps]).reshape(-1, 2), des))
    return out


def drop_masked(feats: Features, boxes: Iterable[PixelBox], scale: float) -> Features:
    """Keep only keypoints outside the padded boxes (the background)."""
    pts, des = feats
    if des is None or not len(pts):
        return feats
    keep = np.ones(len(pts), dtype=bool)
    pad = settings.CAMERA_MASK_PAD
    for x1, y1, x2, y2 in boxes:
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        inside = (
            (pts[:, 0] >= (x1 - px) * scale)
            & (pts[:, 0] <= (x2 + px) * scale)
            & (pts[:, 1] >= (y1 - py) * scale)
            & (pts[:, 1] <= (y2 + py) * scale)
        )
        keep &= ~inside
    return pts[keep], des[keep]


def estimate_motion(
    features: list[Features],
    boxes: Callable[[int], list[PixelBox]],
    image_size: tuple[int, int],
    total: int,
    scale: float = settings.CAMERA_SCALE,
) -> CameraMotion:
    """Per-frame homography to frame 0 for `total` frames (frames the decoder did not return fall back)."""
    s = np.diag([scale, scale, 1.0])
    s_inv = np.diag([1 / scale, 1 / scale, 1.0])
    match = _Matcher((round(image_size[0] * scale), round(image_size[1] * scale)))
    out: list[np.ndarray | None] = []
    key: tuple[int, Features, np.ndarray] | None = None  # (frame, features, work-pixel H to frame 0)
    prev: tuple[int, Features, np.ndarray] | None = None
    for i, raw in enumerate(features[:total]):
        feats = drop_masked(raw, boxes(i), scale)
        h0 = _register(i, feats, key, prev, match)
        if h0 is None:
            out.append(None)
            prev = None
            continue
        if key is None or i - key[0] >= settings.CAMERA_KEYFRAME_EVERY or h0[1]:
            key = (i, feats, h0[0])
        prev = (i, feats, h0[0])
        out.append(s_inv @ h0[0] @ s)
    out.extend([None] * (total - len(out)))
    return CameraMotion(out, image_size)


def _register(
    i: int,
    feats: Features,
    key: tuple[int, Features, np.ndarray] | None,
    prev: tuple[int, Features, np.ndarray] | None,
    match: _Matcher,
) -> tuple[np.ndarray, bool] | None:
    """(work-pixel H from frame i to frame 0, re-anchor now?) or None when no good match exists."""
    if key is None:
        return (np.eye(3), True) if i == 0 else None
    h, n = match(feats, key[1])
    if h is not None:
        return key[2] @ h, n < settings.CAMERA_REANCHOR_INLIERS
    # The keyframe is too far away (fast pan): chain from the previous frame and re-anchor there.
    if prev is not None and prev[0] != key[0]:
        h, _ = match(feats, prev[1])
        if h is not None:
            return prev[2] @ h, True
    return None
