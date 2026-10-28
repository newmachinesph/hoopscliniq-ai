"""Image boxes to court feet: foot points, a 4-point homography (per frame when the camera moves), clamping,
smoothing and gap filling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from app import settings
from app.sam import Box

# Half-court corners in court feet, in calibration order:
# [baseline-left, baseline-right, halfcourt-right, halfcourt-left]. NFHS court: 50 ft wide, 84 ft long.
COURT_CORNERS = [(0.0, 0.0), (50.0, 0.0), (50.0, 47.0), (0.0, 47.0)]
COURT_WIDTH = 50.0
COURT_LENGTH = 84.0
RIM = (25.0, 5.25)

Point = tuple[float, float]


@dataclass(frozen=True)
class Calibration:
    points: list[Point]
    image_size: tuple[int, int] | None
    source: str  # "given" | "estimated"

    def scaled_to(self, width: int, height: int) -> list[Point]:
        """Calibration points in the pixel space of the boxes (SAM reports its own frame size)."""
        if not self.image_size or not width or not height:
            return list(self.points)
        sx, sy = width / self.image_size[0], height / self.image_size[1]
        return [(x * sx, y * sy) for x, y in self.points]


def parse_calibration(raw: object, source: str = "given") -> Calibration:
    """Accept `[[x,y] x4]` or `{"points": [[x,y] x4], "imageSize": [w,h]}`. Raises ValueError otherwise."""
    size = None
    if isinstance(raw, dict):
        size_raw = raw.get("imageSize")
        if size_raw is not None:
            if not (
                isinstance(size_raw, list) and len(size_raw) == 2 and all(_num(v) and v > 0 for v in size_raw)
            ):
                raise ValueError("imageSize must be [width, height]")
            size = (int(size_raw[0]), int(size_raw[1]))
        source = raw.get("source", source)
        raw = raw.get("points")
    if not (isinstance(raw, list) and len(raw) == 4):
        raise ValueError("calibration needs 4 points")
    points = []
    for p in raw:
        if not (isinstance(p, list) and len(p) == 2 and all(_num(v) for v in p)):
            raise ValueError("each calibration point must be [x, y]")
        points.append((float(p[0]), float(p[1])))
    return Calibration(points, size, "estimated" if source == "estimated" else "given")


def _num(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def load_default_calibration(path: Path = settings.DEFAULT_CALIBRATION_PATH) -> Calibration:
    return parse_calibration(json.loads(path.read_text(encoding="utf-8")), source="estimated")


def homography(src: list[Point], dst: list[Point]) -> np.ndarray:
    """3x3 matrix H with dst ~ H @ src, from exactly 4 point pairs (direct linear transform)."""
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h: np.ndarray, x: float, y: float) -> Point:
    u, v, w = h @ np.array([x, y, 1.0])
    return (float(u / w), float(v / w))


def foot_point(b: Box) -> Point:
    return ((b.x1 + b.x2) / 2, b.y2)


def clamp(p: Point) -> Point:
    return (min(max(p[0], 0.0), COURT_WIDTH), min(max(p[1], 0.0), COURT_LENGTH))


def fill_gaps(series: list[Point | None], max_gap: int = settings.MAX_FILL_GAP) -> list[Point | None]:
    """Linear interpolation across interior gaps of at most `max_gap` frames; longer gaps stay None."""
    out = list(series)
    known = [i for i, p in enumerate(series) if p is not None]
    for a, b in pairwise(known):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            pa, pb = series[a], series[b]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                out[a + k] = (pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t)
    return out


def hold_nearest(series: list[Point | None]) -> list[Point]:
    """Fill what interpolation left (edges, long gaps) with the last seen position."""
    first = next((p for p in series if p is not None), (25.0, 23.5))
    out: list[Point] = []
    last = first
    for p in series:
        last = p if p is not None else last
        out.append(last)
    return out


def smooth(series: list[Point], window: int = settings.SMOOTH_WINDOW) -> list[Point]:
    """Centered moving average; the window shrinks at the edges."""
    half = window // 2
    n = len(series)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        xs = [p[0] for p in series[lo:hi]]
        ys = [p[1] for p in series[lo:hi]]
        out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return out


@dataclass
class CourtTrack:
    obj: int
    observed: int  # frames where SAM saw this object
    positions: list[Point]  # one court point per clip frame, feet


def top_tracks(tracks: dict[int, dict[int, Box]], limit: int = settings.MAX_TRACKS) -> list[int]:
    """Object ids of the `limit` longest tracks, longest first (ties by id for determinism)."""
    return [obj for obj, _ in sorted(tracks.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:limit]]


def court_tracks(tracks: dict[int, dict[int, Box]], mats: list[np.ndarray], frames: int) -> list[CourtTrack]:
    """Court tracks of the longest objects. `mats[i]` maps box pixels of frame i to court feet."""
    out = []
    for obj in top_tracks(tracks):
        boxes = tracks[obj]
        raw: list[Point | None] = [None] * frames
        for f, b in boxes.items():
            if 0 <= f < frames:
                raw[f] = clamp(apply_h(mats[f], *foot_point(b)))
        filled = hold_nearest(fill_gaps(raw))
        positions = [(round(x, 2), round(y, 2)) for x, y in smooth(filled)]
        out.append(CourtTrack(obj, sum(1 for p in raw if p is not None), positions))
    return out


def scale_h(sx: float, sy: float) -> np.ndarray:
    return np.diag([sx, sy, 1.0])


def image_to_court(
    h_cal: np.ndarray, to_frame0: list[np.ndarray | None] | None, cal_frame: int, frames: int
) -> list[np.ndarray]:
    """Per-frame image-to-court matrices: H_cal . H_(i->calFrame), with H_(i->calFrame) = inv(G_cal) . G_i
    where G_i maps frame i to frame 0. A frame without camera motion (None, or no motion data at all) uses the
    static H_cal. If the calibration frame itself was not registered, the nearest registered frame stands
    in."""
    if not to_frame0:
        return [h_cal] * frames
    g_cal = _nearest_registered(to_frame0, cal_frame)
    if g_cal is None:
        return [h_cal] * frames
    base = h_cal @ np.linalg.inv(g_cal)
    padded = list(to_frame0[:frames]) + [None] * max(0, frames - len(to_frame0))
    return [base @ g if g is not None else h_cal for g in padded]


def _nearest_registered(to_frame0: list[np.ndarray | None], frame: int) -> np.ndarray | None:
    for d in range(len(to_frame0)):
        for i in (frame - d, frame + d):
            if 0 <= i < len(to_frame0) and to_frame0[i] is not None:
                return to_frame0[i]
    return None


def check_quad(
    points: list[Point], size: tuple[int, int], margin: float = settings.CALIBRATION_MARGIN
) -> None:
    """Raise ValueError unless the 4 points form a convex quadrilateral inside the image.

    A corner may sit up to `margin` of the frame size outside the image: on a side-on camera the near
    baseline corner is often just off-screen and can only be extrapolated from the lines.
    """
    if len(points) != 4:
        raise ValueError("send exactly 4 points")
    w, h = size
    for x, y in points:
        if not (-margin * w <= x <= (1 + margin) * w and -margin * h <= y <= (1 + margin) * h):
            raise ValueError("a point is outside the image")
    crosses = []
    for i in range(4):
        (ax, ay), (bx, by), (cx, cy) = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        crosses.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        raise ValueError("the 4 points must form a convex quadrilateral, in corner order")
    area = 0.5 * sum(
        points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1] for i in range(4)
    )
    if abs(area) < 1e-3 * w * h:
        raise ValueError("the 4 points are too close together")
