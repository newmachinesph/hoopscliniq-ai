"""Court position of the ball per frame: holder position, interpolation, in-court projection, else null."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from app import settings

Point = tuple[float, float]
RIM: Point = (25.0, 5.25)
COURT_W, COURT_L = 50.0, 47.0
MARGIN_FT = 3.0
MAX_INTERP_SECONDS = 2.0


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _inside(p: Point) -> bool:
    return -MARGIN_FT <= p[0] <= COURT_W + MARGIN_FT and -MARGIN_FT <= p[1] <= COURT_L + MARGIN_FT


def ball_court_track(
    holder: Sequence[str | None],
    players: dict[str, Sequence[Point]],
    projected: Sequence[Point | None],
    events: Sequence[dict[str, Any]],
    fps: int = settings.TARGET_FPS,
) -> tuple[list[list[float] | None], dict[str, int]]:
    """([x, y, 0] or None per frame, counts of each source). Never clamps to the court edge.

    Rules, in order: (1) holder position; (2) linear interpolation between held frames at most 2 s apart;
    (5) a shot with no later holder within 2 s travels from the shooter to the rim over its frames;
    (3) the floor projection when inside the court plus a 3 ft margin; (4) else None.
    """
    n = len(holder)
    pos: list[Point | None] = [None] * n
    src: list[str | None] = [None] * n
    for i, pid in enumerate(holder):
        track = players.get(pid) if pid else None
        if track is not None and i < len(track):
            pos[i], src[i] = (float(track[i][0]), float(track[i][1])), "held"

    held = [i for i in range(n) if src[i] == "held"]
    max_gap = round(MAX_INTERP_SECONDS * fps)
    for a, b in pairwise(held):
        if 1 < b - a <= max_gap:
            for i in range(a + 1, b):
                pos[i], src[i] = _lerp(pos[a], pos[b], (i - a) / (b - a)), "interpolated"  # type: ignore[arg-type]

    for ev in events:
        if ev.get("type") != "shot":
            continue
        start, end = ev.get("frame"), ev.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or pos[start] is None or end <= start:
            continue
        for i in range(start + 1, min(end, n - 1) + 1):
            if src[i] is None:
                pos[i], src[i] = _lerp(pos[start], RIM, (i - start) / (end - start)), "interpolated"  # type: ignore[arg-type]

    for i in range(n):
        p = projected[i] if i < len(projected) else None
        if src[i] is None and p is not None and _inside(p):
            pos[i], src[i] = p, "projected"

    counts = {"held": 0, "interpolated": 0, "projected": 0, "missing": 0}
    out: list[list[float] | None] = []
    for p, s in zip(pos, src, strict=True):
        counts[s or "missing"] += 1
        out.append(None if p is None else [round(p[0], 2), round(p[1], 2), 0])
    return out, counts
