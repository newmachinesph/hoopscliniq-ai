"""Ball, holder, passes and shots from SAM boxes. Simple, deterministic heuristics (README "Ball and events");
every event says `detect: "heuristic"` and its confidence is only ever "low" or "medium"."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app import settings
from app.geometry import RIM, Point
from app.sam import Box

PixelBox = Sequence[float]


def centre(b: PixelBox) -> Point:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pick_ball(
    frames: dict[int, list[Box]],
    total: int,
    in_play: Callable[[int, Box], bool] | None = None,
    max_jump: float = math.inf,
) -> list[Box | None]:
    """One ball box per frame.

    Continuity first: the same SAM object as the last ball seen (within BALL_MEMORY_FRAMES), else the
    candidate nearest to it if it is at most `max_jump` px per elapsed frame away. Otherwise (and when the
    followed ball has been away from every player for BALL_IDLE_FRAMES while another ball is in play) start
    again: a candidate `in_play` (near a player, so not a ball on the scorer's table or a rack) wins, then the
    object SAM saw longest.
    """
    lengths: dict[int, int] = {}
    for boxes in frames.values():
        for b in boxes:
            lengths[b.obj] = lengths.get(b.obj, 0) + 1

    def playing(f: int, c: Box) -> bool:
        return bool(in_play and in_play(f, c))

    out: list[Box | None] = []
    last: Box | None = None
    last_frame = -(10**9)
    idle = 0
    for f in range(total):
        cands = frames.get(f) or []
        if not cands:
            out.append(None)
            continue
        pick = _follow(last, f - last_frame, cands, max_jump) if last is not None else None
        if pick is not None and idle >= settings.BALL_IDLE_FRAMES and any(playing(f, c) for c in cands):
            pick = None
        if pick is None:
            pick = min(cands, key=lambda c: (not playing(f, c), -lengths[c.obj], c.obj))
        idle = 0 if playing(f, pick) else idle + 1
        out.append(pick)
        last, last_frame = pick, f
    return out


def _follow(prev: Box, elapsed: int, cands: list[Box], max_jump: float) -> Box | None:
    if elapsed > settings.BALL_MEMORY_FRAMES:
        return None
    same = [c for c in cands if c.obj == prev.obj]
    if same:
        return same[0]
    near = min(cands, key=lambda c: (_dist(centre(_xy(c)), centre(_xy(prev))), c.obj))
    return near if _dist(centre(_xy(near)), centre(_xy(prev))) <= max_jump * elapsed else None


def _xy(b: Box) -> tuple[float, float, float, float]:
    return (b.x1, b.y1, b.x2, b.y2)


def near_player(players: dict[str, Sequence[PixelBox | None]], f: int, ball: PixelBox) -> bool:
    """True when the ball centre is inside or within HOLDER_REACH x box height of a player box in frame f."""
    cx, cy = centre(ball)
    for boxes in players.values():
        pb = boxes[f] if f < len(boxes) else None
        if pb is None:
            continue
        x1, y1, x2, y2 = pb
        d = math.hypot(max(x1 - cx, 0.0, cx - x2), max(y1 - cy, 0.0, cy - y2))
        if d <= settings.HOLDER_REACH * max(y2 - y1, 1.0):
            return True
    return False


def find_holders(
    ball: Sequence[PixelBox | None], players: dict[str, Sequence[PixelBox | None]]
) -> list[str | None]:
    """Per frame, the player whose box contains or is nearest the ball centre (image space), if the ball is
    within HOLDER_REACH x box height of the box; None when the ball is unseen or nobody is close enough."""
    out: list[str | None] = []
    for f, b in enumerate(ball):
        if b is None:
            out.append(None)
            continue
        cx, cy = centre(b)
        best: tuple[tuple[float, float], str] | None = None
        for pid in sorted(players):
            boxes = players[pid]
            pb = boxes[f] if f < len(boxes) else None
            if pb is None:
                continue
            x1, y1, x2, y2 = pb
            height = max(y2 - y1, 1.0)
            d = math.hypot(max(x1 - cx, 0.0, cx - x2), max(y1 - cy, 0.0, cy - y2))
            if d > settings.HOLDER_REACH * height:
                continue
            key = (d / height, abs(cx - (x1 + x2) / 2) / max(x2 - x1, 1.0))
            if best is None or key < best[0]:
                best = (key, pid)
        out.append(best[1] if best else None)
    return out


@dataclass(frozen=True)
class Segment:
    pid: str
    start: int
    end: int
    held: int  # frames inside [start, end] where this player was the holder


def hold_segments(holder: Sequence[str | None]) -> list[Segment]:
    """Runs of one holder. Gaps of at most HOLD_MERGE_GAP unseen frames stay inside a run; runs held for fewer
    than MIN_HOLD_FRAMES frames are dropped as blips; then neighbouring runs of the same player that only a
    blip or a short gap separated are joined."""
    runs: list[Segment] = []
    for f, pid in enumerate(holder):
        if pid is None:
            continue
        last = runs[-1] if runs else None
        if last and last.pid == pid and f - last.end - 1 <= settings.HOLD_MERGE_GAP:
            runs[-1] = Segment(pid, last.start, f, last.held + 1)
        else:
            runs.append(Segment(pid, f, f, 1))
    kept = [r for r in runs if r.held >= settings.MIN_HOLD_FRAMES]
    joined: list[Segment] = []
    join_gap = settings.HOLD_MERGE_GAP + settings.MIN_HOLD_FRAMES
    for r in kept:
        last = joined[-1] if joined else None
        if last and last.pid == r.pid and r.start - last.end - 1 <= join_gap:
            joined[-1] = Segment(r.pid, last.start, r.end, last.held + r.held)
        else:
            joined.append(r)
    return joined


def _event(kind: str, confidence: str, **fields: Any) -> dict[str, Any]:
    return {"type": kind, **fields, "detect": "heuristic", "confidence": confidence}


def _shot(
    seg: Segment,
    flight: range,
    ball_px: Sequence[PixelBox | None],
    ball_court: Sequence[Point | None],
    players_px: dict[str, Sequence[PixelBox | None]],
) -> dict[str, Any] | None:
    seen = [f for f in flight if ball_court[f] is not None and ball_px[f] is not None]
    if not seen:
        return None
    near = next((f for f in seen if _dist(ball_court[f], RIM) <= settings.SHOT_RIM_FEET), None)  # type: ignore[arg-type]
    if near is not None:
        return _event("shot", "medium", **{"from": seg.pid}, frame=seg.end, end=near, made=None)
    holder_box = next(
        (
            players_px[seg.pid][f]
            for f in range(seg.end, seg.start - 1, -1)
            if players_px[seg.pid][f] is not None
        ),
        None,
    )
    release = next(
        (ball_court[f] for f in range(seg.end, seg.start - 1, -1) if ball_court[f] is not None), None
    )
    if holder_box is None or release is None:
        return None
    top, height = holder_box[1], holder_box[3] - holder_box[1]
    rose = any(centre(ball_px[f])[1] < top - settings.SHOT_RISE * height for f in seen)  # type: ignore[arg-type]
    toward = _dist(ball_court[seen[-1]], RIM) <= _dist(release, RIM) - settings.SHOT_TOWARD_RIM_FEET  # type: ignore[arg-type]
    if rose and toward:
        return _event("shot", "low", **{"from": seg.pid}, frame=seg.end, end=seen[-1], made=None)
    return None


def detect_events(
    holder: Sequence[str | None],
    ball_px: Sequence[PixelBox | None],
    ball_court: Sequence[Point | None],
    players_px: dict[str, Sequence[PixelBox | None]],
    fps: int,
) -> list[dict[str, Any]]:
    """Passes and shots between consecutive holder segments.

    shot: after a segment ends, a seen ball gets within SHOT_RIM_FEET of the rim (medium), or rises above
          the holder's box top and ends at least SHOT_TOWARD_RIM_FEET closer to the rim (low), before anyone
          holds it again (and within PASS_MAX_GAP_SECONDS).
    pass: otherwise, the next segment is a different player of the same team, at most PASS_MAX_GAP_SECONDS
          later. medium when the ball was seen between the two holders, the gap is short and both held it
          for MEDIUM_HOLD_FRAMES; else low (a hand-off with no frame in between is always low).
    """
    segs = hold_segments(holder)
    max_gap = round(settings.PASS_MAX_GAP_SECONDS * fps)
    medium_gap = round(settings.PASS_MEDIUM_GAP_SECONDS * fps)
    n = len(holder)
    events: list[dict[str, Any]] = []
    for k, a in enumerate(segs):
        b = segs[k + 1] if k + 1 < len(segs) else None
        last_flight = min((b.start - 1) if b else n - 1, a.end + max_gap)
        shot = _shot(a, range(a.end + 1, last_flight + 1), ball_px, ball_court, players_px)
        if shot:
            events.append(shot)
            continue
        if b is None or b.pid == a.pid or b.pid[0] != a.pid[0]:
            continue
        gap = b.start - a.end - 1
        if gap > max_gap:
            continue
        in_flight = any(ball_px[f] is not None for f in range(a.end + 1, b.start))
        medium = in_flight and gap <= medium_gap and min(a.held, b.held) >= settings.MEDIUM_HOLD_FRAMES
        events.append(
            _event(
                "pass", "medium" if medium else "low", **{"from": a.pid}, to=b.pid, frame=a.end, end=b.start
            )
        )
    return events
