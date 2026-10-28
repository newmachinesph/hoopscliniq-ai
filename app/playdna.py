"""Assemble a play-dna v2 document (the schema of the app's lib/play-dna/*.play.json) from court tracks."""

from __future__ import annotations

from typing import Any

from app.geometry import CourtTrack
from app.sam import Box

UNNAMED = "Unnamed play"

PixelBox = list[int]


def _id_key(pid: str) -> tuple[int, int]:
    return (0 if pid.startswith("o") else 1, int(pid[1:]))


def _team(pid: str) -> str:
    return "offense" if pid.startswith("o") else "defense"


def to_clip_pixels(b: Box, sx: float, sy: float, w: int, h: int) -> PixelBox:
    """SAM box (SAM's frame size) -> integer box in clip.mp4 pixels, clamped to the frame."""
    x1, x2 = (min(max(round(v * sx), 0), w) for v in (b.x1, b.x2))
    y1, y2 = (min(max(round(v * sy), 0), h) for v in (b.y1, b.y2))
    return [x1, y1, x2, y2]


def build_overlay(
    *,
    tracks: dict[int, dict[int, Box]],
    ids: dict[int, str],
    fps: int,
    frames: int,
    sam_size: tuple[int, int],
    clip_size: tuple[int, int],
    ball: list[PixelBox | None] | None = None,
) -> dict[str, Any]:
    """Per-frame SAM boxes of the kept players (and the ball), for drawing on clip.mp4.

    `boxes[i]` is frame i of clip.mp4 (the same index as playDna `tracks.players[id][i]`), or None where SAM
    did not see that player in that frame. Nothing is interpolated here. `ball` is already in clip pixels.
    """
    w, h = clip_size
    sx = w / sam_size[0] if sam_size[0] else 1.0
    sy = h / sam_size[1] if sam_size[1] else 1.0
    players: dict[str, dict[str, Any]] = {}
    for obj, pid in sorted(ids.items(), key=lambda kv: _id_key(kv[1])):
        boxes: list[PixelBox | None] = [None] * frames
        for f, b in tracks.get(obj, {}).items():
            if 0 <= f < frames:
                boxes[f] = to_clip_pixels(b, sx, sy, w, h)
        players[pid] = {"team": _team(pid), "boxes": boxes}
    out: dict[str, Any] = {"fps": fps, "frames": frames, "imageSize": [w, h], "players": players}
    if ball is not None:
        out["ball"] = ball
    return out


def build_play_dna(
    *,
    job_id: str,
    fps: int,
    frames: int,
    court_tracks: list[CourtTrack],
    ids: dict[int, str],
    calibration_source: str,
    sam_frames: int,
    image_size: tuple[int, int],
    name: str = UNNAMED,
    ball: list[list[float] | None] | None = None,
    holder: list[str | None] | None = None,
    events: list[dict[str, Any]] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_id = {ids[t.obj]: t for t in court_tracks}
    order = sorted(by_id, key=_id_key)
    return {
        "version": 2,
        "dictionary": "1.5",
        "name": name,
        "source": {"clip": job_id, "authored": False, "fps": fps, "frames": frames},
        "court": {
            "units": "ft",
            "x": "across, 0-50",
            "y": "out from the attacked baseline",
            "rim": [25, 5.25],
            "dimensions": "NFHS",
        },
        "players": {pid: {"team": _team(pid)} for pid in order},
        "tracks": {
            "players": {pid: [[x, y] for x, y in by_id[pid].positions] for pid in order},
            "ball": ball or [],
            "holder": holder or [],
            "areas": [],
        },
        "events": events or [],
        "actions": [],
        "moments": [{"frame": 0, "label": "Start"}],
        "fixes": [],
        "meta": {
            "playersFound": len(court_tracks),
            "calibration": calibration_source,
            "samFrames": sam_frames,
            "imageSize": list(image_size),
            # Share of clip frames where SAM actually saw each player; the rest is interpolated or held.
            "coverage": {pid: round(by_id[pid].observed / frames, 3) if frames else 0.0 for pid in order},
            "samObjectIds": {pid: by_id[pid].obj for pid in order},
            **(extra_meta or {}),
        },
    }
