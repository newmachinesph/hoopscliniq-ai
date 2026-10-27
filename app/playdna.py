"""Assemble a play-dna v2 document (the schema of the app's lib/play-dna/*.play.json) from court tracks."""

from __future__ import annotations

from typing import Any

from app.geometry import CourtTrack

UNNAMED = "Unnamed play"


def _id_key(pid: str) -> tuple[int, int]:
    return (0 if pid.startswith("o") else 1, int(pid[1:]))


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
        "players": {pid: {"team": "offense" if pid.startswith("o") else "defense"} for pid in order},
        "tracks": {
            "players": {pid: [[x, y] for x, y in by_id[pid].positions] for pid in order},
            "ball": [],
            "holder": [],
            "areas": [],
        },
        "events": [],
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
        },
    }
