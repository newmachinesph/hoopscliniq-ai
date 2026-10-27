"""Split tracks into two teams by jersey color, then pick the offense by court position."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from app import settings
from app.geometry import CourtTrack
from app.sam import Box

RGB = tuple[float, float, float]
FrameReader = Callable[[list[int]], dict[int, Image.Image]]


def jersey_box(b: Box, top: float = settings.TEAM_CROP_TOP) -> tuple[int, int, int, int]:
    """Top part of a player box (shirt, not shorts or floor), as a PIL crop box."""
    height = b.y2 - b.y1
    return (int(b.x1), int(b.y1), int(b.x2), int(b.y1 + max(1.0, height * top)))


def mean_rgb(image: Image.Image, crop: tuple[int, int, int, int]) -> RGB | None:
    x1, y1, x2, y2 = crop
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    arr = np.asarray(image.crop((x1, y1, x2, y2)).convert("RGB"), dtype=float)
    r, g, b = arr.reshape(-1, 3).mean(axis=0)
    return (float(r), float(g), float(b))


def two_means(colors: list[RGB], iterations: int = 20) -> list[int]:
    """Deterministic 2-means: seeds are the two colors farthest apart. Returns a label (0/1) per color."""
    if len(colors) < 2:
        return [0] * len(colors)
    pts = np.asarray(colors, dtype=float)
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(dists)), dists.shape)
    centers = np.stack([pts[i], pts[j]])
    labels = np.zeros(len(pts), dtype=int)
    for _ in range(iterations):
        labels = np.argmin(np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2), axis=1)
        new = np.stack([pts[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in (0, 1)])
        if np.allclose(new, centers):
            break
        centers = new
    return [int(v) for v in labels]


def track_colors(
    tracks: dict[int, dict[int, Box]],
    objs: list[int],
    read_frames: FrameReader,
    sample: int = settings.TEAM_SAMPLE_FRAMES,
) -> dict[int, RGB]:
    """Mean jersey color of each object over its first `sample` frames."""
    wanted = {obj: sorted(tracks[obj])[:sample] for obj in objs}
    images = read_frames(sorted({f for frames in wanted.values() for f in frames}))
    out: dict[int, RGB] = {}
    for obj, frames in wanted.items():
        samples = [
            c
            for f in frames
            if f in images and (c := mean_rgb(images[f], jersey_box(tracks[obj][f]))) is not None
        ]
        if samples:
            out[obj] = tuple(float(v) for v in np.mean(np.asarray(samples), axis=0))  # type: ignore[assignment]
        else:
            out[obj] = (0.0, 0.0, 0.0)
    return out


def pick_offense(labels: dict[int, int], court: dict[int, CourtTrack], frame: int) -> int:
    """The team with more players in the attacked half (y < 47) at `frame`; ties go to the team farther out,
    since defenders sit between their man and the rim."""
    counts = {0: 0, 1: 0}
    depth = {0: [], 1: []}
    for obj, team in labels.items():
        y = court[obj].positions[frame][1]
        depth[team].append(y)
        if y < 47:
            counts[team] += 1
    if counts[0] != counts[1]:
        return 0 if counts[0] > counts[1] else 1
    mean = {k: (sum(v) / len(v) if v else 0.0) for k, v in depth.items()}
    return 0 if mean[0] >= mean[1] else 1


def assign_ids(court_tracks: list[CourtTrack], labels: dict[int, int], offense: int) -> dict[int, str]:
    """o1.., d1.. by track length (the list is already longest first)."""
    ids: dict[int, str] = {}
    n = {"o": 0, "d": 0}
    for t in court_tracks:
        side = "o" if labels.get(t.obj, offense) == offense else "d"
        n[side] += 1
        ids[t.obj] = f"{side}{n[side]}"
    return ids


def ffmpeg_frame_reader(clip: Path, ffmpeg: str = "ffmpeg") -> FrameReader:
    """Read selected frames of a clip as PIL images (ffmpeg -> PNG -> PIL)."""

    def read(frames: list[int]) -> dict[int, Image.Image]:
        if not frames:
            return {}
        select = "+".join(f"eq(n\\,{f})" for f in frames)
        with tempfile.TemporaryDirectory() as tmp:
            pattern = str(Path(tmp) / "f%05d.png")
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(clip),
                    "-vf",
                    f"select={select}",
                    "-fps_mode",
                    "passthrough",
                    pattern,
                ],
                check=True,
            )
            files = sorted(Path(tmp).glob("f*.png"))
            out = {}
            for f, path in zip(frames, files, strict=False):
                with Image.open(path) as im:
                    out[f] = im.convert("RGB")
            return out

    return read
