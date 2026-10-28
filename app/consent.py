"""Consent mask: render a clip with chosen tracked players hidden, from their SAM 3.1 boxes."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import settings
from app.jobs import HIDDEN_CLIP, JobStore, now_iso

log = logging.getLogger(__name__)

Box = list[int]


def box_for_frame(boxes: list[Box | None], f: int, hold: int) -> Box | None:
    """The player's box in frame `f`, else the nearest box within `hold` frames (the earlier one on a tie)."""
    for d in range(hold + 1):
        for g in (f - d, f + d):
            if 0 <= g < len(boxes) and boxes[g] is not None:
                return boxes[g]
    return None


def padded(box: Box, width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    """The box grown by `pad` of its width and height on each side, clamped to the image."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        max(0, round(x1 - px)),
        max(0, round(y1 - py)),
        min(width, round(x2 + px)),
        min(height, round(y2 + py)),
    )


def obscure(frame: np.ndarray, region: tuple[int, int, int, int]) -> None:
    """Pixelate the region heavily (1/16 size, back up with nearest neighbour), then darken it. In place."""
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        return
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    small = cv2.resize(
        crop,
        (max(1, w // settings.HIDE_PIXEL_DIV), max(1, h // settings.HIDE_PIXEL_DIV)),
        interpolation=cv2.INTER_AREA,
    )
    blocks = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = (blocks * (1.0 - settings.HIDE_DARKEN)).astype(np.uint8)


def ffmpeg_command(ffmpeg: str, size: tuple[int, int], fps: float, dst: Path) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        f"{fps:.6g}",
        "-i",
        "-",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]


def render_hidden(
    src: Path, dst: Path, overlay: dict[str, Any], player_ids: list[str], ffmpeg: str = "ffmpeg"
) -> dict[str, Any]:
    """Write `dst` (H.264, same size and fps as `src`) with the players hidden in every frame. Raw frames go
    to ffmpeg through a pipe. Returns `{ frames, framesMissed: { id: n } }`: a frame is missed for a player
    with no box within HIDE_HOLD_FRAMES, so the player may still show there."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot open {src.name}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or float(overlay["fps"])
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    boxes = {pid: overlay["players"][pid]["boxes"] for pid in player_ids}
    missed = dict.fromkeys(player_ids, 0)
    frames = 0
    broken = False
    proc = subprocess.Popen(
        ffmpeg_command(ffmpeg, (width, height), fps, tmp), stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for pid in player_ids:
                box = box_for_frame(boxes[pid], frames, settings.HIDE_HOLD_FRAMES)
                if box is None:
                    missed[pid] += 1
                else:
                    obscure(frame, padded(box, width, height, settings.HIDE_PAD))
            try:
                proc.stdin.write(frame.tobytes())
            except OSError:  # ffmpeg exited early; its stderr says why
                broken = True
                break
            frames += 1
        with contextlib.suppress(OSError):
            proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0 or broken or frames == 0:
            raise RuntimeError(f"ffmpeg failed after {frames} frames: {err.strip()[-300:]}")
    except BaseException:
        proc.kill()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
    os.replace(tmp, dst)
    return {"frames": frames, "framesMissed": missed}


def set_hidden(
    store: JobStore, job_id: str, overlay: dict[str, Any], player_ids: list[str], ffmpeg: str = "ffmpeg"
) -> dict[str, Any] | None:
    """Hide these players in the job's clip (renders clip-hidden.mp4, saves hidden.json), return the record.
    An empty list removes the hidden clip so everyone shows again and returns None."""
    if not player_ids:
        store.clear_hidden(job_id)
        return None
    d = store.dir(job_id)
    result = render_hidden(d / "clip.mp4", d / HIDDEN_CLIP, overlay, player_ids, ffmpeg)
    record = {"playerIds": player_ids, **result, "at": now_iso()}
    store.write_hidden(job_id, record)  # commits the volume with the clip
    log.info("job %s: hid %s on %d frames", job_id, player_ids, result["frames"])
    return record
