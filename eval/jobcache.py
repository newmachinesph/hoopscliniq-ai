"""Eval jobs and their local cache (standard library only: the label server imports this).

`eval/jobs.json` maps each eval clip to the backend job that processed it:

    { "backend": "https://...modal.run",
      "clips": { "<clip>.mp4": { "jobId": "job-...", "wallSeconds": 95.7, "ranAt": "2026-10-28" } } }

`eval/cache/<clip stem>/` (gitignored) holds what was fetched once from the backend: `job.json` (the job view,
with the overlay and play.json), `clip.mp4` (the 12 fps re-encode the boxes refer to) and `frames/<idx>.jpg`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

EVAL = Path(__file__).resolve().parent
JOBS_FILE = EVAL / "jobs.json"
CACHE = EVAL / "cache"
LABELS = EVAL / "labels"
FRAMES_PER_CLIP = 8
FRAME_STEP_SECONDS = 2.5
USER_AGENT = "hoopscliniq-eval/0.1"
TIMEOUT_SECONDS = 120


def load_jobs(path: Path | None = None) -> dict[str, Any]:
    path = path or JOBS_FILE
    if not path.exists():
        return {"backend": "", "clips": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def backend_url(jobs: dict[str, Any], override: str | None = None) -> str:
    return (override or os.environ.get("HOOPSCLINIQ_API_URL") or jobs.get("backend") or "").rstrip("/")


def label_path(clip: str) -> Path:
    return LABELS / f"{Path(clip).stem}.json"


def eval_frames(fps: float, frames: int) -> list[int]:
    """The frames to label: one every 2.5 s, mid-window (1.25 s, 3.75 s, ...), at most 8. Starting at
    1.25 s keeps out the first frames, where SAM is still picking up players."""
    step = max(1, round(FRAME_STEP_SECONDS * fps))
    return list(range(step // 2, frames, step))[:FRAMES_PER_CLIP]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
        return r.read()


def cached_job(clip: str, job_id: str, base: str, refresh: bool = False) -> dict[str, Any]:
    """The job view (`GET /film/jobs/<id>`), fetched once and then read from the cache."""
    path = CACHE / Path(clip).stem / "job.json"
    if refresh or not path.exists():
        if not base:
            raise RuntimeError(f"{clip}: no cached job.json and no backend URL (set HOOPSCLINIQ_API_URL)")
        job = json.loads(_get(f"{base}/film/jobs/{job_id}"))
        if job.get("status") != "done":
            raise RuntimeError(f"{clip}: job {job_id} is {job.get('status')}, not done")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def cached_clip(clip: str, job_id: str, base: str) -> Path:
    """clip.mp4 of the job, downloaded once."""
    path = CACHE / Path(clip).stem / "clip.mp4"
    if not path.exists():
        if not base:
            raise RuntimeError(f"{clip}: no cached clip.mp4 and no backend URL (set HOOPSCLINIQ_API_URL)")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(_get(f"{base}/film/jobs/{job_id}/clip"))
        tmp.replace(path)
    return path


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # optional: the dev extra installs it

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg or put ffmpeg on PATH") from None
        return exe


def cached_frames(clip_mp4: Path, indices: list[int]) -> dict[int, Path]:
    """JPEGs of the given clip.mp4 frame indices, extracted once with ffmpeg."""
    out = clip_mp4.parent / "frames"
    paths = {i: out / f"{i}.jpg" for i in indices}
    missing = [i for i, p in paths.items() if not p.exists()]
    if missing:
        out.mkdir(parents=True, exist_ok=True)
        select = "+".join(f"eq(n\\,{i})" for i in missing)
        tmp = out / "tmp"
        tmp.mkdir(exist_ok=True)
        subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip_mp4),
             "-vf", f"select={select}", "-fps_mode", "passthrough", "-pix_fmt", "yuvj420p", "-q:v", "2",
             str(tmp / "%03d.jpg")],
            check=True,
        )  # fmt: skip
        for k, i in enumerate(missing, start=1):
            (tmp / f"{k:03d}.jpg").replace(paths[i])
        shutil.rmtree(tmp, ignore_errors=True)
    return paths
