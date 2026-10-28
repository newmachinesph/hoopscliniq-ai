"""Film jobs: storage on the Modal Volume and the pipeline (frames -> SAM track -> geometry/teams -> name)."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app import settings
from app.geometry import Calibration, court_tracks, load_default_calibration, parse_calibration
from app.naming import suggest_name
from app.playdna import build_overlay, build_play_dna
from app.sam import parse_sam_text, stream_sam, upload_clip
from app.teams import assign_ids, ffmpeg_frame_reader, pick_offense, track_colors, two_means
from app.world import load_world

log = logging.getLogger(__name__)

JOB_ID = re.compile(r"^job-[a-z2-7]{12}$")
PIPELINE = ("frames", "track", "name")
STEP_DEFS = [
    ("upload", "Upload", "Phone or file"),
    ("frames", "Frames", "Modal"),
    ("track", "Track", "SAM 3.1"),
    ("name", "Name", "Muse Spark"),
    ("review", "Review", "Coach confirms"),
    ("cite", "Cite", "ClinIQ"),
]
FRIENDLY = {
    "queued": "The job could not start. Try the upload again.",
    "frames": "We could not read that video. Try an MP4, MOV or WebM under 60 seconds.",
    "track": "Player tracking did not finish. Try the upload again.",
    "name": "Tracking finished, but naming the play failed. Try again.",
}


def new_job_id() -> str:
    return "job-" + base64.b32encode(os.urandom(10)).decode().lower()[:12]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def steps_for(status: str, failed_step: str | None = None) -> list[dict[str, str]]:
    current = failed_step if status == "failed" else status
    pos = PIPELINE.index(current) if current in PIPELINE else (len(PIPELINE) if status == "done" else -1)
    out = []
    for sid, label, engine in STEP_DEFS:
        if sid == "upload":
            st = "done"
        elif sid in PIPELINE:
            i = PIPELINE.index(sid)
            if i < pos:
                st = "done"
            elif i == pos:
                st = "failed" if status == "failed" else "running"
            else:
                st = "queued"
        else:
            st = "planned"
        out.append({"id": sid, "label": label, "engine": engine, "status": st})
    return out


def job_view(
    job: dict[str, Any],
    play_dna: dict[str, Any] | None,
    overlay: dict[str, Any] | None = None,
    clip_url: str | None = None,
) -> dict[str, Any]:
    return {
        "jobId": job["jobId"],
        "status": job["status"],
        "finishedAt": job.get("finishedAt"),
        "steps": steps_for(job["status"], job.get("failedStep")),
        "suggestedName": job.get("suggestedName"),
        "playDna": play_dna,
        "error": job.get("error"),
        # clip.mp4 exists once the frames step has recorded the re-encoded video.
        "clipUrl": clip_url if job.get("video") else None,
        "overlay": overlay,
    }


def job_summary(job: dict[str, Any], play_dna: dict[str, Any] | None) -> dict[str, Any]:
    """One row of `GET /film/jobs`."""
    name = job.get("suggestedName") or {}
    meta = (play_dna or {}).get("meta") or {}
    return {
        "jobId": job["jobId"],
        "finishedAt": job.get("finishedAt"),
        "title": name.get("title"),
        "playersFound": meta.get("playersFound", 0),
    }


class JobStore:
    """Jobs under <root>/jobs/<jobId>/. `commit`/`reload` are the Modal Volume's (no-ops locally)."""

    def __init__(
        self, root: Path, commit: Callable[[], None] | None = None, reload: Callable[[], None] | None = None
    ) -> None:
        self.root = root
        self.commit = commit or (lambda: None)
        self.reload = reload or (lambda: None)

    def dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def _read_json(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
        self.commit()

    def read(self, job_id: str, fresh: bool = True) -> dict | None:
        if fresh:
            self.reload()
        path = self.dir(job_id) / "job.json"
        job = self._read_json(path)
        if job is None and fresh:
            # Seen live: a reload while the job function commits can briefly hide job.json. Look once more.
            time.sleep(settings.JOB_READ_RETRY_SECONDS)
            self.reload()
            job = self._read_json(path)
        return job

    def write(self, job: dict) -> None:
        self._write_json(self.dir(job["jobId"]) / "job.json", job)

    def read_play(self, job_id: str) -> dict | None:
        return self._read_json(self.dir(job_id) / "play.json")

    def write_play(self, job_id: str, play: dict) -> None:
        self._write_json(self.dir(job_id) / "play.json", play)

    def read_overlay(self, job_id: str) -> dict | None:
        return self._read_json(self.dir(job_id) / "overlay.json")

    def write_overlay(self, job_id: str, overlay: dict) -> None:
        self._write_json(self.dir(job_id) / "overlay.json", overlay)

    def clip(self, job_id: str) -> Path | None:
        """Path of clip.mp4, or None when it does not exist (reloads the Volume once before giving up)."""
        path = self.dir(job_id) / "clip.mp4"
        if not path.is_file():
            self.reload()
        return path if path.is_file() else None

    def done_jobs(self, limit: int) -> list[dict]:
        """Finished jobs, newest first (by finishedAt), at most `limit`."""
        self.reload()
        jobs = []
        for path in (self.root / "jobs").glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("unreadable job file %s", path)
                continue
            if job.get("status") == "done":
                jobs.append(job)
        jobs.sort(key=lambda j: j.get("finishedAt") or "", reverse=True)
        return jobs[:limit]


# ---------- ffmpeg ----------


def probe(path: Path, ffprobe: str = "ffprobe", count_frames: bool = False) -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_frames,r_frame_rate:format=duration",
        "-of",
        "json",
    ]
    if count_frames:
        cmd.insert(1, "-count_frames")
    out = subprocess.run([*cmd, str(path)], capture_output=True, text=True, check=True, timeout=120)
    data = json.loads(out.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError("no video stream")
    s = streams[0]
    frames = s.get("nb_read_frames")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "frames": int(frames) if frames and str(frames).isdigit() else None,
        "duration": float(data.get("format", {}).get("duration") or 0.0),
    }


def reencode(src: Path, dst: Path, ffmpeg: str = "ffmpeg") -> None:
    vf = f"fps={settings.TARGET_FPS},scale='min({settings.MAX_WIDTH},iw)':-2"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-an",
            "-vf",
            vf,
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
        ],
        check=True,
        timeout=600,
    )


# ---------- pipeline ----------


def _calibration(job: dict) -> Calibration:
    raw = job.get("calibration")
    if raw is None:
        return load_default_calibration()
    cal = parse_calibration(raw)
    if cal.image_size is None and job.get("video", {}).get("sourceSize"):
        # Points given without a size are in the uploaded video's pixels.
        cal = Calibration(cal.points, tuple(job["video"]["sourceSize"]), cal.source)
    return cal


def _scaled_reader(reader: Callable, size: tuple[int, int]) -> Callable:
    """SAM reports boxes in its own frame size; resize decoded frames to match before cropping."""

    def read(frames: list[int]) -> dict:
        return {f: (im if im.size == size else im.resize(size)) for f, im in reader(frames).items()}

    return read


def build_from_sam(
    job: dict, sam_text: str, clip: Path, ffmpeg: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(play-dna, overlay) from the raw SAM text. Both use the same player ids and frame index."""
    video = job["video"]
    sam = parse_sam_text(sam_text)
    size = (sam.width or video["width"], sam.height or video["height"])
    frames = max(video["frames"] or 0, sam.frame_count)
    tracks = sam.tracks()
    cal = _calibration(job)
    court = court_tracks(tracks, cal, size, frames)
    objs = [t.obj for t in court]
    colors = track_colors(tracks, objs, _scaled_reader(ffmpeg_frame_reader(clip, ffmpeg), size))
    labels = dict(zip(objs, two_means([colors[o] for o in objs]), strict=True))
    by_obj = {t.obj: t for t in court}
    offense = pick_offense(labels, by_obj, frames // 2) if court else 0
    ids = assign_ids(court, labels, offense)
    play = build_play_dna(
        job_id=job["jobId"],
        fps=settings.TARGET_FPS,
        frames=frames,
        court_tracks=court,
        ids=ids,
        calibration_source=cal.source,
        sam_frames=sam.frame_count,
        image_size=size,
    )
    overlay = build_overlay(
        tracks=tracks,
        ids=ids,
        fps=settings.TARGET_FPS,
        frames=frames,
        sam_size=size,
        clip_size=(video["width"], video["height"]),
    )
    return play, overlay


def run_job(
    store: JobStore,
    job_id: str,
    meta_client: Any,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    on_error: Callable[[BaseException], None] | None = None,
) -> dict:
    job = store.read(job_id)
    if job is None:
        raise KeyError(job_id)
    d = store.dir(job_id)
    timings: dict[str, float] = job.setdefault("timings", {})

    def advance(status: str) -> None:
        job["status"] = status
        store.write(job)

    try:
        advance("frames")
        t0 = time.monotonic()
        upload = d / job["upload"]["file"]
        clip = d / "clip.mp4"
        reencode(upload, clip, ffmpeg)
        info = probe(clip, ffprobe, count_frames=True)
        job["video"] = {
            **job.get("video", {}),
            "fps": settings.TARGET_FPS,
            "frames": info["frames"],
            "width": info["width"],
            "height": info["height"],
            "duration": info["duration"],
        }
        upload.unlink(missing_ok=True)
        timings["frames"] = round(time.monotonic() - t0, 2)

        advance("track")
        t0 = time.monotonic()
        file_id = upload_clip(meta_client, clip)
        sam_text = stream_sam(meta_client, file_id, d / "sam.txt")
        store.commit()
        timings["track"] = round(time.monotonic() - t0, 2)
        t0 = time.monotonic()
        play, overlay = build_from_sam(job, sam_text, clip, ffmpeg)
        timings["build"] = round(time.monotonic() - t0, 2)
        play["meta"]["timings"] = dict(timings)
        store.write_overlay(job_id, overlay)
        store.write_play(job_id, play)

        advance("name")
        t0 = time.monotonic()
        suggestion = suggest_name(meta_client, load_world().playbook(), play)
        timings["name"] = round(time.monotonic() - t0, 2)
        job["suggestedName"] = suggestion
        play["name"] = suggestion["title"]
        play["meta"]["timings"] = dict(timings)
        store.write_play(job_id, play)
        job["finishedAt"] = now_iso()
        advance("done")
    except Exception as exc:
        log.exception("job %s failed at %s", job_id, job.get("status"))
        if on_error:
            on_error(exc)
        job["failedStep"] = job.get("status") if job.get("status") in FRIENDLY else "queued"
        job["error"] = FRIENDLY[job["failedStep"]]
        job["finishedAt"] = now_iso()
        advance("failed")
    return job
