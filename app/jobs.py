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
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app import settings
from app.ballpos import ball_court_track
from app.camera import CameraMotion, detect_features, estimate_motion, read_gray_frames
from app.events import centre, detect_events, find_holders, near_player, pick_ball
from app.geometry import (
    apply_h,
    court_tracks,
    homography,
    image_to_court,
    landmark_corners,
    load_default_calibration,
    parse_calibration,
    scale_h,
)
from app.naming import suggest_name
from app.playdna import build_overlay, build_play_dna, to_clip_pixels
from app.sam import Box, SamResult, parse_sam_text, stream_sam, stream_sam_reusing, upload_clip
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


def calibration_view(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """`calibration` of the job view: the corners in clip.mp4 pixels. An upload-time calibration came from the
    coach too, so only the built-in estimate says "estimated"."""
    if not record:
        return None
    source = "estimated" if record.get("source") == "estimated" else "coach"
    return {
        "source": source,
        "frame": record.get("frame", 0),
        "points": record.get("points"),
        "landmarks": record.get("landmarks") or "half",
    }


def job_view(
    job: dict[str, Any],
    play_dna: dict[str, Any] | None,
    overlay: dict[str, Any] | None = None,
    clip_url: str | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = (play_dna or {}).get("meta") or {}
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
        "calibration": calibration_view(calibration),
        "cameraMotion": meta.get("cameraMotion"),
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

    def read_calibration(self, job_id: str) -> dict | None:
        return self._read_json(self.dir(job_id) / "calibration.json")

    def write_calibration(self, job_id: str, record: dict) -> None:
        self._write_json(self.dir(job_id) / "calibration.json", record)

    def read_camera(self, job_id: str) -> CameraMotion | None:
        data = self._read_json(self.dir(job_id) / "camera.json")
        return CameraMotion.from_json(data) if data else None

    def write_camera(self, job_id: str, camera: CameraMotion) -> None:
        self._write_json(self.dir(job_id) / "camera.json", camera.to_json())

    def read_text(self, job_id: str, name: str) -> str | None:
        try:
            return (self.dir(job_id) / name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

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

ColorFn = Callable[[dict[int, dict], list[int], tuple[int, int]], dict[int, tuple[float, float, float]]]


def calibration_record(job: dict) -> dict[str, Any]:
    """The calibration the pipeline starts with, in clip.mp4 pixels at frame 0: the coach's (upload field)
    or the built-in Hazen Union estimate."""
    video = job["video"]
    clip_size = (video["width"], video["height"])
    raw = job.get("calibration")
    cal = load_default_calibration() if raw is None else parse_calibration(raw)
    size = cal.image_size or tuple(video.get("sourceSize") or clip_size)
    sx, sy = clip_size[0] / size[0], clip_size[1] / size[1]
    return {
        "source": cal.source,
        "frame": 0,
        "landmarks": "half",
        "points": [[round(x * sx, 1), round(y * sy, 1)] for x, y in cal.points],
        "imageSize": list(clip_size),
    }


def _scaled_reader(reader: Callable, size: tuple[int, int]) -> Callable:
    """SAM reports boxes in its own frame size; resize decoded frames to match before cropping."""

    def read(frames: list[int]) -> dict:
        return {f: (im if im.size == size else im.resize(size)) for f, im in reader(frames).items()}

    return read


def clip_colors(clip: Path, ffmpeg: str) -> ColorFn:
    """Jersey colors read from clip.mp4 (ffmpeg), for the team split."""

    def colors(tracks: dict[int, dict], objs: list[int], sam_size: tuple[int, int]) -> dict:
        return track_colors(tracks, objs, _scaled_reader(ffmpeg_frame_reader(clip, ffmpeg), sam_size))

    return colors


def build_outputs(
    job: dict,
    sam: SamResult,
    ball_sam: SamResult | None,
    camera: CameraMotion | None,
    calibration: dict[str, Any],
    colors: ColorFn,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, int]]:
    """(play-dna, overlay, team labels) from parsed SAM output, camera motion and a calibration record.

    Positions: court = H_cal . H_(i->calFrame) . p for a box foot point (players) or box centre (ball) in
    clip.mp4 pixels. Team labels come from `job["teamLabels"]` when they cover the kept objects (calibrate
    re-uses them), else from jersey colors.
    """
    video = job["video"]
    clip_size = (video["width"], video["height"])
    sam_size = (sam.width or clip_size[0], sam.height or clip_size[1])
    frames = max(video["frames"] or 0, sam.frame_count)
    h_cal = homography(
        [tuple(p) for p in calibration["points"]], landmark_corners(calibration.get("landmarks"))
    )
    mats = image_to_court(h_cal, camera.to_frame0 if camera else None, int(calibration["frame"]), frames)
    to_clip = scale_h(clip_size[0] / sam_size[0], clip_size[1] / sam_size[1])
    tracks = sam.tracks()
    court = court_tracks(tracks, [m @ to_clip for m in mats], frames)
    objs = [t.obj for t in court]
    stored = {int(k): v for k, v in (job.get("teamLabels") or {}).items()}
    if objs and all(o in stored for o in objs):
        labels = {o: stored[o] for o in objs}
    else:
        rgb = colors(tracks, objs, sam_size)
        labels = dict(zip(objs, two_means([rgb[o] for o in objs]), strict=True))
    by_obj = {t.obj: t for t in court}
    offense = pick_offense(labels, by_obj, frames // 2) if court else 0
    ids = assign_ids(court, labels, offense)

    overlay = build_overlay(
        tracks=tracks, ids=ids, fps=settings.TARGET_FPS, frames=frames, sam_size=sam_size, clip_size=clip_size
    )
    players_px = {pid: p["boxes"] for pid, p in overlay["players"].items()}
    ball_px = _ball_pixels(ball_sam, frames, clip_size, players_px) if ball_sam is not None else None
    if ball_px is not None:
        overlay["ball"] = ball_px
    extra: dict[str, Any] = {
        "calibrationFrame": int(calibration["frame"]),
        "cameraMotion": camera.summary() if camera else None,
        "ballTracked": ball_px is not None,
    }
    ball_track: list[list[float] | None] = []
    holder: list[str | None] = []
    events: list[dict[str, Any]] = []
    if ball_px is not None:
        raw = [None if b is None else apply_h(mats[f], *centre(b)) for f, b in enumerate(ball_px)]
        holder = find_holders(ball_px, players_px)
        events = detect_events(holder, ball_px, raw, players_px, settings.TARGET_FPS)
        ball_track, extra["ballSource"] = ball_court_track(
            holder, {ids[t.obj]: t.positions for t in court}, raw, events, settings.TARGET_FPS
        )
        extra["ballFrames"] = sum(1 for b in ball_px if b is not None)
        extra["holderFrames"] = sum(1 for h in holder if h is not None)
    play = build_play_dna(
        job_id=job["jobId"],
        fps=settings.TARGET_FPS,
        frames=frames,
        court_tracks=court,
        ids=ids,
        calibration_source=calibration["source"],
        sam_frames=sam.frame_count,
        image_size=sam_size,
        ball=ball_track,
        holder=holder,
        events=events,
        extra_meta=extra,
    )
    return play, overlay, labels


def _ball_pixels(
    ball_sam: SamResult, frames: int, clip_size: tuple[int, int], players_px: dict[str, list]
) -> list[list[int] | None]:
    w, h = clip_size
    sx = w / ball_sam.width if ball_sam.width else 1.0
    sy = h / ball_sam.height if ball_sam.height else 1.0

    def px(b: Box) -> list[int]:
        return to_clip_pixels(b, sx, sy, w, h)

    picked = pick_ball(
        ball_sam.frames,
        frames,
        lambda f, b: near_player(players_px, f, px(b)),
        max_jump=settings.BALL_MAX_JUMP * (ball_sam.width or w),
    )
    return [None if b is None else px(b) for b in picked]


def build_from_sam(
    job: dict,
    sam_text: str,
    clip: Path,
    ffmpeg: str,
    ball_text: str | None = None,
    camera: CameraMotion | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(play-dna, overlay) from the raw SAM text. Both use the same player ids and frame index."""
    ball = parse_sam_text(ball_text) if ball_text is not None else None
    play, overlay, _ = build_outputs(
        job, parse_sam_text(sam_text), ball, camera, calibration_record(job), clip_colors(clip, ffmpeg)
    )
    return play, overlay


def _timed(fn: Callable, *args: Any) -> tuple[Any, float]:
    t0 = time.monotonic()
    return fn(*args), round(time.monotonic() - t0, 2)


def _ball_sam(client: Any, file_id: str, clip: Path, out: Path) -> str | None:
    """Second SAM call ("basketball") on the same uploaded file, tried twice (seen live: a one-off
    `server_error` mid-stream). A failure costs the ball, not the job."""
    for attempt in range(1, settings.SAM_BALL_ATTEMPTS + 1):
        try:
            text, file_id = stream_sam_reusing(client, file_id, clip, out, settings.SAM_BALL_PROMPT)
            return text
        except Exception:
            log.warning("ball tracking attempt %d failed", attempt, exc_info=True)
    return None


def _mask_boxes(sam: SamResult, ball: SamResult | None, clip_size: tuple[int, int]) -> Callable[[int], list]:
    """Every SAM player and ball box of a frame, in clip.mp4 pixels (kept out of the camera features)."""

    def scaled(res: SamResult, f: int) -> list[tuple[float, float, float, float]]:
        sx = clip_size[0] / res.width if res.width else 1.0
        sy = clip_size[1] / res.height if res.height else 1.0
        return [(b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy) for b in res.frames.get(f, [])]

    return lambda f: scaled(sam, f) + (scaled(ball, f) if ball else [])


def track_step(
    job: dict, d: Path, meta_client: Any
) -> tuple[SamResult, SamResult | None, CameraMotion, dict]:
    """SAM players + SAM ball (same uploaded file) + camera features run side by side; then camera motion."""
    clip = d / "clip.mp4"
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    file_id = upload_clip(meta_client, clip)
    job["samFile"] = {"id": file_id, "uploadedAt": now_iso()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        players = pool.submit(_timed, stream_sam, meta_client, file_id, d / "sam.txt")
        ball = pool.submit(_timed, _ball_sam, meta_client, file_id, clip, d / "sam-ball.txt")
        feats = pool.submit(_timed, lambda: detect_features(read_gray_frames(clip)))
        sam_text, timings["samPlayers"] = players.result()
        ball_text, timings["samBall"] = ball.result()
        features, timings["cameraFeatures"] = feats.result()
    timings["track"] = round(time.monotonic() - t0, 2)
    sam = parse_sam_text(sam_text)
    ball_sam = parse_sam_text(ball_text) if ball_text is not None else None
    video = job["video"]
    clip_size = (video["width"], video["height"])
    frames = max(video["frames"] or 0, sam.frame_count)
    camera, timings["camera"] = _timed(
        estimate_motion, features, _mask_boxes(sam, ball_sam, clip_size), clip_size, frames
    )
    return sam, ball_sam, camera, timings


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
        source = probe(upload, ffprobe)
        reencode(upload, clip, ffmpeg)
        info = probe(clip, ffprobe, count_frames=True)
        job["video"] = {
            **job.get("video", {}),
            "fps": settings.TARGET_FPS,
            "frames": info["frames"],
            "width": info["width"],
            "height": info["height"],
            "duration": info["duration"],
            "sourceSize": [source["width"], source["height"]],
        }
        upload.unlink(missing_ok=True)
        timings["frames"] = round(time.monotonic() - t0, 2)

        advance("track")
        sam, ball_sam, camera, track_timings = track_step(job, d, meta_client)
        timings.update(track_timings)
        store.write_camera(job_id, camera)
        t0 = time.monotonic()
        calibration = calibration_record(job)
        play, overlay, labels = build_outputs(
            job, sam, ball_sam, camera, calibration, clip_colors(clip, ffmpeg)
        )
        job["teamLabels"] = {str(k): v for k, v in labels.items()}
        timings["build"] = round(time.monotonic() - t0, 2)
        play["meta"]["timings"] = dict(timings)
        store.write_calibration(job_id, calibration)
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


# ---------- coach calibration ----------


def recalibrate(
    store: JobStore,
    job: dict,
    frame: int,
    points: list[list[float]],
    name_play: Callable[[dict], dict] | None,
    ffmpeg: str = "ffmpeg",
    landmarks: str = "half",
) -> dict:
    """Re-project the stored tracks with the coach's 4 points (no new SAM call), rebuild play.json and the
    overlay, re-run naming, save calibration.json. `landmarks` says what the points mark: "half" (half-court
    corners) or "lane" (painted-lane corners). `job` must be done; the caller validates the input."""
    job_id = job["jobId"]
    t0 = time.monotonic()
    sam_text = store.read_text(job_id, "sam.txt")
    if sam_text is None:
        raise FileNotFoundError("sam.txt")
    ball_text = store.read_text(job_id, "sam-ball.txt")
    sam = parse_sam_text(sam_text)
    ball_sam = parse_sam_text(ball_text) if ball_text is not None else None
    camera = store.read_camera(job_id)
    video = job["video"]
    calibration = {
        "source": "coach",
        "frame": frame,
        "landmarks": landmarks,
        "points": [[round(float(x), 1), round(float(y), 1)] for x, y in points],
        "imageSize": [video["width"], video["height"]],
    }
    clip = store.dir(job_id) / "clip.mp4"
    play, overlay, labels = build_outputs(job, sam, ball_sam, camera, calibration, clip_colors(clip, ffmpeg))
    timings = dict(job.get("timings") or {})
    timings["calibrateBuild"] = round(time.monotonic() - t0, 2)
    suggestion = job.get("suggestedName")
    if name_play is not None:
        t1 = time.monotonic()
        # Wall-clock cap: the HTTP client timeout alone did not stop a 104 s call. Past it, keep the old name.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            suggestion = pool.submit(name_play, play).result(timeout=settings.CALIBRATE_NAME_TIMEOUT)
        except Exception:
            log.warning(
                "naming after calibration failed or timed out; keeping the previous name", exc_info=True
            )
        finally:
            pool.shutdown(wait=False)
        timings["calibrateName"] = round(time.monotonic() - t1, 2)
    timings["calibrate"] = round(time.monotonic() - t0, 2)
    if suggestion:
        play["name"] = suggestion["title"]
    play["meta"]["timings"] = timings
    job["teamLabels"] = {str(k): v for k, v in labels.items()}
    job["suggestedName"] = suggestion
    job["calibratedAt"] = now_iso()
    job["timings"] = timings
    store.write_calibration(job_id, calibration)
    store.write_overlay(job_id, overlay)
    store.write_play(job_id, play)
    store.write(job)
    return job
