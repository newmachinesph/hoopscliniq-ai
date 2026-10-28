"""FastAPI routes: health, film upload, job status, job list, clip and calibration, ClinIQ ask."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import settings
from app.geometry import check_quad, parse_calibration
from app.jobs import JOB_ID, JobStore, job_summary, job_view, new_job_id, now_iso, recalibrate
from app.models import AskRequest, AskResponse, JobList, JobView

log = logging.getLogger(__name__)

HEADERS = {"Cache-Control": "no-store", "x-demo": "live"}
CLIP_HEADERS = {
    "Cache-Control": f"public, max-age={settings.CLIP_CACHE_SECONDS}",
    "Access-Control-Allow-Origin": "*",
}
CHUNK = 1024 * 1024


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers=HEADERS)


class ModelsCheck:
    """`modelsReachable` for /health: both models listed by the Meta API, cached for 60 s."""

    def __init__(self, list_models: Callable[[], list[str]]) -> None:
        self.list_models = list_models
        self.at = 0.0
        self.value = False

    def __call__(self) -> bool:
        if time.monotonic() - self.at < settings.MODELS_CHECK_TTL and self.at:
            return self.value
        try:
            ids = set(self.list_models())
            self.value = {settings.SAM_MODEL, settings.SPARK_MODEL} <= ids
        except Exception:
            log.warning("models check failed", exc_info=True)
            self.value = False
        self.at = time.monotonic()
        return self.value


def create_app(
    *,
    store: JobStore,
    start_job: Callable[[str], None],
    ask: Callable[[str], dict[str, Any]],
    models_reachable: Callable[[], bool],
    probe_duration: Callable[[Path], float],
    name_play: Callable[[dict], dict] | None = None,
) -> FastAPI:
    api = FastAPI(title="hoopscliniq-ai", version=settings.VERSION)

    @api.middleware("http")
    async def demo_headers(request: Request, call_next):
        response = await call_next(request)
        # Routes that set their own caching (the clip) keep it; everything else is no-store.
        for key, value in HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    @api.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return error(400, f"Bad request: {where or 'body'} {first.get('msg', 'is invalid')}".strip())

    @api.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error(exc.status_code, str(exc.detail))

    @api.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return error(500, "Something went wrong on our side. Try again.")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": settings.VERSION, "modelsReachable": models_reachable()}

    @api.post("/film/upload", status_code=202)
    def upload(file: UploadFile = File(...), calibration: str | None = Form(None)) -> JSONResponse:  # noqa: B008
        ext = Path(file.filename or "").suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            return error(400, "Upload an MP4, MOV or WebM video.")
        cal = None
        if calibration:
            try:
                cal_raw = json.loads(calibration)
                parse_calibration(cal_raw)
                cal = cal_raw
            except (ValueError, TypeError) as exc:
                return error(400, f"Bad calibration: {exc}")
        job_id = new_job_id()
        d = store.dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"upload{ext}"
        size = 0
        with dest.open("wb") as out:
            while chunk := file.file.read(CHUNK):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(d, ignore_errors=True)
                    return error(413, "That video is over 200 MB.")
                out.write(chunk)
        try:
            duration = probe_duration(dest)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            return error(400, "We could not read that video.")
        if duration > settings.MAX_UPLOAD_SECONDS + 0.5:
            shutil.rmtree(d, ignore_errors=True)
            return error(
                400, f"That video is {duration:.0f} s long; the limit is {settings.MAX_UPLOAD_SECONDS} s."
            )
        job = {
            "jobId": job_id,
            "status": "queued",
            "createdAt": now_iso(),
            "finishedAt": None,
            "suggestedName": None,
            "error": None,
            "upload": {"file": dest.name, "name": file.filename, "bytes": size, "duration": duration},
            "calibration": cal,
        }
        store.write(job)
        start_job(job_id)
        return JSONResponse(
            {"jobId": job_id, "status": "queued", "statusUrl": f"/film/jobs/{job_id}"}, status_code=202
        )

    @api.get("/film/jobs", response_model=JobList)
    def job_list() -> Any:
        jobs = store.done_jobs(settings.RECENT_JOBS)
        return {"jobs": [job_summary(job, store.read_play(job["jobId"])) for job in jobs]}

    @api.get("/film/jobs/{job_id}", response_model=JobView)
    def job_status(job_id: str, request: Request) -> Any:
        if not JOB_ID.match(job_id):
            return error(400, "Malformed job id.")
        job = store.read(job_id)
        if job is None:
            return error(404, "Unknown job.")
        return _view(job, request)

    def _view(job: dict, request: Request) -> dict[str, Any]:
        job_id = job["jobId"]
        clip_url = str(request.url_for("job_clip", job_id=job_id))
        return job_view(
            job, store.read_play(job_id), store.read_overlay(job_id), clip_url, store.read_calibration(job_id)
        )

    @api.post("/film/jobs/{job_id}/calibrate", response_model=JobView)
    async def calibrate(job_id: str, request: Request) -> Any:
        if not JOB_ID.match(job_id):
            return error(400, "Malformed job id.")
        job = await run_in_threadpool(store.read, job_id)
        if job is None:
            return error(404, "Unknown job.")
        if job.get("status") != "done":
            return error(409, "This job is not finished yet. Calibrate once the analysis is done.")
        try:
            body = await request.json()
        except ValueError:
            body = None
        try:
            frame, points = parse_calibrate_body(body, job["video"])
        except ValueError as exc:
            return error(400, f"Bad calibration: {exc}")
        done = await run_in_threadpool(recalibrate, store, job, frame, points, name_play)
        return _view(done, request)

    @api.api_route("/film/jobs/{job_id}/clip", methods=["GET", "HEAD"], name="job_clip")
    def job_clip(job_id: str) -> Any:
        if not JOB_ID.match(job_id):
            return error(400, "Malformed job id.")
        path = store.clip(job_id)
        if path is None:
            return error(404, "No clip for this job.")
        # FileResponse answers Range requests with 206 and Content-Range, so browsers can seek.
        return FileResponse(path, media_type="video/mp4", headers=CLIP_HEADERS)

    @api.post("/cliniq/ask", response_model=AskResponse)
    async def cliniq_ask(request: Request) -> Any:
        try:
            body = AskRequest.model_validate(await request.json())
        except (ValueError, ValidationError):
            return error(
                400, f'Send JSON: {{"question": "..."}} with at most {settings.MAX_QUESTION} characters.'
            )
        return await run_in_threadpool(ask, body.question)

    return api


def parse_calibrate_body(body: Any, video: dict) -> tuple[int, list[list[float]]]:
    """`{ frame, points: [[x, y] x4] }` in clip.mp4 pixels -> (frame, points). Raises ValueError."""
    if not isinstance(body, dict):
        raise ValueError('send JSON {"frame": int, "points": [[x, y], [x, y], [x, y], [x, y]]}')
    frame = body.get("frame")
    if not isinstance(frame, int) or isinstance(frame, bool):
        raise ValueError("frame must be an integer")
    if not 0 <= frame < int(video.get("frames") or 0):
        raise ValueError(f"frame must be between 0 and {int(video.get('frames') or 0) - 1}")
    raw = body.get("points")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("send exactly 4 points")
    points = []
    for p in raw:
        if not (
            isinstance(p, list)
            and len(p) == 2
            and all(isinstance(v, int | float) and not isinstance(v, bool) for v in p)
        ):
            raise ValueError("each point must be [x, y]")
        points.append([float(p[0]), float(p[1])])
    check_quad([(x, y) for x, y in points], (int(video["width"]), int(video["height"])))
    return frame, points
