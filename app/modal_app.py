"""Modal app `hoopscliniq-ai`: the FastAPI web endpoint plus one CPU function per film job.

Deploy from the repo root:  python -m modal deploy app/modal_app.py
Dev server:                 python -m modal serve app/modal_app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
FILM_MOUNT = "/film"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]>=0.115",
        "openai>=1.99",
        "pydantic>=2.8",
        "numpy>=2.0",
        "pillow>=10.4",
        "python-multipart>=0.0.9",
        "sentry-sdk[fastapi]>=2.10",
        "opencv-python-headless>=4.10",
    )
    .env({"HOOPSCLINIQ_FILM_ROOT": FILM_MOUNT, "HOOPSCLINIQ_DATA_DIR": "/root/data"})
    .add_local_dir(ROOT / "data", "/root/data")
    .add_local_python_source("app")
)

app = modal.App("hoopscliniq-ai")
volume = modal.Volume.from_name("hoopscliniq-film", create_if_missing=True)
secrets = [modal.Secret.from_name("hoopscliniq-keys"), modal.Secret.from_name("hoopscliniq-sentry")]


def _init_observability() -> None:
    import os

    import sentry_sdk

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if dsn:
        from sentry_sdk.integrations.openai import OpenAIIntegration

        # The OpenAI integration reports every model-call exception as unhandled, including the naming
        # timeout that POST /calibrate catches on purpose (HOOPSCLINIQ-AI-1). Real failures still reach
        # Sentry: job errors through capture_exception and log.exception, request errors through FastAPI.
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.0,
            send_default_pii=False,
            disabled_integrations=[OpenAIIntegration()],
        )


def _reload() -> None:
    # Volume.reload() refuses while a file is open (a clip being streamed); serve the current view instead.
    try:
        volume.reload()
    except Exception:
        logging.getLogger(__name__).warning("volume reload skipped", exc_info=True)


def _store():
    from app.jobs import JobStore

    return JobStore(Path(FILM_MOUNT), commit=volume.commit, reload=_reload)


@app.function(image=image, volumes={FILM_MOUNT: volume}, secrets=secrets, cpu=4.0, memory=4096, timeout=900)
def process_job(job_id: str) -> str:
    import sentry_sdk

    from app import settings
    from app.jobs import run_job

    _init_observability()
    job = run_job(_store(), job_id, settings.meta_client(), on_error=sentry_sdk.capture_exception)
    return job["status"]


@app.function(image=image, volumes={FILM_MOUNT: volume}, secrets=secrets, timeout=300, min_containers=0)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from app import settings
    from app.api import ModelsCheck, create_app
    from app.cliniq import ask
    from app.jobs import probe
    from app.naming import suggest_name
    from app.world import load_world

    _init_observability()
    world = load_world()
    check = ModelsCheck(lambda: [m.id for m in settings.meta_client(timeout=10.0).models.list().data])

    def answer(question: str) -> dict:
        client, model = settings.cliniq_client()
        return ask(client, model, world, question)

    def start(job_id: str) -> None:
        process_job.spawn(job_id)

    def name_play(play: dict) -> dict:
        client = settings.meta_client(timeout=settings.CALIBRATE_NAME_TIMEOUT, max_retries=0)
        return suggest_name(client, world.playbook(), play, reasoning_effort=settings.CALIBRATE_NAME_EFFORT)

    return create_app(
        store=_store(),
        start_job=start,
        ask=answer,
        models_reachable=check,
        probe_duration=lambda p: probe(p)["duration"],
        name_play=name_play,
    )
