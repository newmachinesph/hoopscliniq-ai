from __future__ import annotations

import os
from pathlib import Path

VERSION = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HOOPSCLINIQ_DATA_DIR", ROOT / "data"))
WORLD_PATH = DATA_DIR / "world.json"
DEFAULT_CALIBRATION_PATH = DATA_DIR / "calibration" / "hazen-union.json"

# Modal Volume mount point inside the containers; jobs live under <FILM_ROOT>/jobs/<jobId>/.
FILM_ROOT = Path(os.environ.get("HOOPSCLINIQ_FILM_ROOT", "/film"))

META_BASE_URL = "https://api.meta.ai/v1"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"
SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"
SAM_PROMPT = "basketball player"
SAM_FILE_TTL_SECONDS = 86400

TARGET_FPS = 12
MAX_WIDTH = 1280
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_UPLOAD_SECONDS = 60
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}

MAX_TRACKS = 10
SMOOTH_WINDOW = 5
MAX_FILL_GAP = 6
TEAM_SAMPLE_FRAMES = 10
TEAM_CROP_TOP = 0.4

MAX_QUESTION = 500
MAX_TOOL_CALLS = 6
MODELS_CHECK_TTL = 60.0


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def meta_client(timeout: float = 600.0):
    from openai import OpenAI

    return OpenAI(api_key=env("META_API_KEY"), base_url=META_BASE_URL, timeout=timeout, max_retries=1)


def cliniq_client():
    """(client, model) for ClinIQ: Meta by default, Baseten's Muse Spark Open with CLINIQ_PROVIDER=baseten."""
    from openai import OpenAI

    if env("CLINIQ_PROVIDER", "meta").lower() == "baseten":
        client = OpenAI(
            api_key=env("BASETEN_API_KEY"), base_url=BASETEN_BASE_URL, timeout=120.0, max_retries=1
        )
        return client, env("BASETEN_MODEL_ID")
    return meta_client(timeout=120.0), SPARK_MODEL
