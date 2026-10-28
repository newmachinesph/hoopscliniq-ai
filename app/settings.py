from __future__ import annotations

import os
from pathlib import Path

VERSION = "0.3.2"

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
RECENT_JOBS = 10
CLIP_CACHE_SECONDS = 3600
JOB_READ_RETRY_SECONDS = 0.5
MODELS_CHECK_TTL = 60.0


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def meta_client(timeout: float = 600.0, max_retries: int = 1):
    from openai import OpenAI

    return OpenAI(
        api_key=env("META_API_KEY"), base_url=META_BASE_URL, timeout=timeout, max_retries=max_retries
    )


def cliniq_client():
    """(client, model) for ClinIQ: Meta by default, Baseten's Muse Spark Open with CLINIQ_PROVIDER=baseten."""
    from openai import OpenAI

    if env("CLINIQ_PROVIDER", "meta").lower() == "baseten":
        client = OpenAI(
            api_key=env("BASETEN_API_KEY"), base_url=BASETEN_BASE_URL, timeout=120.0, max_retries=1
        )
        return client, env("BASETEN_MODEL_ID")
    return meta_client(timeout=120.0), SPARK_MODEL


# Camera motion (OpenCV ORB + RANSAC on background features, see app/camera.py).
CAMERA_SCALE = 0.5  # features are found on a half-size frame
CAMERA_FEATURES = 1000
CAMERA_RATIO = 0.75  # Lowe ratio test
CAMERA_RANSAC_PX = 3.0  # reprojection threshold, work pixels
CAMERA_MIN_INLIERS = 30  # fewer inliers = poor match: the frame falls back to the static calibration
CAMERA_REANCHOR_INLIERS = 80  # overlap with the keyframe is getting thin: this frame becomes the keyframe
CAMERA_KEYFRAME_EVERY = 12  # frames (1 s at 12 fps)
CAMERA_MASK_PAD = 0.1  # player and ball boxes grow by 10 % before they are masked out
# Naming inside POST /calibrate gives up after this long (no retry) and keeps the previous name, so the call
# stays under the demo's 60 s proxy timeout.
CALIBRATE_NAME_TIMEOUT = 40.0
CALIBRATION_MARGIN = 0.5  # a calibration corner may lie this share of the frame outside the image

# Ball, holder and events (heuristics, see README "Ball and events").
SAM_BALL_PROMPT = "basketball"
SAM_BALL_ATTEMPTS = 2
BALL_MEMORY_FRAMES = 6  # the previous ball counts for continuity up to this many frames back
BALL_IDLE_FRAMES = 12  # a followed ball away from every player this long gives way to a ball in play
BALL_MAX_JUMP = 0.06  # continuity jump limit, share of the frame width per frame
HOLDER_REACH = 0.25  # ball centre within this share of the player box height from the box = holder
HOLD_MERGE_GAP = 2  # same holder on both sides of a gap this short = one possession segment
MIN_HOLD_FRAMES = 3  # shorter holder segments are ignored for events
PASS_MAX_GAP_SECONDS = 1.5
PASS_MEDIUM_GAP_SECONDS = 0.75
MEDIUM_HOLD_FRAMES = 6
SHOT_RIM_FEET = 4.0
SHOT_TOWARD_RIM_FEET = 3.0
SHOT_RISE = 0.1  # the ball centre must clear the holder's box top by this share of the box height
