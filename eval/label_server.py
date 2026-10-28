"""Local labelling helper for the eval clips (standard library only).

    python -m eval.label_server --clips ../clips/eval [--port 8765]

For every clip in the folder that has a job in eval/jobs.json, it fetches the job view and clip.mp4 from the
backend once, cuts 8 frames (one every 2.5 s) to JPEG under eval/cache/, and then serves everything from the
cache at http://127.0.0.1:8765. Each frame starts with the job's overlay player boxes; the page saves
eval/labels/<clip>.json after every change:

    { "clip", "jobId", "fps", "imageSize",
      "frames": { "<frameIdx>": [[x1, y1, x2, y2], ...] }, "done": [frameIdx, ...] }

Boxes are clip.mp4 pixels. Only frames in `done` are scored by `python -m eval.run`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from eval import jobcache

UI = Path(__file__).with_name("label_ui.html")
MIN_BOX = 2  # px; smaller boxes are dropped on save
MAX_BODY = 1_000_000
STEM = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass
class Clip:
    name: str
    job_id: str
    fps: int
    size: tuple[int, int]
    frames: list[int]
    prefill: dict[int, list[list[int]]]
    jpgs: dict[int, Path] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return Path(self.name).stem


def prefill_boxes(overlay: dict[str, Any], frame: int) -> list[list[int]]:
    """The overlay player boxes on one clip.mp4 frame."""
    boxes = (p["boxes"][frame] for p in overlay["players"].values() if frame < len(p["boxes"]))
    return [list(map(int, b)) for b in boxes if b]


def prepare(clips_dir: Path, base: str) -> dict[str, Clip]:
    jobs = jobcache.load_jobs()["clips"]
    clips: dict[str, Clip] = {}
    for mp4 in sorted(clips_dir.glob("*.mp4")):
        entry = jobs.get(mp4.name)
        if not entry:
            print(f"skip {mp4.name}: no job in eval/jobs.json", file=sys.stderr)
            continue
        job = jobcache.cached_job(mp4.name, entry["jobId"], base)
        overlay = job["overlay"]
        indices = jobcache.eval_frames(overlay["fps"], overlay["frames"])
        jpgs = jobcache.cached_frames(jobcache.cached_clip(mp4.name, entry["jobId"], base), indices)
        clip = Clip(
            name=mp4.name,
            job_id=entry["jobId"],
            fps=overlay["fps"],
            size=(overlay["imageSize"][0], overlay["imageSize"][1]),
            frames=indices,
            prefill={i: prefill_boxes(overlay, i) for i in indices},
            jpgs=jpgs,
        )
        clips[clip.stem] = clip
        print(f"ready {mp4.name} ({entry['jobId']}, frames {indices})", file=sys.stderr)
    return clips


def load_state(clip: Clip) -> dict[str, Any]:
    """Saved labels where they exist, the overlay boxes elsewhere."""
    path = jobcache.label_path(clip.name)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    frames = saved.get("frames") or {}
    return {
        "frames": {str(i): frames.get(str(i), clip.prefill[i]) for i in clip.frames},
        "done": [i for i in saved.get("done", []) if i in clip.frames],
    }


def clean_box(raw: Any, size: tuple[int, int]) -> list[int] | None:
    if not (isinstance(raw, list) and len(raw) == 4):
        raise ValueError("each box must be [x1, y1, x2, y2]")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in raw):
        raise ValueError("box values must be numbers")
    w, h = size
    x1, x2 = sorted((min(max(round(raw[0]), 0), w), min(max(round(raw[2]), 0), w)))
    y1, y2 = sorted((min(max(round(raw[1]), 0), h), min(max(round(raw[3]), 0), h)))
    if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
        return None
    return [x1, y1, x2, y2]


def labels_from_body(clip: Clip, body: Any) -> dict[str, Any]:
    """Validate `{ frames, done }` from the page and build the label file. Raises ValueError."""
    if not isinstance(body, dict) or not isinstance(body.get("frames"), dict):
        raise ValueError("send {frames: {frameIdx: [[x1, y1, x2, y2], ...]}, done: [frameIdx]}")
    frames: dict[str, list[list[int]]] = {}
    for key, boxes in body["frames"].items():
        if not (isinstance(key, str) and key.isdigit() and int(key) in clip.frames):
            raise ValueError(f"frame {key} is not one of {clip.frames}")
        if not isinstance(boxes, list):
            raise ValueError(f"frame {key}: boxes must be a list")
        frames[key] = [b for b in (clean_box(raw, clip.size) for raw in boxes) if b]
    done = body.get("done", [])
    if not isinstance(done, list) or not all(isinstance(i, int) and i in clip.frames for i in done):
        raise ValueError(f"done must list frames from {clip.frames}")
    return {
        "clip": clip.name,
        "jobId": clip.job_id,
        "fps": clip.fps,
        "imageSize": list(clip.size),
        "frames": dict(sorted(frames.items(), key=lambda kv: int(kv[0]))),
        "done": sorted(set(done)),
    }


class Labeller:
    def __init__(self, clips: dict[str, Clip]) -> None:
        self.clips = clips
        self.lock = threading.Lock()

    def summary(self) -> dict[str, Any]:
        rows = []
        for stem, clip in self.clips.items():
            rows.append(
                {
                    "stem": stem,
                    "clip": clip.name,
                    "jobId": clip.job_id,
                    "fps": clip.fps,
                    "imageSize": list(clip.size),
                    "frames": clip.frames,
                    "done": len(load_state(clip)["done"]),
                }
            )
        return {"clips": rows, "total": sum(len(c.frames) for c in self.clips.values())}

    def save(self, clip: Clip, body: Any) -> dict[str, Any]:
        labels = labels_from_body(clip, body)
        path = jobcache.label_path(clip.name)
        with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(labels, indent=1) + "\n", encoding="utf-8")
            tmp.replace(path)
        return {"ok": True, "done": len(labels["done"])}


def make_handler(app: Labeller) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet: one line per error only
            if len(args) > 1 and str(args[1]).startswith(("4", "5")):
                sys.stderr.write(f"{self.command} {self.path} -> {args[1]}\n")

        def send(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def json(self, status: int, data: Any) -> None:
            self.send(status, json.dumps(data).encode(), "application/json")

        def clip(self, stem: str) -> Clip | None:
            return app.clips.get(stem) if STEM.match(stem) else None

        def do_GET(self) -> None:
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not parts:
                self.send(200, UI.read_bytes(), "text/html; charset=utf-8")
            elif parts == ["api", "clips"]:
                self.json(200, app.summary())
            elif len(parts) == 3 and parts[:2] == ["api", "labels"] and (clip := self.clip(parts[2])):
                self.json(200, load_state(clip))
            elif len(parts) == 3 and parts[0] == "frames" and (clip := self.clip(parts[1])):
                idx = parts[2].removesuffix(".jpg")
                jpg = clip.jpgs.get(int(idx)) if idx.isdigit() else None
                if jpg is None:
                    self.json(404, {"error": "no such frame"})
                else:
                    self.send(200, jpg.read_bytes(), "image/jpeg", "max-age=3600")
            else:
                self.json(404, {"error": "not found"})

        def do_POST(self) -> None:
            parts = [p for p in self.path.split("/") if p]
            clip = self.clip(parts[2]) if len(parts) == 3 and parts[:2] == ["api", "labels"] else None
            if clip is None:
                self.json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self.json(413, {"error": "body too large"})
                return
            try:
                self.json(200, app.save(clip, json.loads(self.rfile.read(length) or b"null")))
            except ValueError as exc:  # json.JSONDecodeError is a ValueError
                self.json(400, {"error": str(exc)})

    return Handler


def main() -> int:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--clips", type=Path, required=True, help="folder with the eval clips (*.mp4)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--url", help="backend URL for the one-time fetch (default: eval/jobs.json)")
    args = parser.parse_args()
    if not args.clips.is_dir():
        sys.exit(f"{args.clips} is not a folder")
    base = jobcache.backend_url(jobcache.load_jobs(), args.url)
    clips = prepare(args.clips, base)
    if not clips:
        sys.exit(f"no clip in {args.clips} has a job in eval/jobs.json")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Labeller(clips)))
    print(f"labelling {len(clips)} clips at http://127.0.0.1:{args.port}  (Ctrl+C to stop)", file=sys.stderr)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
