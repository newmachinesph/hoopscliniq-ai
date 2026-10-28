"""Eval harness: the pipeline's player boxes vs checked reference labels.

    python -m eval.run                       # every clip in eval/jobs.json that has labelled frames
    python -m eval.run --clip ../clips/<file>.mp4 [--labels eval/labels/<clip>.json] [--stream raw-sse.txt]

Default mode reads each job's stored outputs (the job view: overlay boxes and play.json meta, fetched
once into eval/cache/) and the labels in eval/labels/<clip>.json. Only frames listed in `done` are scored (a
label file without `done` scores all its frames). It reports, per clip and overall: player box precision and
recall at IoU >= 0.5 (greedy one-to-one matching), players found vs labelled per frame, camera fallback
frames, ball frames, pipeline wall time and a SAM cost estimate, and writes
eval/results/summary.json and summary.md. With no labelled frames it prints "nothing measured" and exits 0.

`--clip` mode scores a raw SAM stream (or a fresh SAM call at 12 fps, needs META_API_KEY) against one label
file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.sam import Box, SamResult, deltas_from_sse, parse_sam_text
from eval import jobcache

ROOT = Path(__file__).resolve().parent.parent
IOU_MATCH = 0.5
SAM_USD_PER_1000_FRAMES = 0.20  # Meta Model API price for sam-3.1
NOTHING = "nothing measured"
Rect = tuple[float, float, float, float]


def iou(a: Rect, b: Rect) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_frame(pred: list[Rect], truth: list[Rect], threshold: float = IOU_MATCH) -> int:
    """Greedy one-to-one matching by IoU, best pairs first. Returns true positives."""
    pairs = sorted(((iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)), reverse=True)
    used_p: set[int] = set()
    used_t: set[int] = set()
    tp = 0
    for score_, i, j in pairs:
        if score_ < threshold:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        tp += 1
    return tp


def precision_recall(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    return {
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
    }


def score_frames(pred: dict[int, list[Rect]], truth: dict[int, list[Rect]]) -> dict[str, Any]:
    """Counts over the frames in `truth` (the labelled ones). A frame with no predictions is all misses."""
    tp = fp = fn = found = labelled = 0
    for f, boxes in truth.items():
        guesses = pred.get(f, [])
        hit = match_frame(guesses, boxes)
        tp += hit
        fp += len(guesses) - hit
        fn += len(boxes) - hit
        found += len(guesses)
        labelled += len(boxes)
    n = len(truth)
    return {
        "labelledFrames": n,
        "truePositives": tp,
        "falsePositives": fp,
        "falseNegatives": fn,
        **precision_recall(tp, fp, fn),
        "playersFound": found,
        "playersLabelled": labelled,
        "playersFoundPerFrame": round(found / n, 2) if n else None,
        "playersLabelledPerFrame": round(labelled / n, 2) if n else None,
    }


def overlay_boxes(overlay: dict[str, Any]) -> dict[int, list[Rect]]:
    """Player boxes per clip.mp4 frame from the job overlay (the kept tracks, as the demo draws them)."""
    out: dict[int, list[Rect]] = {}
    for player in overlay["players"].values():
        for f, box in enumerate(player["boxes"]):
            if box:
                out.setdefault(f, []).append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
    return out


def label_boxes(labels: dict[str, Any], size: list[int] | tuple[int, int] | None) -> dict[int, list[Rect]]:
    """Boxes of the scored frames (`done`, or every frame when the file has no `done`), scaled to `size`."""
    frames = labels.get("frames") or {}
    keys = labels["done"] if "done" in labels else list(frames)
    sx = sy = 1.0
    if size and labels.get("imageSize"):
        sx, sy = size[0] / labels["imageSize"][0], size[1] / labels["imageSize"][1]
    return {
        int(key): [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in frames.get(str(key), [])]
        for key in keys
    }


def server_seconds(timings: dict[str, float]) -> float | None:
    """Sum of the sequential server steps (`track` is the wall time of its parallel block)."""
    steps = [timings[k] for k in ("frames", "track", "camera", "build", "name") if k in timings]
    return round(sum(steps), 2) if steps else None


def sam_cost(meta: dict[str, Any]) -> float:
    """SAM 3.1 calls only: players, plus the ball call when it ran. Naming and Modal compute not included."""
    calls = 2 if "samBall" in (meta.get("timings") or {}) or meta.get("ballTracked") else 1
    return round((meta.get("samFrames") or 0) * calls * SAM_USD_PER_1000_FRAMES / 1000, 4)


def clip_report(
    clip: str, entry: dict[str, Any], job: dict[str, Any], labels: dict[str, Any]
) -> dict[str, Any]:
    overlay = job["overlay"]
    meta = (job.get("playDna") or {}).get("meta") or {}
    camera = meta.get("cameraMotion") or {}
    return {
        "clip": clip,
        "jobId": job["jobId"],
        "labelledBy": labels.get("labelledBy") or "a person",
        **score_frames(overlay_boxes(overlay), label_boxes(labels, overlay.get("imageSize"))),
        "samFrames": meta.get("samFrames"),
        "cameraFallbackFrames": camera.get("framesFallback"),
        "ballFrames": meta.get("ballFrames"),
        "wallSeconds": entry.get("wallSeconds"),
        "serverSeconds": server_seconds(meta.get("timings") or {}),
        "samCostUsd": sam_cost(meta),
    }


def overall(reports: list[dict[str, Any]]) -> dict[str, Any]:
    def total(key: str) -> int:
        return sum(r.get(key) or 0 for r in reports)

    def mean(key: str) -> float | None:
        vals = [r[key] for r in reports if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    n = total("labelledFrames")
    tp, fp, fn = total("truePositives"), total("falsePositives"), total("falseNegatives")
    return {
        "clips": len(reports),
        "labelledFrames": n,
        "truePositives": tp,
        "falsePositives": fp,
        "falseNegatives": fn,
        **precision_recall(tp, fp, fn),
        "playersFoundPerFrame": round(total("playersFound") / n, 2) if n else None,
        "playersLabelledPerFrame": round(total("playersLabelled") / n, 2) if n else None,
        "samFrames": total("samFrames"),
        "cameraFallbackFrames": total("cameraFallbackFrames"),
        "ballFrames": total("ballFrames"),
        "wallSecondsMean": mean("wallSeconds"),
        "serverSecondsMean": mean("serverSeconds"),
        "samCostUsdPerClip": mean("samCostUsd"),
        "samCostUsdTotal": round(sum(r["samCostUsd"] for r in reports), 4),
    }


COLUMNS: list[tuple[str, str | None]] = [
    ("Clip", "clip"),
    ("Frames", "labelledFrames"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("TP / FP / FN", None),
    ("Found / labelled per frame", None),
    ("Camera fallback", None),
    ("Ball frames", None),
    ("Wall s", "wallSeconds"),
    ("Server s", "serverSeconds"),
    ("SAM $", "samCostUsd"),
]


def summary_row(r: dict[str, Any]) -> list[str]:
    frames = r.get("samFrames") or 0
    cells = {
        "TP / FP / FN": f"{r['truePositives']} / {r['falsePositives']} / {r['falseNegatives']}",
        "Found / labelled per frame": f"{r['playersFoundPerFrame']} / {r['playersLabelledPerFrame']}",
        "Camera fallback": f"{r['cameraFallbackFrames']} of {frames}",
        "Ball frames": f"{r['ballFrames']} of {frames}",
    }
    return [cells[name] if key is None else str(r.get(key)) for name, key in COLUMNS]


def summary_md(summary: dict[str, Any]) -> str:
    o = summary["overall"]
    head = [name for name, _ in COLUMNS]
    lines = [
        "# Eval summary",
        "",
        f"Generated {summary['generatedAt']}. {o['clips']} clips, {o['labelledFrames']} checked frames, "
        f"labelled by {'; '.join(summary.get('labelledBy') or ['a person'])}. "
        "Player boxes from the job overlay (the kept tracks), matched one-to-one to labels at "
        f"IoU >= {IOU_MATCH}.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    lines += ["| " + " | ".join(summary_row(r)) + " |" for r in summary["clips"]]
    total = {
        **o,
        "clip": "**overall**",
        "wallSeconds": o["wallSecondsMean"],
        "serverSeconds": o["serverSecondsMean"],
        "samCostUsd": o["samCostUsdTotal"],
    }
    lines.append("| " + " | ".join(summary_row(total)) + " |")
    lines += [
        "",
        "- Wall s: upload to `done` as seen by `scripts/run_clip.py` (overall: mean). "
        "Server s: sum of the server steps (overall: mean).",
        f"- SAM $: SAM 3.1 frames (players + ball calls) at ${SAM_USD_PER_1000_FRAMES:.2f} per 1,000 frames; "
        "Muse Spark naming and Modal compute are not included (overall: total).",
        "- Camera fallback and ball frames count every clip frame, not only the labelled ones.",
    ]
    if summary["notLabelled"]:
        lines.append(f"- Not labelled yet, not scored: {', '.join(summary['notLabelled'])}.")
    return "\n".join(lines) + "\n"


def run_summary(base_override: str | None, refresh: bool) -> int:
    jobs = jobcache.load_jobs()
    base = jobcache.backend_url(jobs, base_override)
    reports, skipped = [], []
    for clip, entry in sorted(jobs["clips"].items()):
        path = jobcache.label_path(clip)
        labels = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not label_boxes(labels, None):
            skipped.append(clip)
            continue
        if labels.get("jobId") and labels["jobId"] != entry["jobId"]:
            sys.exit(f"{path} was labelled on {labels['jobId']}, but eval/jobs.json says {entry['jobId']}")
        job = jobcache.cached_job(clip, entry["jobId"], base, refresh)
        reports.append(clip_report(clip, entry, job, labels))
    if not reports:
        print(f"no labelled frames for the {len(jobs['clips'])} clips in eval/jobs.json: {NOTHING}")
        return 0
    summary = {
        "generatedAt": datetime.now(UTC).strftime("%Y-%m-%d"),
        "iouThreshold": IOU_MATCH,
        "matching": "greedy one-to-one, best IoU first",
        "boxes": "job overlay players (the kept tracks) vs the done frames of eval/labels, clip.mp4 pixels",
        "labelledBy": sorted({r["labelledBy"] for r in reports}),
        "overall": overall(reports),
        "clips": reports,
        "notLabelled": skipped,
    }
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    md = summary_md(summary)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out / 'summary.json'} and summary.md", file=sys.stderr)
    return 0


def rect(b: Box) -> Rect:
    return (b.x1, b.y1, b.x2, b.y2)


def score(sam: SamResult, labels: dict) -> dict:
    """Labels may carry `imageSize`; their boxes are scaled to SAM's frame size before matching."""
    truth = label_boxes(labels, (sam.width, sam.height) if sam.width else None)
    pred = {f: [rect(b) for b in boxes] for f, boxes in sam.frames.items()}
    counts = score_frames(pred, truth)
    return {k: counts[k] for k in ("labelledFrames", "truePositives", "falsePositives", "falseNegatives")} | {
        "precision": counts["precision"],
        "recall": counts["recall"],
    }


def per_frame(sam: SamResult) -> dict:
    counts = [len(sam.frames.get(i, [])) for i in range(sam.frame_count)]
    if not counts:
        return {"frames": 0}
    return {
        "frames": len(counts),
        "playersMin": min(counts),
        "playersMedian": statistics.median(counts),
        "playersMax": max(counts),
        "framesWith10Plus": sum(1 for c in counts if c >= 10),
        "objectIds": len(sam.tracks()),
    }


def load_sam(clip: Path, stream: Path | None) -> SamResult:
    if stream:
        with stream.open(encoding="utf-8") as f:
            head = f.read(64)
            f.seek(0)
            text = deltas_from_sse(f) if head.startswith(("event:", "data:")) else f.read()
        return parse_sam_text(text)
    from dotenv import load_dotenv

    from app import settings
    from app.jobs import reencode
    from app.sam import stream_sam, upload_clip

    load_dotenv(ROOT / ".env")
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory() as tmp:
        small = Path(tmp) / "clip.mp4"
        reencode(clip, small, imageio_ffmpeg.get_ffmpeg_exe())
        client = settings.meta_client()
        text = stream_sam(client, upload_clip(client, small), Path(tmp) / "sam.txt")
    return parse_sam_text(text)


def table(rows: dict) -> str:
    lines = ["| Measure | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows.items()]
    return "\n".join(lines)


def run_clip(args: argparse.Namespace) -> int:
    sam = load_sam(args.clip, args.stream)
    result: dict = {"clip": args.clip.name, "source": str(args.stream or "sam-3.1 call"), **per_frame(sam)}
    labels = None
    if args.labels and args.labels.exists():
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels and label_boxes(labels, None):
        if labels.get("fps") and float(labels["fps"]) != args.fps:
            sys.exit(f"labels are at {labels['fps']} fps but SAM frames are at {args.fps} fps; pass --fps")
        result.update(score(sam, labels))
        print(table(result))
    else:
        print(table(result))
        print(f"\nno labels yet, {NOTHING}")
        result["note"] = f"no labels yet, {NOTHING}"
    out = ROOT / "eval" / "results" / f"{args.clip.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip", type=Path, help="score one clip from a SAM stream instead of the eval jobs")
    parser.add_argument("--labels", type=Path)
    parser.add_argument(
        "--stream", type=Path, help="saved SAM stream (raw SSE or sam.txt) instead of an API call"
    )
    parser.add_argument("--fps", type=float, default=12, help="frame rate SAM saw (12 for a fresh call)")
    parser.add_argument("--url", help="backend URL (default: HOOPSCLINIQ_API_URL, then eval/jobs.json)")
    parser.add_argument("--refresh", action="store_true", help="fetch the job views again, not the cache")
    args = parser.parse_args()
    if args.clip is None:
        return run_summary(args.url, args.refresh)
    return run_clip(args)


if __name__ == "__main__":
    raise SystemExit(main())
