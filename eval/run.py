"""Eval harness: SAM 3.1 player boxes vs hand labels.

    python -m eval.run --clip ../clips/<file>.mp4 [--labels eval/labels/<clip>.json] [--stream raw-sse.txt]

Boxes come from `--stream` (a saved raw SAM event stream or the pipeline's sam.txt), or from a fresh SAM call
on the clip re-encoded to 12 fps (needs META_API_KEY; about $0.05 per 20 s). With labels: precision and recall
of player boxes at IoU >= 0.5 on the labelled frames. Without labels: players-found per frame only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

from app.sam import Box, SamResult, deltas_from_sse, parse_sam_text

ROOT = Path(__file__).resolve().parent.parent
IOU_MATCH = 0.5
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
    for score, i, j in pairs:
        if score < threshold:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        tp += 1
    return tp


def rect(b: Box) -> Rect:
    return (b.x1, b.y1, b.x2, b.y2)


def score(sam: SamResult, labels: dict) -> dict:
    """Labels may carry `imageSize`; their boxes are scaled to SAM's frame size before matching."""
    sx = sy = 1.0
    if labels.get("imageSize") and sam.width:
        sx, sy = sam.width / labels["imageSize"][0], sam.height / labels["imageSize"][1]
    tp = fp = fn = 0
    for key, truth in labels["frames"].items():
        pred = [rect(b) for b in sam.frames.get(int(key), [])]
        scaled = [(t[0] * sx, t[1] * sy, t[2] * sx, t[3] * sy) for t in truth]
        hit = match_frame(pred, scaled)
        tp += hit
        fp += len(pred) - hit
        fn += len(truth) - hit
    return {
        "labelledFrames": len(labels["frames"]),
        "truePositives": tp,
        "falsePositives": fp,
        "falseNegatives": fn,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument(
        "--stream", type=Path, help="saved SAM stream (raw SSE or sam.txt) instead of an API call"
    )
    parser.add_argument("--fps", type=float, default=12, help="frame rate SAM saw (12 for a fresh call)")
    args = parser.parse_args()

    sam = load_sam(args.clip, args.stream)
    result: dict = {"clip": args.clip.name, "source": str(args.stream or "sam-3.1 call"), **per_frame(sam)}
    labels = None
    if args.labels and args.labels.exists():
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels and labels.get("frames"):
        if labels.get("fps") and float(labels["fps"]) != args.fps:
            sys.exit(f"labels are at {labels['fps']} fps but SAM frames are at {args.fps} fps; pass --fps")
        result.update(score(sam, labels))
        print(table(result))
    else:
        print(table(result))
        print("\nno labels yet, nothing measured")
        result["note"] = "no labels yet, nothing measured"
    out = ROOT / "eval" / "results" / f"{args.clip.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
