"""End-to-end check of one clip against the deployed backend: upload, poll until done, print the results.

    python scripts/run_clip.py ../clips/hazen-girls-lakeregion-2026-02-11_t1200.mp4 [--url https://...modal.run]

The URL comes from --url or HOOPSCLINIQ_API_URL. Saves play.json to eval/results/<clip>.play.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
POLL_SECONDS = 2.0
TIMEOUT_SECONDS = 900


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--url", default=os.environ.get("HOOPSCLINIQ_API_URL", ""))
    parser.add_argument("--calibration", help="JSON string passed through as the `calibration` form field")
    args = parser.parse_args()
    if not args.url:
        sys.exit("Set --url or HOOPSCLINIQ_API_URL to the deployed web URL.")
    base = args.url.rstrip("/")

    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        t0 = time.monotonic()
        with args.clip.open("rb") as f:
            data = {"calibration": args.calibration} if args.calibration else None
            files = {"file": (args.clip.name, f, "video/mp4")}
            r = client.post(f"{base}/film/upload", files=files, data=data)
        print(f"POST /film/upload -> {r.status_code} in {time.monotonic() - t0:.1f}s: {r.text}")
        r.raise_for_status()
        job_id = r.json()["jobId"]

        seen: dict[str, float] = {}
        job: dict = {}
        while time.monotonic() - t0 < TIMEOUT_SECONDS:
            job = client.get(f"{base}/film/jobs/{job_id}").json()
            if job["status"] not in seen:
                seen[job["status"]] = time.monotonic() - t0
                print(f"  {seen[job['status']]:6.1f}s  status={job['status']}")
            if job["status"] in ("done", "failed"):
                break
            time.sleep(POLL_SECONDS)

    total = time.monotonic() - t0
    print(f"\njobId: {job_id}\nstatus: {job.get('status')} after {total:.1f}s (wall clock, upload included)")
    print("steps:", ", ".join(f"{s['id']}={s['status']}" for s in job.get("steps", [])))
    if job.get("error"):
        print("error:", job["error"])
    play = job.get("playDna")
    if play:
        meta = play["meta"]
        print("server step timings (s):", json.dumps(meta.get("timings")))
        print(
            f"playersFound: {meta['playersFound']} | samFrames: {meta['samFrames']} | "
            f"clip frames: {play['source']['frames']} | imageSize: {meta['imageSize']} | "
            f"calibration: {meta['calibration']}"
        )
        print("coverage:", json.dumps(meta.get("coverage")))
        out = ROOT / "eval" / "results" / f"{args.clip.stem}.play.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(play), encoding="utf-8")
        print("saved", out)
    print("suggestedName:", json.dumps(job.get("suggestedName")))
    return 0 if job.get("status") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
