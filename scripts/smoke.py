"""Smoke test the deployed backend: GET /health, then POST /cliniq/ask with one question.

python scripts/smoke.py "What did we run against Westview's zone and what worked?" [--url https://...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--url", default=os.environ.get("HOOPSCLINIQ_API_URL", ""))
    args = parser.parse_args()
    if not args.url:
        sys.exit("Set --url or HOOPSCLINIQ_API_URL to the deployed web URL.")
    base = args.url.rstrip("/")
    with httpx.Client(timeout=httpx.Timeout(180.0)) as client:
        h = client.get(f"{base}/health")
        print(f"GET /health -> {h.status_code} {h.text}  [x-demo={h.headers.get('x-demo')}]")
        t0 = time.monotonic()
        r = client.post(f"{base}/cliniq/ask", json={"question": args.question})
        print(f"POST /cliniq/ask -> {r.status_code} in {time.monotonic() - t0:.1f}s")
        body = r.json()
        print(json.dumps(body, indent=1))
    if r.status_code != 200:
        return 1
    hrefs = [c["href"] for c in body["answer"]["citations"]]
    linked = [h for h in hrefs if h.startswith(("/film/", "/plays/"))]
    print(f"\nnoSource={body['noSource']} citations={len(hrefs)} film/plays citations={len(linked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
