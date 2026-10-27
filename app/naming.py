"""Muse Spark names the tracked play in the coach's own playbook words (a suggestion the coach confirms)."""

from __future__ import annotations

import json
from typing import Any

from app import settings
from app.playdna import UNNAMED

MAX_WHY = 160
CONFIDENCE = ("high", "medium", "low")

SYSTEM = (
    "You are an assistant basketball coach. You match tracked player movement to a play in the coach's own "
    "playbook. Court coordinates are feet: x across the court 0-50, y out from the attacked baseline (rim at "
    "25, 5.25; half-court line at y=47). The tracking comes from one phone or broadcast camera with an "
    "approximate court calibration, so positions are noisy and team labels can be wrong. Choose a playbook "
    "id "
    'only when the movement really matches its description; otherwise answer "unknown". Be honest about '
    "confidence. `why` is one short sentence (max 160 characters) a coach can check against the clip."
)


def naming_schema(play_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "playId": {"type": "string", "enum": [*play_ids, "unknown"]},
            "title": {"type": "string"},
            "confidence": {"type": "string", "enum": list(CONFIDENCE)},
            "why": {"type": "string"},
        },
        "required": ["playId", "title", "confidence", "why"],
        "additionalProperties": False,
    }


def track_summary(play_dna: dict[str, Any]) -> str:
    """Each player's court position every second (rounded to 1 ft) plus the start and end frame."""
    fps = play_dna["source"]["fps"]
    frames = play_dna["source"]["frames"]
    lines = [f"{frames} frames at {fps} fps ({frames / fps:.1f} s). Positions as (x,y) ft."]
    for pid, track in play_dna["tracks"]["players"].items():
        if not track:
            continue
        team = play_dna["players"][pid]["team"]
        idx = range(0, len(track), fps)
        pts = " ".join(f"{i // fps}s({round(track[i][0])},{round(track[i][1])})" for i in idx)
        start, end = track[0], track[-1]
        lines.append(
            f"{pid} {team}: start ({round(start[0])},{round(start[1])}) "
            f"end ({round(end[0])},{round(end[1])}); {pts}"
        )
    return "\n".join(lines)


def naming_messages(playbook: list[dict], play_dna: dict[str, Any]) -> list[dict[str, str]]:
    book = "\n".join(f"- {p['id']}: {p['title']}. {p['description']}" for p in playbook)
    meta = play_dna.get("meta", {})
    user = (
        f"Coach's playbook:\n{book}\n\n"
        f"Tracked clip ({meta.get('playersFound', '?')} players found, "
        f"calibration {meta.get('calibration', '?')}):\n"
        f"{track_summary(play_dna)}\n\n"
        "Which playbook play is this? Reply with the JSON object only."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def to_suggested_name(raw: dict[str, Any], playbook: list[dict]) -> dict[str, Any]:
    titles = {p["id"]: p["title"] for p in playbook}
    play_id = raw.get("playId") if raw.get("playId") in titles else "unknown"
    confidence = raw.get("confidence") if raw.get("confidence") in CONFIDENCE else "low"
    title = titles[play_id] if play_id != "unknown" else UNNAMED
    why = str(raw.get("why", "")).strip()
    if len(why) > MAX_WHY:
        why = why[: MAX_WHY - 1].rstrip() + "…"
    return {"playId": play_id, "title": title, "confidence": confidence, "why": why, "needsReview": True}


def suggest_name(client: Any, playbook: list[dict], play_dna: dict[str, Any]) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=settings.SPARK_MODEL,
        messages=naming_messages(playbook, play_dna),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "play_name",
                "schema": naming_schema([p["id"] for p in playbook]),
                "strict": True,
            },
        },
    )
    return to_suggested_name(json.loads(resp.choices[0].message.content or "{}"), playbook)
