"""ClinIQ: Muse Spark answers a coach's question from tool results only, citing every paragraph."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app import settings
from app.world import World

log = logging.getLogger(__name__)

NO_SOURCE = "I have no source for that in your film."
MARKER = re.compile(r"\[(\d+)\]")
KINDS = ("play", "clip", "stat")

SYSTEM = (
    "You are ClinIQ, the assistant inside Hoops ClinIQ for a high-school basketball coach (team: {team}). "
    "Answer ONLY from the results of your tools (fetch_play, fetch_clip, fetch_stat); never from general "
    "knowledge, and never invent numbers. Look things up before answering. Every citation you use must be a "
    "`citation` object copied exactly from a tool result. Write short paragraphs; every paragraph ends with "
    '[n] markers (1-based) that index your citations list. If the tools return nothing relevant, say "I have '
    'no source for that in your film." and nothing else.\n\n{catalog}'
)
FINAL = (
    "Now write the final answer as JSON: body = paragraphs (each ends with [n] markers into citations), "
    "citations = the citation objects you used (kind, label, href copied from tool results), confidence = "
    "high|medium|low, or null when you have no source."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_play",
            "description": "A playbook play: description, times run, points per possession, clip links.",
            "parameters": {
                "type": "object",
                "properties": {"playId": {"type": "string", "description": "Play id from the catalog."}},
                "required": ["playId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_clip",
            "description": "A game or scout film: result, defenses faced, points by set and defense, and the "
            "tagged possessions (clips), optionally only those of one play.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filmId": {"type": "string", "description": "Film id from the catalog."},
                    "playId": {"type": "string", "description": "Optional play id to filter possessions."},
                },
                "required": ["filmId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_stat",
            "description": "Season and team stats matching a short query, e.g. 'turnovers' or 'vs zone'.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "body": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "label": {"type": "string"},
                    "href": {"type": "string"},
                },
                "required": ["kind", "label", "href"],
                "additionalProperties": False,
            },
        },
        "confidence": {"anyOf": [{"type": "string", "enum": ["high", "medium", "low"]}, {"type": "null"}]},
    },
    "required": ["body", "citations", "confidence"],
    "additionalProperties": False,
}


def run_tool(world: World, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "fetch_play":
        return world.fetch_play(str(args.get("playId", "")))
    if name == "fetch_clip":
        play_id = args.get("playId")
        return world.fetch_clip(str(args.get("filmId", "")), str(play_id) if play_id else None)
    if name == "fetch_stat":
        return world.fetch_stat(str(args.get("query", "")))
    return {"found": False, "error": f"Unknown tool {name}."}


def no_source() -> dict[str, Any]:
    return {"body": [NO_SOURCE], "citations": [], "confidence": None, "credits": 0}


def postcheck(raw: dict[str, Any], world: World) -> tuple[dict[str, Any], bool]:
    """Keep only citations that exist in the world, renumber the [n] markers to match, and require every
    paragraph to carry at least one valid marker. Returns (answer, noSource)."""
    citations = raw.get("citations") or []
    body = [str(p).strip() for p in (raw.get("body") or []) if str(p).strip()]
    kept: list[dict[str, str]] = []
    renumber: dict[int, int] = {}
    for i, c in enumerate(citations, start=1):
        if not isinstance(c, dict) or c.get("href") not in world.hrefs or c.get("kind") not in KINDS:
            continue
        key = (c["kind"], c["href"])
        existing = next((j for j, k in enumerate(kept, start=1) if (k["kind"], k["href"]) == key), None)
        if existing:
            renumber[i] = existing
            continue
        kept.append({"kind": c["kind"], "label": str(c.get("label") or c["href"]), "href": c["href"]})
        renumber[i] = len(kept)
    if not kept or not body:
        return no_source(), True

    def fix(paragraph: str) -> str:
        text = MARKER.sub(
            lambda m: f"[{renumber[int(m.group(1))]}]" if int(m.group(1)) in renumber else "", paragraph
        )
        return re.sub(r"(\[\d+\])(\1)+", r"\1", text).replace(" .", ".").strip()

    fixed = [fix(p) for p in body]
    if any(not MARKER.search(p) for p in fixed):
        return no_source(), True
    confidence = raw.get("confidence") if raw.get("confidence") in ("high", "medium", "low") else None
    return {"body": fixed, "citations": kept, "confidence": confidence, "credits": 1}, False


def ask(client: Any, model: str, world: World, question: str) -> dict[str, Any]:
    system = SYSTEM.format(team=world.data.get("team", "our team"), catalog=world.catalog())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    calls = 0
    while calls < settings.MAX_TOOL_CALLS:
        resp = client.chat.completions.create(
            model=model, messages=list(messages), tools=TOOLS, tool_choice="auto"
        )
        msg = resp.choices[0].message
        tool_calls = list(msg.tool_calls or [])
        if not tool_calls:
            break
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": t.id,
                        "type": "function",
                        "function": {"name": t.function.name, "arguments": t.function.arguments},
                    }
                    for t in tool_calls
                ],
            }
        )
        for t in tool_calls:
            calls += 1
            if calls > settings.MAX_TOOL_CALLS:
                result: dict[str, Any] = {
                    "found": False,
                    "error": "Tool budget used up; answer with what you have.",
                }
            else:
                try:
                    args = json.loads(t.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(world, t.function.name, args if isinstance(args, dict) else {})
            log.info("cliniq tool %s -> found=%s", t.function.name, result.get("found"))
            messages.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(result)})
    messages.append({"role": "user", "content": FINAL})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA, "strict": True},
        },
    )
    try:
        raw = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        raw = {}
    answer, missing = postcheck(raw if isinstance(raw, dict) else {}, world)
    return {"question": question, "answer": answer, "noSource": missing}
