"""Muse Spark writes a short study quiz for one player spot from the frames of a playbook play."""

from __future__ import annotations

import json
import logging
import math
import random
import secrets
import string
import time
from typing import Any

from app import settings
from app.models import PlayAction, PlayFrame, QuizRequest

log = logging.getLogger(__name__)

MAX_PROMPT = 200
MAX_OPTION = 120
MAX_WHY = 220
MIN_QUESTIONS = 2
OPTION_IDS = ("a", "b", "c")
ID_CHARS = string.ascii_lowercase + string.digits
FAILED = "ClinIQ could not write a quiz for this play. Try again."
TOO_SLOW = "ClinIQ took too long to write this quiz. Try again."

SYSTEM = (
    "You are an assistant basketball coach. You write study questions for a high-school player who plays "
    "spot {spot} in a play from the coach's playbook. The play is given frame by frame, in words. Ask only "
    "about what the frames show: where to go, who to screen, who gets the ball, what to read. Never invent "
    "actions that are not in the frames. Frames are counted from 1, as in the description, everywhere: in "
    "the text and in the fields. Each question names the frame it shows (`frame`, the frame the player sees "
    "before answering) and the frame that shows the answer (`answerFrame`, never before `frame`). "
    "Give exactly 3 short answer options, one correct (`correct` is "
    "its index, 0 to 2) and two plausible wrong ones. `why` is one sentence that points to the frame and "
    'the action. Players are numbered 1 to 5; "left" and "right" are as seen facing the basket from the '
    "top of the court."
)


class QuizError(Exception):
    """A quiz that could not be written: `status` and a friendly `message` for the response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def zone(x: float, y: float) -> str:
    """Court zone name for a point in tenths of a foot (x 0..500 sideline to sideline, y 0..420 from the
    baseline, rim at (250, 52)). Left is the low-x side. Reads after "the": "at the left elbow"."""
    dx = x - 250
    ax = abs(dx)
    side = "left" if dx < 0 else "right"
    if y <= 140 and ax >= 160:
        return f"{side} corner"
    if y <= 190 and ax <= 60:  # the lane
        if y <= 75 and ax <= 30:
            return "rim"
        if y <= 110:
            return f"{side} low post"
        if y <= 150:
            return f"{side} mid post"
        return "high post" if ax <= 30 else f"{side} elbow"
    if y <= 140:
        return f"{side} baseline"
    if y >= 360:
        return "half court"
    if y <= 250:
        if ax <= 30 and y > 190:
            return "high post"
        return f"{side} elbow" if ax <= 110 else f"{side} wing"
    if ax <= 80:
        return "top of the key"
    return f"{side} wing"


def nearest(points: list[tuple[float, float]], target: tuple[float, float]) -> int:
    """Index of the point nearest `target` (the first on a tie)."""
    return min(range(len(points)), key=lambda i: math.dist(points[i], target))


def describe_action(action: PlayAction, offense: list[tuple[float, float]]) -> str:
    actor = nearest(offense, action.from_) + 1
    target = nearest(offense, action.to) + 1
    where = zone(*action.to)
    kind = action.kind.lower()
    if kind in ("pass", "handoff", "screen") and target != actor:
        if kind == "pass":
            return f"{actor} passes to {target}"
        if kind == "handoff":
            return f"{actor} hands off to {target}"
        return f"{actor} screens for {target} at the {zone(*action.from_)}"
    if kind == "screen":
        return f"{actor} sets a screen at the {zone(*action.from_)}"
    if kind == "dribble":
        return f"{actor} dribbles to the {where}"
    via = f" via the {zone(*action.via)}" if action.via else ""
    verb = "cuts" if kind == "cut" else f"moves ({kind})"
    return f"{actor} {verb} to the {where}{via}"


def describe_frame(frame: PlayFrame, number: int) -> str:
    offense = list(frame.offense)
    spots = ", ".join(f"{i + 1} at the {zone(*p)}" for i, p in enumerate(offense))
    lines = [f"Frame {number}: {spots}. Player {frame.ball + 1} has the ball."]
    if frame.actions:
        lines.append("Actions: " + "; ".join(describe_action(a, offense) for a in frame.actions) + ".")
    if frame.note:
        lines.append(f"Coach's note: {frame.note.strip()}")
    return "\n".join(lines)


def describe_play(req: QuizRequest) -> str:
    """The play in words, frame by frame (frame numbers start at 1, as players and the model count them)."""
    head = f'Play "{req.play.title}", {len(req.play.frames)} frames. Frame numbers below start at 1.'
    return head + "\n\n" + "\n".join(describe_frame(f, i + 1) for i, f in enumerate(req.play.frames))


def quiz_schema() -> dict[str, Any]:
    question = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "frame": {"type": "integer"},
            "answerFrame": {"type": "integer"},
            "options": {"type": "array", "items": {"type": "string"}},
            "correct": {"type": "integer"},
            "why": {"type": "string"},
        },
        "required": ["prompt", "frame", "answerFrame", "options", "correct", "why"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"questions": {"type": "array", "items": question}},
        "required": ["questions"],
        "additionalProperties": False,
    }


def quiz_messages(req: QuizRequest) -> list[dict[str, str]]:
    user = (
        f"{describe_play(req)}\n\n"
        f"Write {req.count} questions for the player at spot {req.spot}. "
        "Frame numbers in your answer start at 1. Reply with the JSON object only."
    )
    return [{"role": "system", "content": SYSTEM.format(spot=req.spot)}, {"role": "user", "content": user}]


def trim(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def clean_question(raw: Any, frames: int) -> dict[str, Any] | None:
    """One valid question (text trimmed to the limits, frames turned 0-based), or None. The model counts
    frames from 1 so the numbers it writes in the text match the frame shown to the player."""
    if not isinstance(raw, dict):
        return None
    frame, answer, correct = raw.get("frame"), raw.get("answerFrame"), raw.get("correct")
    if not (is_index(frame) and is_index(answer) and is_index(correct)):
        return None
    if not (1 <= frame <= frames and frame <= answer <= frames and 0 <= correct < len(OPTION_IDS)):
        return None
    options = raw.get("options")
    if not isinstance(options, list) or len(options) != len(OPTION_IDS):
        return None
    if not all(isinstance(o, str) for o in options):
        return None
    options = [trim(o, MAX_OPTION) for o in options]
    prompt, why = raw.get("prompt"), raw.get("why")
    if not (isinstance(prompt, str) and isinstance(why, str)) or not prompt.strip() or not why.strip():
        return None
    if not all(options) or len(set(options)) != len(options):
        return None
    return {
        "prompt": trim(prompt, MAX_PROMPT),
        "frame": frame - 1,
        "answerFrame": answer - 1,
        "options": options,
        "correct": correct,
        "why": trim(why, MAX_WHY),
    }


def build_questions(raw: Any, req: QuizRequest) -> list[dict[str, Any]]:
    """Valid questions only (at most `count`), options shuffled per question, ids q1.. and a, b, c."""
    items = raw.get("questions") if isinstance(raw, dict) else None
    frames = len(req.play.frames)
    valid = [q for q in (clean_question(r, frames) for r in items or []) if q]
    out = []
    for i, q in enumerate(valid[: req.count]):
        order = list(range(len(OPTION_IDS)))
        random.Random(f"{req.play.id}:{i}").shuffle(order)
        out.append(
            {
                "id": f"q{i + 1}",
                "prompt": q["prompt"],
                "playId": req.play.id,
                "frame": q["frame"],
                "answerFrame": q["answerFrame"],
                "options": [{"id": OPTION_IDS[n], "text": q["options"][old]} for n, old in enumerate(order)],
                "correctId": OPTION_IDS[order.index(q["correct"])],
                "why": q["why"],
            }
        )
    return out


def is_timeout(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or type(exc).__name__ == "APITimeoutError"


def write_quiz(client: Any, req: QuizRequest) -> dict[str, Any]:
    """The `POST /study/quiz` response body. Raises QuizError with a friendly message."""
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=settings.SPARK_MODEL,
            messages=quiz_messages(req),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "play_quiz", "schema": quiz_schema(), "strict": True},
            },
            reasoning_effort=settings.QUIZ_EFFORT,
        )
        content = resp.choices[0].message.content or "{}"
    except Exception as exc:
        log.warning("quiz call failed for play %s", req.play.id, exc_info=True)
        if is_timeout(exc):
            raise QuizError(504, TOO_SLOW) from exc
        raise QuizError(502, FAILED) from exc
    try:
        raw = json.loads(content)
    except ValueError:
        log.warning("quiz reply was not JSON for play %s: %.200s", req.play.id, content)
        raw = {}
    questions = build_questions(raw, req)
    if len(questions) < MIN_QUESTIONS:
        log.warning("quiz for play %s had %d valid questions", req.play.id, len(questions))
        raise QuizError(502, FAILED)
    suffix = "".join(secrets.choice(ID_CHARS) for _ in range(6))
    return {
        "quiz": {
            "id": f"gen-{req.play.id}-{suffix}",
            "title": f"{req.play.title}: spot {req.spot}",
            "topic": "Written by ClinIQ",
            "due": None,
            "spot": req.spot,
            "questions": questions,
        },
        "source": {"playId": req.play.id, "title": req.play.title, "frames": len(req.play.frames)},
        "model": settings.SPARK_MODEL,
        "seconds": round(time.monotonic() - t0, 2),
    }
