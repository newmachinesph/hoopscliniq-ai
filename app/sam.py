"""SAM 3.1 on the Meta Model API: upload a clip, stream the segmentation, parse boxes out of the text."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import settings

FRAME_TAG = re.compile(r"<(\d+)f>")
# Each match is `idx<|box;...|><|mask;...|>`; the idx directly precedes `<|box`. Masks are ignored.
BOX = re.compile(r"(\d+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);w=(\d+);h=(\d+)\|>")


@dataclass(frozen=True)
class Box:
    obj: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class SamResult:
    frames: dict[int, list[Box]] = field(default_factory=dict)
    width: int = 0
    height: int = 0

    @property
    def frame_count(self) -> int:
        return max(self.frames) + 1 if self.frames else 0

    def tracks(self) -> dict[int, dict[int, Box]]:
        """Object id -> {frame index -> box}. SAM keeps object ids stable across frames."""
        out: dict[int, dict[int, Box]] = {}
        for frame, boxes in self.frames.items():
            for b in boxes:
                out.setdefault(b.obj, {})[frame] = b
        return out


def parse_sam_text(text: str) -> SamResult:
    """Parse the concatenated `response.output_text.delta` text into boxes per frame."""
    result = SamResult()
    parts = FRAME_TAG.split(text)
    # parts = [prefix, idx, body, idx, body, ...]
    for i in range(1, len(parts) - 1, 2):
        frame = int(parts[i])
        boxes = []
        for m in BOX.finditer(parts[i + 1]):
            obj, x1, y1, x2, y2, w, h = (int(v) for v in m.groups())
            boxes.append(Box(obj, x1, y1, x2, y2))
            result.width, result.height = w, h
        result.frames[frame] = boxes
    return result


def deltas_from_sse(lines: Iterable[str]) -> str:
    """Concatenate every `response.output_text.delta` from raw server-sent-event lines."""
    chunks = []
    for line in lines:
        if not line.startswith("data: ") or '"response.output_text.delta"' not in line:
            continue
        event = json.loads(line[6:])
        if event.get("type") == "response.output_text.delta":
            chunks.append(event["delta"])
    return "".join(chunks)


def upload_clip(client: Any, path: Path) -> str:
    with path.open("rb") as f:
        uploaded = client.files.create(
            file=(path.name, f, "video/mp4"),
            purpose="user_data",
            expires_after={"anchor": "created_at", "seconds": settings.SAM_FILE_TTL_SECONDS},
        )
    return uploaded.id


def stream_sam(client: Any, file_id: str, out_path: Path) -> str:
    """Run SAM 3.1 on an uploaded clip, write the raw text to `out_path` and return it."""
    stream = client.responses.create(
        model=settings.SAM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": settings.SAM_PROMPT},
                    {"type": "input_video", "file_id": file_id},
                ],
            }
        ],
        stream=True,
    )
    chunks: list[str] = []
    for event in stream:
        kind = getattr(event, "type", "")
        if kind == "response.output_text.delta":
            chunks.append(event.delta)
        elif kind in ("response.failed", "error"):
            raise RuntimeError(f"SAM stream failed: {getattr(event, 'response', event)}")
    text = "".join(chunks)
    out_path.write_text(text, encoding="utf-8")
    return text
