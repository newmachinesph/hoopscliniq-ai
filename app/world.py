"""The demo's fictional world (data/world.json) and the lookups ClinIQ's tools run against it."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app import settings

MAX_POSSESSIONS = 40
MAX_STATS = 5


class World:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.plays: dict[str, dict] = {p["id"]: p for p in data["plays"]}
        self.films: dict[str, dict] = {f["id"]: f for f in data["films"]}
        self.stats: list[dict] = data["stats"]
        self.hrefs: set[str] = self._hrefs()

    def _hrefs(self) -> set[str]:
        out = {s["href"] for s in self.stats}
        for p in self.plays.values():
            out.add(p["href"])
            out.update(c["href"] for c in p["clips"])
        for f in self.films.values():
            out.add(f["href"])
            out.update(x["href"] for x in f["possessions"])
        return out

    def playbook(self) -> list[dict]:
        return [
            {"id": p["id"], "title": p["title"], "description": p["description"]} for p in self.plays.values()
        ]

    def catalog(self) -> str:
        """Ids the model may pass to the tools (titles only, no facts)."""
        plays = ", ".join(f"{p['id']} ({p['title']})" for p in self.plays.values())
        films = ", ".join(f"{f['id']} ({f['title']}, {f['date']})" for f in self.films.values())
        return f"Play ids: {plays}.\nFilm ids: {films}."

    # ---------- tools ----------

    def fetch_play(self, play_id: str) -> dict:
        play = self.plays.get(play_id) or self._play_by_title(play_id)
        if not play:
            return {"found": False, "error": f"No play '{play_id}' in the playbook."}
        return {
            "found": True,
            "citation": {"kind": "play", "label": play["title"], "href": play["href"]},
            **play,
        }

    def _play_by_title(self, text: str) -> dict | None:
        key = text.strip().lower()
        return next((p for p in self.plays.values() if p["title"].lower() == key), None)

    def fetch_clip(self, film_id: str, play_id: str | None = None) -> dict:
        film = self.films.get(film_id)
        if not film:
            return {"found": False, "error": f"No film '{film_id}'."}
        rows = film["possessions"]
        if play_id:
            rows = [r for r in rows if r["playId"] == play_id]
        summary = by_set_and_defense(rows)
        return {
            "found": True,
            "film": {
                k: film[k]
                for k in ("id", "title", "opponent", "opponentDefense", "date", "result", "offense")
            },
            "citation": {"kind": "clip", "label": f"{film['title']} ({film['date']})", "href": film["href"]},
            "bySetAndDefense": summary,
            "possessions": [
                {**r, "citation": {"kind": "clip", "label": _clip_label(film, r), "href": r["href"]}}
                for r in rows[:MAX_POSSESSIONS]
            ],
            "possessionsShown": min(len(rows), MAX_POSSESSIONS),
            "possessionsTotal": len(rows),
        }

    def fetch_stat(self, query: str) -> dict:
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        scored = []
        for s in self.stats:
            hay = f"{s['id']} {s['label']}".lower()
            score = sum(1 for w in words if w in hay)
            if score:
                scored.append((score, s))
        scored.sort(key=lambda t: -t[0])
        hits = [
            {**s, "citation": {"kind": "stat", "label": s["label"], "href": s["href"]}}
            for _, s in scored[:MAX_STATS]
        ]
        return {"found": bool(hits), "stats": hits}


def _clip_label(film: dict, row: dict) -> str:
    set_name = row["set"] or "Possession"
    return f"{film['title']}: {set_name} vs {row['defense']}, {row['outcome']}"


def by_set_and_defense(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["tag"] != "confirmed":
            continue
        groups.setdefault((r["set"], r["defense"]), []).append(r)
    out = []
    for (set_name, defense), items in groups.items():
        points = sum(i["points"] for i in items)
        out.append(
            {
                "set": set_name,
                "playId": items[0]["playId"],
                "defense": defense,
                "runs": len(items),
                "points": points,
                "pointsPerPossession": round(points / len(items), 2),
                "scored": sum(1 for i in items if i["points"] > 0),
            }
        )
    return sorted(out, key=lambda g: (g["defense"], -g["runs"]))


@lru_cache(maxsize=1)
def load_world(path: Path = settings.WORLD_PATH) -> World:
    return World(json.loads(path.read_text(encoding="utf-8")))
