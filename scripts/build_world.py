"""Build data/world.json from the demo's mocks so ids, titles and numbers match the demo app exactly.

The demo's mocks are TypeScript with seeded generation (film possessions, play stats derived from them),
so instead of retyping them we run them once with Node's type stripping and keep only what ClinIQ needs.

    python scripts/build_world.py [--demo ../hoopscliniq-demo]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "world.json"

DUMP_TS = """
import { films } from "./film.ts"
import { plays } from "./plays.ts"
import { gameLog } from "./insights.ts"
import { opponents } from "./season.ts"
console.log(JSON.stringify({
  plays: plays.map((p) => ({ id: p.id, title: p.title, category: p.category, summary: p.summary, vs: p.vs,
    stats: p.stats, clips: p.clips, gamePlan: p.gamePlan ?? null })),
  films: films.map((f) => ({ id: f.id, kind: f.kind, title: f.title, date: f.date, us: f.us, them: f.them,
    final: f.final, possessions: f.possessions.map((p) => ({ id: p.id, side: p.side, start: p.start,
      set: p.set,
      playId: p.playId ?? null, result: p.result, points: p.points, defense: p.defense, note: p.note ?? null,
      tag: p.tag?.status ?? null })) })),
  gameLog, opponents,
}))
"""


def dump_mocks(demo: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for src in (demo / "src" / "mocks").glob("*.ts"):
            text = src.read_text(encoding="utf-8")
            # Node's ESM loader needs explicit extensions; "@/..." imports are type-only and get stripped.
            text = re.sub(r'from "\./([a-z0-9-]+)"', r'from "./\1.ts"', text)
            (work / src.name).write_text(text, encoding="utf-8")
        (work / "package.json").write_text('{"type": "module"}', encoding="utf-8")
        (work / "dump.ts").write_text(DUMP_TS, encoding="utf-8")
        node = shutil.which("node") or "node"
        out = subprocess.run([node, "dump.ts"], cwd=work, capture_output=True, text=True, check=True)
        return json.loads(out.stdout)


def clip_href(film_id: str, t: float) -> str:
    return f"/film/{film_id}?t={int(t)}"


def ppp(rows: list[dict]) -> float:
    return round(sum(r["points"] for r in rows) / len(rows), 2) if rows else 0.0


def build_stats(raw: dict, our_possessions: list[dict]) -> list[dict]:
    log = raw["gameLog"]
    games = len(log)
    wins = sum(1 for g in log if g["win"])
    us_pts = sum(g["us"] for g in log)
    them_pts = sum(g["them"] for g in log)
    poss = sum(g["poss"] for g in log)
    tov = sum(g["tov"] for g in log)
    opp3m = sum(g["opp3m"] for g in log)
    opp3a = sum(g["opp3a"] for g in log)
    confirmed = [p for p in our_possessions if p["tag"] == "confirmed"]
    zone = [p for p in confirmed if "zone" in p["defense"].lower()]
    man = [p for p in confirmed if "man" in p["defense"].lower() and "zone" not in p["defense"].lower()]
    press = [p for p in confirmed if "press" in p["defense"].lower()]
    cedar = next(g for g in log if g["opponentId"] == "cedar-park" and g.get("filmId"))
    href = "/study/results"
    rows = [
        ("season-record", "Season record", f"{wins}-{games - wins}"),
        ("points-per-game", "Points per game", round(us_pts / games, 1)),
        ("opp-points-per-game", "Opponent points per game", round(them_pts / games, 1)),
        ("offensive-rating", "Offensive rating (points per 100 possessions)", round(us_pts / poss * 100, 1)),
        (
            "defensive-rating",
            "Defensive rating (opponent points per 100 possessions)",
            round(them_pts / poss * 100, 1),
        ),
        ("turnovers-per-game", "Turnovers per game", round(tov / games, 1)),
        ("opp-3pt-pct", "Opponent three-point percentage", f"{round(opp3m / opp3a * 100, 1)}%"),
        ("ppp-vs-zone", "Points per possession vs zone (coach-tagged film)", ppp(zone)),
        ("ppp-vs-man", "Points per possession vs man (coach-tagged film)", ppp(man)),
        ("ppp-vs-press", "Points per possession vs press (coach-tagged film)", ppp(press)),
        ("cedar-park-turnovers", f"Turnovers in the Cedar Park loss ({cedar['date'][:10]})", cedar["tov"]),
        ("tagged-possessions", "Our offensive possessions tagged on game film", len(confirmed)),
    ]
    return [{"id": i, "label": label, "value": value, "href": href} for i, label, value in rows]


def build_world(raw: dict) -> dict:
    opponents = {o["name"]: o for o in raw["opponents"]}
    plays = [
        {
            "id": p["id"],
            "title": p["title"],
            "description": p["summary"],
            "category": p["category"],
            "vs": p["vs"],
            "timesRun": p["stats"]["timesRun"],
            "pointsPerPossession": p["stats"]["ppp"],
            "successRate": p["stats"]["successRate"],
            "inWestviewGamePlan": p["gamePlan"] == "westview",
            "href": f"/plays/{p['id']}",
            "clips": [
                {
                    "filmId": c["filmId"],
                    "label": f"{c['clock']} {c['label']}",
                    "href": clip_href(c["filmId"], c["t"]),
                }
                for c in p["clips"]
            ],
        }
        for p in raw["plays"]
    ]
    films = []
    ours: list[dict] = []
    for f in raw["films"]:
        scout = f["kind"] == "scout"
        possessions = [
            {
                "clipId": p["id"],
                "playId": p["playId"],
                "set": p["set"],
                "defense": p["defense"],
                "outcome": p["result"],
                "points": p["points"],
                "note": p["note"],
                "tag": p["tag"],
                "href": clip_href(f["id"], p["start"]),
            }
            for p in f["possessions"]
            if p["side"] == "us"
        ]
        if not scout:
            ours.extend(possessions)
        us, them = f["final"]["us"], f["final"]["them"]
        result = (
            f"{f['us']['name']} {us}, {f['them']['name']} {them} (scout film)"
            if scout
            else f"{'W' if us > them else 'L'} {us}-{them}"
        )
        # In scout film the tagged side ("us") is the scouted opponent.
        opponent = f["us"]["name"] if scout else f["them"]["name"]
        films.append(
            {
                "id": f["id"],
                "kind": f["kind"],
                "title": f["title"],
                "opponent": opponent,
                "opponentDefense": opponents.get(opponent, {}).get("defense"),
                "date": f["date"][:10],
                "result": result,
                "offense": f["us"]["name"],
                "href": f"/film/{f['id']}",
                "possessions": possessions,
            }
        )
    return {
        "about": "Fictional demo world (Lakeshore varsity boys), generated from hoopscliniq-demo/src/mocks "
        "by scripts/build_world.py. Ids match the demo app's routes.",
        "team": "Lakeshore",
        "opponents": [{"id": o["id"], "name": o["name"], "defense": o["defense"]} for o in raw["opponents"]],
        "plays": plays,
        "films": films,
        "stats": build_stats(raw, ours),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=ROOT.parent / "hoopscliniq-demo")
    args = parser.parse_args()
    world = build_world(dump_mocks(args.demo))
    OUT.write_text(json.dumps(world, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT}: {len(world['plays'])} plays, {len(world['films'])} films, {len(world['stats'])} stats"
    )


if __name__ == "__main__":
    main()
