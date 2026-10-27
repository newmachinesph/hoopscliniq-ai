import json
from types import SimpleNamespace

from app import cliniq
from app.cliniq import NO_SOURCE, ask, postcheck
from app.world import load_world

WORLD = load_world()


def test_world_ids_match_the_demo_routes():
    assert "/plays/horns-twist" in WORLD.hrefs
    assert "/film/cedar-park" in WORLD.hrefs
    assert "/film/westview" in WORLD.hrefs
    assert "/study/results" in WORLD.hrefs
    assert "/plays/not-a-play" not in WORLD.hrefs


def test_postcheck_keeps_valid_citations():
    raw = {
        "body": ["Zone Gut scored against the 2-3 [1].", "The film shows it [2]."],
        "citations": [
            {"kind": "play", "label": "Zone Gut", "href": "/plays/zone-gut"},
            {"kind": "clip", "label": "@ Westview", "href": "/film/westview"},
        ],
        "confidence": "medium",
    }
    answer, missing = postcheck(raw, WORLD)
    assert missing is False
    assert answer["credits"] == 1
    assert answer["confidence"] == "medium"
    assert [c["href"] for c in answer["citations"]] == ["/plays/zone-gut", "/film/westview"]


def test_postcheck_drops_unknown_hrefs_and_renumbers():
    raw = {
        "body": ["Made-up source [1], real one [2]."],
        "citations": [
            {"kind": "clip", "label": "Fake", "href": "/film/nowhere"},
            {"kind": "play", "label": "Zone Gut", "href": "/plays/zone-gut"},
        ],
        "confidence": "high",
    }
    answer, missing = postcheck(raw, WORLD)
    assert missing is False
    assert answer["body"] == ["Made-up source , real one [1]."]
    assert answer["citations"] == [{"kind": "play", "label": "Zone Gut", "href": "/plays/zone-gut"}]


def test_postcheck_paragraph_without_marker_means_no_source():
    raw = {
        "body": ["Cited [1].", "Uncited claim."],
        "citations": [{"kind": "play", "label": "Zone Gut", "href": "/plays/zone-gut"}],
        "confidence": "high",
    }
    answer, missing = postcheck(raw, WORLD)
    assert missing is True
    assert answer == {"body": [NO_SOURCE], "citations": [], "confidence": None, "credits": 0}


def test_postcheck_marker_only_pointing_at_a_dropped_citation_means_no_source():
    raw = {"body": ["Claim [1]."], "citations": [{"kind": "play", "label": "x", "href": "/plays/fake"}]}
    assert postcheck(raw, WORLD)[1] is True


def test_postcheck_empty_citations_means_no_source():
    answer, missing = postcheck({"body": ["I have no source for that in your film."], "citations": []}, WORLD)
    assert missing is True and answer["credits"] == 0


def test_postcheck_dedupes_same_href():
    raw = {
        "body": ["A [1] and B [2]."],
        "citations": [
            {"kind": "play", "label": "Zone Gut", "href": "/plays/zone-gut"},
            {"kind": "play", "label": "Zone Gut again", "href": "/plays/zone-gut"},
        ],
    }
    answer, missing = postcheck(raw, WORLD)
    assert not missing
    assert answer["body"] == ["A [1] and B [1]."]
    assert len(answer["citations"]) == 1


# ---------- tool loop with a fake client ----------


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


def _call(i, name, args):
    return SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)


def test_tool_loop_then_structured_final_answer():
    final = {
        "body": ["Short Corner Flood was in the plan against the 2-3 [1]."],
        "citations": [{"kind": "play", "label": "Short Corner Flood", "href": "/plays/short-corner-flood"}],
        "confidence": "medium",
    }
    client = FakeClient(
        [
            _msg(tool_calls=[_call(1, "fetch_play", {"playId": "short-corner-flood"})]),
            _msg(content="done"),
            _msg(content=json.dumps(final)),
        ]
    )
    out = ask(client, "muse-spark-1.3", WORLD, "What works against the zone?")
    assert out["noSource"] is False
    assert out["answer"]["citations"][0]["href"] == "/plays/short-corner-flood"
    tool_msg = client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["found"] is True
    assert client.requests[2]["response_format"]["type"] == "json_schema"


def test_tool_loop_stops_at_the_budget():
    calls = [_msg(tool_calls=[_call(i, "fetch_stat", {"query": "turnovers"})]) for i in range(10)]
    client = FakeClient(
        [*calls[: cliniq.settings.MAX_TOOL_CALLS], _msg(content=json.dumps({"body": [], "citations": []}))]
    )
    out = ask(client, "m", WORLD, "turnovers?")
    assert len(client.requests) == cliniq.settings.MAX_TOOL_CALLS + 1
    assert out["noSource"] is True


def test_tools_return_citations_from_the_world():
    clip = WORLD.fetch_clip("westview", "zone-gut")
    assert clip["found"] and all(p["playId"] == "zone-gut" for p in clip["possessions"])
    assert all(p["citation"]["href"] in WORLD.hrefs for p in clip["possessions"])
    assert WORLD.fetch_play("Horns Twist")["href"] == "/plays/horns-twist"
    assert WORLD.fetch_play("nope")["found"] is False
    assert WORLD.fetch_stat("turnovers per game")["stats"][0]["id"] == "turnovers-per-game"
    assert WORLD.fetch_stat("zzz qqq")["found"] is False
