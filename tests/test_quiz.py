import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import settings
from app.api import create_app
from app.jobs import JobStore
from app.models import QuizRequest
from app.quiz import QuizError, build_questions, describe_play, quiz_messages, write_quiz, zone

FRAME_0 = {
    "offense": [[250, 300], [80, 260], [420, 260], [190, 190], [310, 190]],
    "ball": 0,
    "actions": [
        {"kind": "pass", "from": [250, 300], "to": [80, 260]},
        {"kind": "screen", "from": [310, 190], "to": [250, 300]},
    ],
    "note": "Five sets the screen.",
}
FRAME_1 = {
    "offense": [[250, 300], [80, 260], [420, 260], [190, 190], [310, 190]],
    "ball": 1,
    "actions": [{"kind": "cut", "from": [80, 260], "to": [30, 60]}],
}
PLAY = {"id": "horns", "title": "Horns", "frames": [FRAME_0, FRAME_1]}


def question(**over):
    q = {
        "prompt": "Who gets the ball first?",
        "frame": 1,
        "answerFrame": 2,
        "options": ["Player 1", "Player 2", "Player 3"],
        "correct": 1,
        "why": "Frame 1 shows 1 passing to 2.",
    }
    return {**q, **over}


class FakeClient:
    def __init__(self, questions=None, content=None, error=None) -> None:
        self.kwargs: dict = {}
        self.payload = content if content is not None else json.dumps({"questions": questions})
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def request(**over):
    return QuizRequest.model_validate({"play": PLAY, **over})


@pytest.mark.parametrize(
    ("point", "name"),
    [
        ((20, 30), "left corner"),
        ((480, 30), "right corner"),
        ((250, 52), "rim"),
        ((200, 90), "left low post"),
        ((300, 130), "right mid post"),
        ((250, 170), "high post"),
        ((190, 190), "left elbow"),
        ((310, 190), "right elbow"),
        ((120, 100), "left baseline"),
        ((250, 230), "high post"),
        ((350, 230), "right elbow"),
        ((450, 200), "right wing"),
        ((80, 260), "left wing"),
        ((250, 300), "top of the key"),
        ((250, 400), "half court"),
    ],
)
def test_zone_names(point, name):
    assert zone(*point) == name


def test_describe_play_in_words():
    text = describe_play(request())
    assert 'Play "Horns", 2 frames' in text
    assert "Frame 1: 1 at the top of the key, 2 at the left wing" in text
    assert "Player 1 has the ball." in text
    assert "1 passes to 2" in text
    assert "5 screens for 1 at the right elbow" in text
    assert "Coach's note: Five sets the screen." in text
    assert "Frame 2:" in text and "Player 2 has the ball." in text
    assert "2 cuts to the left corner" in text


def test_messages_name_the_spot_and_count():
    system, user = quiz_messages(request(spot=4, count=5))
    assert "spot 4" in system["content"] and "Never invent" in system["content"]
    assert "5 questions" in user["content"] and "spot 4" in user["content"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"play": {"id": "", "title": "t", "frames": [FRAME_0]}},
        {"play": {"id": "x" * 81, "title": "t", "frames": [FRAME_0]}},
        {"play": {"id": "p", "title": "   ", "frames": [FRAME_0]}},
        {"play": {"id": "p", "title": "t", "frames": []}},
        {"play": {"id": "p", "title": "t", "frames": [FRAME_0] * 13}},
        {"play": {"id": "p", "title": "t", "frames": [{**FRAME_0, "offense": [[1, 1]] * 4}]}},
        {"play": {"id": "p", "title": "t", "frames": [{**FRAME_0, "offense": [[900, 1]] * 5}]}},
        {"play": {"id": "p", "title": "t", "frames": [{**FRAME_0, "ball": 5}]}},
        {"play": {"id": "p", "title": "t", "frames": [{**FRAME_0, "actions": FRAME_0["actions"] * 5}]}},
        {"play": {"id": "p", "title": "t", "frames": [{**FRAME_0, "note": "n" * 201}]}},
        {"play": PLAY, "spot": 0},
        {"play": PLAY, "spot": 6},
        {"play": PLAY, "count": 1},
        {"play": PLAY, "count": 6},
    ],
)
def test_request_validation_rejects(body):
    with pytest.raises(ValidationError):
        QuizRequest.model_validate(body)


def test_request_defaults_and_extra_fields():
    req = QuizRequest.model_validate({"play": {**PLAY, "section": "Half court", "tags": ["a"]}})
    assert (req.spot, req.count) == (2, 3)
    assert req.play.frames[0].actions[0].from_ == (250, 300)


def test_invalid_questions_are_dropped():
    req = request(count=5)
    bad = [
        question(frame=3),  # outside the play
        question(frame=0),  # the model counts frames from 1
        question(answerFrame=3),
        question(frame=2, answerFrame=1),  # answer before the question frame
        question(options=["a", "b"]),
        question(options=["a", "a", "b"]),
        question(options=["a", " ", "b"]),
        question(correct=3),
        question(correct=True),
        question(prompt="  "),
        "text",
    ]
    assert build_questions({"questions": [*bad, question()]}, req)[0]["id"] == "q1"
    assert len(build_questions({"questions": [*bad, question()]}, req)) == 1
    assert build_questions({"questions": "x"}, req) == []
    assert build_questions([], req) == []


def test_text_is_trimmed_and_count_caps():
    req = request(count=2)
    long = question(prompt="p" * 300, why="w" * 300, options=["o" * 200, "b", "c"])
    qs = build_questions({"questions": [long, question(), question(), question()]}, req)
    assert len(qs) == 2
    assert len(qs[0]["prompt"]) == 200 and qs[0]["prompt"].endswith("…")
    assert len(qs[0]["why"]) == 220 and qs[0]["why"].endswith("…")
    assert all(len(o["text"]) <= 120 for o in qs[0]["options"])


def test_shuffle_is_deterministic_and_keeps_the_answer():
    req = request(count=5)
    raw = {"questions": [question() for _ in range(5)]}
    first = build_questions(raw, req)
    assert first == build_questions(raw, req)
    for q in first:
        assert [o["id"] for o in q["options"]] == ["a", "b", "c"]
        assert sorted(o["text"] for o in q["options"]) == ["Player 1", "Player 2", "Player 3"]
        assert {o["id"]: o["text"] for o in q["options"]}[q["correctId"]] == "Player 2"
        assert q["playId"] == "horns"
        assert (q["frame"], q["answerFrame"]) == (0, 1)  # 1-based from the model, 0-based in the response
    assert [q["id"] for q in first] == ["q1", "q2", "q3", "q4", "q5"]
    orders = {tuple(o["text"] for o in q["options"]) for q in first}
    assert len(orders) > 1  # the seed differs per question


def test_write_quiz_builds_the_response_and_calls_spark_strictly():
    client = FakeClient([question(), question(prompt="Where do you go?")])
    out = write_quiz(client, request(spot=3))
    assert client.kwargs["model"] == settings.SPARK_MODEL
    assert client.kwargs["reasoning_effort"] == "low"
    schema = client.kwargs["response_format"]
    assert schema["type"] == "json_schema" and schema["json_schema"]["strict"] is True
    quiz = out["quiz"]
    assert quiz["id"].startswith("gen-horns-") and len(quiz["id"]) == len("gen-horns-") + 6
    assert quiz["title"] == "Horns: spot 3" and quiz["topic"] == "Written by ClinIQ"
    assert quiz["due"] is None and quiz["spot"] == 3 and len(quiz["questions"]) == 2
    assert out["source"] == {"playId": "horns", "title": "Horns", "frames": 2}
    assert out["model"] == "muse-spark-1.3" and isinstance(out["seconds"], float)


@pytest.mark.parametrize(
    ("client", "status"),
    [
        (FakeClient(error=TimeoutError("slow")), 504),
        (FakeClient(error=RuntimeError("api down")), 502),
        (FakeClient(content="not json"), 502),
        (FakeClient([question()]), 502),  # one valid question is not enough
        (FakeClient([question(frame=9), question(frame=9)]), 502),
    ],
)
def test_write_quiz_errors_are_friendly(client, status):
    with pytest.raises(QuizError) as err:
        write_quiz(client, request())
    assert err.value.status == status
    assert "api down" not in err.value.message and "slow" not in err.value.message


def api(make_quiz):
    app = create_app(
        store=JobStore(__import__("pathlib").Path(".")),
        start_job=lambda j: None,
        ask=lambda q: {},
        models_reachable=lambda: True,
        probe_duration=lambda p: 1.0,
        make_quiz=make_quiz,
    )
    return TestClient(app)


def test_quiz_route_400_on_bad_input():
    client = api(lambda req: {})
    for body in ({}, {"play": {"id": "p"}}, {"play": PLAY, "spot": 9}):
        r = client.post("/study/quiz", json=body)
        assert r.status_code == 400 and "error" in r.json() and r.headers["x-demo"] == "live"
    assert client.post("/study/quiz", content=b"not json").status_code == 400


def test_quiz_route_passes_the_result_and_maps_errors():
    seen = []

    def ok(req):
        seen.append(req)
        return {"quiz": {"id": "gen-horns-abcdef"}}

    r = api(ok).post("/study/quiz", json={"play": PLAY, "spot": 4})
    assert r.status_code == 200 and r.json() == {"quiz": {"id": "gen-horns-abcdef"}}
    assert (seen[0].spot, seen[0].count) == (4, 3)

    def fail(status):
        def raise_it(req):
            raise QuizError(status, "friendly")

        return raise_it

    for status in (502, 504):
        r = api(fail(status)).post("/study/quiz", json={"play": PLAY})
        assert r.status_code == status and r.json() == {"error": "friendly"}
    assert api(None).post("/study/quiz", json={"play": PLAY}).status_code == 503
