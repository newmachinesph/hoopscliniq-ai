import json
from types import SimpleNamespace

from app.naming import suggest_name

PLAYBOOK = [{"id": "horns", "title": "Horns", "description": "two bigs at the elbows"}]
PLAY = {
    "source": {"fps": 12, "frames": 24},
    "tracks": {"players": {}},
    "players": {},
    "events": [],
    "meta": {},
}
RAW = {"playId": "horns", "title": "Horns", "confidence": "low", "why": "Two players at the elbows."}


class FakeClient:
    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=json.dumps(RAW))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_default_naming_sends_no_reasoning_effort():
    client = FakeClient()
    assert suggest_name(client, PLAYBOOK, PLAY)["playId"] == "horns"
    assert "reasoning_effort" not in client.kwargs


def test_naming_passes_the_reasoning_effort_when_given():
    client = FakeClient()
    suggest_name(client, PLAYBOOK, PLAY, reasoning_effort="medium")
    assert client.kwargs["reasoning_effort"] == "medium"
