import json

import pytest
from fastapi.testclient import TestClient

from app.api import ModelsCheck, create_app
from app.jobs import JobStore, new_job_id, steps_for


@pytest.fixture
def ctx(tmp_path):
    started: list[str] = []
    store = JobStore(tmp_path)
    app = create_app(
        store=store,
        start_job=started.append,
        ask=lambda q: {
            "question": q,
            "answer": {"body": ["x [1]"], "citations": [], "confidence": None, "credits": 0},
            "noSource": True,
        },
        models_reachable=lambda: True,
        probe_duration=lambda p: 20.0,
    )
    return TestClient(app), store, started


def test_health_and_headers(ctx):
    client, _, _ = ctx
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["modelsReachable"] is True
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-demo"] == "live"


def test_upload_queues_a_job(ctx):
    client, store, started = ctx
    r = client.post("/film/upload", files={"file": ("clip.mp4", b"\x00" * 100, "video/mp4")})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and body["statusUrl"] == f"/film/jobs/{body['jobId']}"
    assert started == [body["jobId"]]
    assert (store.dir(body["jobId"]) / "upload.mp4").exists()
    job = client.get(body["statusUrl"]).json()
    assert job["status"] == "queued" and job["playDna"] is None and job["suggestedName"] is None
    assert [s["status"] for s in job["steps"]] == ["done", "queued", "queued", "queued", "planned", "planned"]
    assert set(job) == {"jobId", "status", "finishedAt", "steps", "suggestedName", "playDna", "error"}


def test_upload_rejects_wrong_type_and_bad_calibration(ctx):
    client, _, started = ctx
    assert client.post("/film/upload", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 400
    r = client.post(
        "/film/upload",
        files={"file": ("a.mp4", b"x", "video/mp4")},
        data={"calibration": json.dumps([[0, 0]])},
    )
    assert r.status_code == 400 and r.headers["x-demo"] == "live"
    assert started == []


def test_upload_rejects_long_video(tmp_path):
    app = create_app(
        store=JobStore(tmp_path),
        start_job=lambda j: None,
        ask=lambda q: {},
        models_reachable=lambda: False,
        probe_duration=lambda p: 75.0,
    )
    r = TestClient(app).post("/film/upload", files={"file": ("a.mov", b"x", "video/quicktime")})
    assert r.status_code == 400 and "60" in r.json()["error"]


def test_job_ids_400_and_404(ctx):
    client, _, _ = ctx
    assert client.get("/film/jobs/not-a-job").status_code == 400
    assert client.get("/film/jobs/job-ABCDEFGHIJKL").status_code == 400
    r = client.get(f"/film/jobs/{new_job_id()}")
    assert r.status_code == 404 and r.headers["cache-control"] == "no-store"


def test_ask_validation(ctx):
    client, _, _ = ctx
    assert client.post("/cliniq/ask", json={}).status_code == 400
    assert client.post("/cliniq/ask", json={"question": "   "}).status_code == 400
    assert client.post("/cliniq/ask", json={"question": "x" * 501}).status_code == 400
    assert client.post("/cliniq/ask", content=b"not json").status_code == 400
    r = client.post("/cliniq/ask", json={"question": " What worked? "})
    assert r.status_code == 200 and r.json()["question"] == "What worked?"


def test_job_id_format():
    for _ in range(50):
        jid = new_job_id()
        assert len(jid) == 16 and jid.startswith("job-")


def test_steps_for_each_status():
    status = lambda s, f=None: [x["status"] for x in steps_for(s, f)]  # noqa: E731
    assert status("frames") == ["done", "running", "queued", "queued", "planned", "planned"]
    assert status("name") == ["done", "done", "done", "running", "planned", "planned"]
    assert status("done") == ["done", "done", "done", "done", "planned", "planned"]
    assert status("failed", "track") == ["done", "done", "failed", "queued", "planned", "planned"]
    engines = [x["engine"] for x in steps_for("queued")]
    assert engines == ["Phone or file", "Modal", "SAM 3.1", "Muse Spark", "Coach confirms", "ClinIQ"]


def test_models_check_caches():
    calls = []
    check = ModelsCheck(lambda: calls.append(1) or ["sam-3.1", "muse-spark-1.3"])
    assert check() is True and check() is True
    assert len(calls) == 1
    failing = ModelsCheck(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert failing() is False
