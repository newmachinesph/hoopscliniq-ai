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
    assert set(job) == {
        "jobId",
        "status",
        "finishedAt",
        "steps",
        "suggestedName",
        "playDna",
        "error",
        "clipUrl",
        "overlay",
        "calibration",
        "cameraMotion",
    }
    assert job["clipUrl"] is None and job["overlay"] is None
    assert job["calibration"] is None and job["cameraMotion"] is None


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


def done_job(store, job_id, finished_at, players=7, status="done"):
    store.write(
        {
            "jobId": job_id,
            "status": status,
            "createdAt": finished_at,
            "finishedAt": finished_at,
            "suggestedName": {
                "playId": "unknown",
                "title": f"Play {job_id[-2:]}",
                "confidence": "low",
                "why": "w",
                "needsReview": True,
            },
            "error": None,
            "video": {"fps": 12, "frames": 2, "width": 40, "height": 20, "duration": 0.2},
        }
    )
    store.write_play(job_id, {"meta": {"playersFound": players}})


def test_job_list_is_done_only_newest_first_max_ten(ctx):
    client, store, _ = ctx
    ids = [f"job-aaaaaaaaaa{c}{d}" for c in "bc" for d in "234567"]  # 12 ids
    for n, jid in enumerate(ids):
        done_job(store, jid, f"2026-10-28T10:{n:02d}:00Z", players=n)
    done_job(store, "job-zzzzzzzzzzzz", "2026-10-28T11:00:00Z", status="failed")
    store.write({"jobId": "job-yyyyyyyyyyyy", "status": "track", "finishedAt": None})
    r = client.get("/film/jobs")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    jobs = r.json()["jobs"]
    assert [j["jobId"] for j in jobs] == ids[::-1][:10]
    assert jobs[0] == {
        "jobId": ids[-1],
        "finishedAt": "2026-10-28T10:11:00Z",
        "title": f"Play {ids[-1][-2:]}",
        "playersFound": 11,
    }


def test_job_list_empty(ctx):
    client, _, _ = ctx
    assert client.get("/film/jobs").json() == {"jobs": []}


def test_job_view_has_absolute_clip_url_and_overlay(ctx):
    client, store, _ = ctx
    jid = new_job_id()
    done_job(store, jid, "2026-10-28T10:00:00Z")
    overlay = {
        "fps": 12,
        "frames": 2,
        "imageSize": [40, 20],
        "players": {"o1": {"team": "offense", "boxes": [[1, 2, 3, 4], None]}},
    }
    store.write_overlay(jid, overlay)
    job = client.get(f"/film/jobs/{jid}").json()
    assert job["clipUrl"] == f"http://testserver/film/jobs/{jid}/clip"
    assert job["overlay"] == {**overlay, "ball": None}  # jobs from before 0.3.0 have no ball


def test_clip_404_when_missing(ctx):
    client, store, _ = ctx
    jid = new_job_id()
    done_job(store, jid, "2026-10-28T10:00:00Z")
    r = client.get(f"/film/jobs/{jid}/clip")
    assert r.status_code == 404 and "error" in r.json()
    assert client.get(f"/film/jobs/{new_job_id()}/clip").status_code == 404
    assert client.get("/film/jobs/nope/clip").status_code == 400


def test_clip_serves_ranges(ctx):
    client, store, _ = ctx
    jid = new_job_id()
    data = bytes(range(256)) * 4
    store.dir(jid).mkdir(parents=True)
    (store.dir(jid) / "clip.mp4").write_bytes(data)
    full = client.get(f"/film/jobs/{jid}/clip")
    assert full.status_code == 200 and full.content == data
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["accept-ranges"] == "bytes"
    r = client.get(f"/film/jobs/{jid}/clip", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 0-99/{len(data)}"
    assert r.content == data[:100]
    assert r.headers["cache-control"] == "public, max-age=3600"
    assert r.headers["access-control-allow-origin"] == "*"
    assert r.headers["x-demo"] == "live"


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


def test_store_read_looks_again_when_job_file_is_briefly_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.JOB_READ_RETRY_SECONDS", 0.0)
    reloads: list[int] = []
    store = JobStore(tmp_path)
    jid = new_job_id()

    def reload() -> None:
        reloads.append(1)
        if len(reloads) == 2:  # the file shows up on the second look
            store.write({"jobId": jid, "status": "track"})

    store.reload = reload
    assert store.read(jid)["status"] == "track"
    assert len(reloads) == 2
