import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.api import create_app
from app.camera import CameraMotion
from app.geometry import check_quad
from app.jobs import JobStore, build_outputs, calibration_record, new_job_id
from app.sam import parse_sam_text
from tests.test_sam_parse import fixture_text

# Half-court corners on a 1280x720 clip, in calibration order; the near baseline corner is off-screen.
COACH = [[838, 281], [1620, 495], [258, 652], [271, 305]]


def no_colors(*_args):
    raise AssertionError("calibrate must reuse the stored team labels")


@pytest.fixture
def ctx(tmp_path):
    store = JobStore(tmp_path)
    named: list[dict] = []

    def name_play(play: dict) -> dict:
        named.append(play)
        return {
            "playId": "unknown",
            "title": "Unnamed play",
            "confidence": "low",
            "why": "w",
            "needsReview": True,
        }

    app = create_app(
        store=store,
        start_job=lambda j: None,
        ask=lambda q: {},
        models_reachable=lambda: True,
        probe_duration=lambda p: 20.0,
        name_play=name_play,
    )
    return TestClient(app), store, named


def finished_job(store: JobStore, status: str = "done", camera: CameraMotion | None = None) -> str:
    """A job as the pipeline leaves it: sam.txt, play.json, overlay.json, calibration.json, team labels."""
    jid = new_job_id()
    sam_text = fixture_text()
    job = {
        "jobId": jid,
        "status": status,
        "finishedAt": "2026-10-28T10:00:00Z",
        "suggestedName": None,
        "error": None,
        "calibration": None,
        "video": {"fps": 12, "frames": 2, "width": 1280, "height": 720, "duration": 0.2},
        "teamLabels": {str(o): o % 2 for o in range(20)},
    }
    store.dir(jid).mkdir(parents=True)
    (store.dir(jid) / "sam.txt").write_text(sam_text, encoding="utf-8")
    if camera:
        store.write_camera(jid, camera)
    cal = calibration_record(job)
    play, overlay, _ = build_outputs(job, parse_sam_text(sam_text), None, camera, cal, no_colors)
    store.write_calibration(jid, cal)
    store.write_play(jid, play)
    store.write_overlay(jid, overlay)
    store.write(job)
    return jid


def test_calibrate_reprojects_without_sam_and_renames(ctx, monkeypatch):
    client, store, named = ctx
    monkeypatch.setattr(jobs, "stream_sam", lambda *a, **k: pytest.fail("no SAM call on calibrate"))
    monkeypatch.setattr(jobs, "upload_clip", lambda *a, **k: pytest.fail("no upload on calibrate"))
    jid = finished_job(store)
    before = client.get(f"/film/jobs/{jid}").json()
    assert before["calibration"]["source"] == "estimated" and before["calibration"]["frame"] == 0
    r = client.post(f"/film/jobs/{jid}/calibrate", json={"frame": 1, "points": COACH})
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["calibration"] == {
        "source": "coach",
        "frame": 1,
        "points": [[float(x), float(y)] for x, y in COACH],
    }
    assert view["playDna"]["meta"]["calibration"] == "coach"
    assert view["playDna"]["meta"]["calibrationFrame"] == 1
    assert view["suggestedName"]["title"] == "Unnamed play" and len(named) == 1
    assert set(view["playDna"]["players"]) == set(before["playDna"]["players"])
    pid = next(iter(before["playDna"]["players"]))
    assert view["playDna"]["tracks"]["players"][pid] != before["playDna"]["tracks"]["players"][pid]
    assert view["overlay"]["players"] == before["overlay"]["players"]  # same boxes, same ids
    assert "calibrate" in view["playDna"]["meta"]["timings"]
    assert client.get(f"/film/jobs/{jid}").json()["calibration"]["source"] == "coach"
    assert store.read_calibration(jid)["points"] == view["calibration"]["points"]


def test_calibrate_uses_the_stored_camera_motion(ctx):
    client, store, _ = ctx
    camera = CameraMotion([np.eye(3), np.array([[1.0, 0, 30], [0, 1, 0], [0, 0, 1]])], (1280, 720))
    jid = finished_job(store, camera=camera)
    assert client.get(f"/film/jobs/{jid}").json()["cameraMotion"] == {"framesTracked": 2, "framesFallback": 0}
    view = client.post(f"/film/jobs/{jid}/calibrate", json={"frame": 0, "points": COACH}).json()
    assert view["cameraMotion"] == {"framesTracked": 2, "framesFallback": 0}


@pytest.mark.parametrize(
    "body",
    [
        None,
        {"frame": 0},
        {"frame": 0, "points": COACH[:3]},
        {"frame": 2, "points": COACH},  # the clip has frames 0 and 1
        {"frame": -1, "points": COACH},
        {"frame": True, "points": COACH},
        {"frame": 0, "points": [[0, 0], [1280, 0], [1280, 720], [0, "x"]]},
        {"frame": 0, "points": [[0, 0], [1280, 720], [1280, 0], [0, 720]]},  # crossed: not convex
        {"frame": 0, "points": [[0, 0], [5000, 0], [1280, 720], [0, 720]]},  # far outside the image
    ],
)
def test_calibrate_rejects_bad_input(ctx, body):
    client, store, named = ctx
    jid = finished_job(store)
    r = client.post(f"/film/jobs/{jid}/calibrate", json=body)
    assert r.status_code == 400 and r.json()["error"].startswith("Bad calibration")
    assert named == []


def test_calibrate_404_409_and_malformed_id(ctx):
    client, store, _ = ctx
    assert (
        client.post(f"/film/jobs/{new_job_id()}/calibrate", json={"frame": 0, "points": COACH}).status_code
        == 404
    )
    assert client.post("/film/jobs/nope/calibrate", json={"frame": 0, "points": COACH}).status_code == 400
    jid = finished_job(store, status="name")
    r = client.post(f"/film/jobs/{jid}/calibrate", json={"frame": 0, "points": COACH})
    assert r.status_code == 409 and r.headers["x-demo"] == "live"


def test_check_quad():
    check_quad([(0, 0), (100, 0), (100, 50), (0, 50)], (100, 50))
    check_quad([(0, 0), (0, 50), (100, 50), (100, 0)], (100, 50))  # either winding
    check_quad([(838, 281), (1620, 495), (258, 652), (271, 305)], (1280, 720))  # one corner off-screen
    with pytest.raises(ValueError):
        check_quad([(0, 0), (100, 50), (100, 0), (0, 50)], (100, 50))
    with pytest.raises(ValueError):
        check_quad([(0, 0), (50, 0), (100, 0), (0, 50)], (100, 50))  # three on a line
    with pytest.raises(ValueError):
        check_quad([(0, 0), (100, 0), (100, 50), (-80, 50)], (100, 50))


def test_estimated_calibration_is_in_clip_pixels():
    """The Hazen estimate is authored at 1920x1080; the job view and calibrate both use clip.mp4 pixels."""
    job = {"calibration": None, "video": {"frames": 2, "width": 1280, "height": 720}}
    record = calibration_record(job)
    assert record["imageSize"] == [1280, 720] and record["frame"] == 0
    # 2133 x 1280/1920 = 1422: the baseline-right corner is off-screen in the source frame too (extrapolated),
    # so it stays past the right edge after scaling; it is inside the calibrate margin.
    assert record["points"] == [[850.0, 285.0], [1422.0, 505.0], [265.0, 641.0], [272.0, 302.0]]
    check_quad([tuple(p) for p in record["points"]], (1280, 720))
    half = calibration_record({"calibration": None, "video": {"frames": 2, "width": 640, "height": 360}})
    assert half["points"][1] == [711.0, 252.5]


def test_slow_naming_keeps_the_previous_name(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr("app.settings.CALIBRATE_NAME_TIMEOUT", 0.05)
    store = JobStore(tmp_path)
    jid = finished_job(store)
    job = store.read(jid)
    job["suggestedName"] = {
        "playId": "x",
        "title": "Old",
        "confidence": "low",
        "why": "w",
        "needsReview": True,
    }

    def slow(play):
        time.sleep(0.5)
        return {"playId": "y", "title": "New", "confidence": "low", "why": "w", "needsReview": True}

    done = jobs.recalibrate(store, job, 0, COACH, slow)
    assert done["suggestedName"]["title"] == "Old"
    assert done["timings"]["calibrateName"] < 0.4
