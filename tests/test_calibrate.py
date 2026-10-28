import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.api import create_app
from app.camera import CameraMotion
from app.geometry import COURT_CORNERS, LANE_CORNERS, apply_h, check_quad, homography
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
        "landmarks": "half",
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
        {"frame": 0, "points": COACH, "landmarks": "paint"},
        {"frame": 0, "points": COACH, "landmarks": ["lane"]},
        {"frame": 0, "points": COACH, "landmarks": "lane"},  # lane corners must be on screen
        {"frame": 0, "points": [[-4, 600], [700, 600], [680, 450], [520, 450]], "landmarks": "lane"},
        {
            "frame": 0,
            "landmarks": "lane",
        },
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


# A synthetic camera: court feet -> 1280x720 clip pixels, from where the half-court corners appear on screen.
CAMERA = homography(COURT_CORNERS, [(300, 650), (1100, 690), (900, 250), (420, 240)])
LANE_PX = [[round(c, 3) for c in apply_h(CAMERA, x, y)] for x, y in LANE_CORNERS]
HALF_PX = [[round(c, 3) for c in apply_h(CAMERA, x, y)] for x, y in COURT_CORNERS]


def test_lane_homography_round_trip():
    """Lane corners seen through a known camera give back that camera's court mapping everywhere."""
    h = homography([tuple(p) for p in LANE_PX], LANE_CORNERS)
    for x, y in [(0, 0), (50, 47), (25, 5.25), (3, 30), (44, 12), (25, 19)]:
        u, v = apply_h(CAMERA, x, y)
        cx, cy = apply_h(h, u, v)
        assert abs(cx - x) < 0.01 and abs(cy - y) < 0.01
    # Lane corners land on the NFHS lane: 12 ft wide, free-throw line 19 ft out.
    assert LANE_CORNERS == [(19.0, 0.0), (31.0, 0.0), (31.0, 19.0), (19.0, 19.0)]


def test_lane_and_half_calibrations_agree_on_court_positions():
    job = {
        "jobId": new_job_id(),
        "video": {"fps": 12, "frames": 2, "width": 1280, "height": 720},
        "teamLabels": {str(o): o % 2 for o in range(20)},
    }
    sam = parse_sam_text(fixture_text())
    half = {"source": "coach", "frame": 0, "points": HALF_PX, "landmarks": "half"}
    lane = {"source": "coach", "frame": 0, "points": LANE_PX, "landmarks": "lane"}
    play_half, _, _ = build_outputs(job, sam, None, None, half, no_colors)
    play_lane, _, _ = build_outputs(job, sam, None, None, lane, no_colors)
    for pid, track in play_half["tracks"]["players"].items():
        for a, b in zip(track, play_lane["tracks"]["players"][pid], strict=True):
            assert abs(a[0] - b[0]) < 0.05 and abs(a[1] - b[1]) < 0.05


def test_calibrate_with_lane_corners(ctx):
    client, store, _ = ctx
    jid = finished_job(store)
    r = client.post(f"/film/jobs/{jid}/calibrate", json={"frame": 0, "points": LANE_PX, "landmarks": "lane"})
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["calibration"]["landmarks"] == "lane" and view["calibration"]["source"] == "coach"
    assert store.read_calibration(jid)["landmarks"] == "lane"
    assert client.get(f"/film/jobs/{jid}").json()["calibration"]["landmarks"] == "lane"
    # The same camera marked by its half-court corners gives the same court positions.
    half = client.post(f"/film/jobs/{jid}/calibrate", json={"frame": 0, "points": HALF_PX}).json()
    assert half["calibration"]["landmarks"] == "half"
    for pid, track in half["playDna"]["tracks"]["players"].items():
        for a, b in zip(track, view["playDna"]["tracks"]["players"][pid], strict=True):
            assert abs(a[0] - b[0]) < 0.2 and abs(a[1] - b[1]) < 0.2


def test_older_calibration_records_read_as_half():
    assert jobs.calibration_view({"source": "coach", "frame": 0, "points": COACH})["landmarks"] == "half"
