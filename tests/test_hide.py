import shutil
import subprocess

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import consent
from app.api import create_app
from app.jobs import JobStore, new_job_id

W, H, FRAMES = 64, 48, 24
BOX = [16, 12, 32, 28]  # padded by 15 %: x 14..34, y 10..30
INSIDE = (slice(14, 34), slice(10, 30))
OUTSIDE = (slice(44, 64), slice(0, 48))


def find_ffmpeg() -> str | None:
    if found := shutil.which("ffmpeg"):
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = find_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")


def gradient_frame() -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    return np.stack([xs * 2 + ys, xs * 2 + ys + 15, xs * 2 + ys + 30], axis=-1).astype(np.uint8)


def make_clip(path) -> None:
    """24 frames of a smooth gradient (stays close to itself through H.264)."""
    frame = gradient_frame()
    raw = ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "12", "-i", "-"]
    out = ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen([FFMPEG, "-loglevel", "error", "-y", *raw, *out], stdin=subprocess.PIPE)
    for _ in range(FRAMES):
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


def read_frames(path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            return out
        out.append(frame.astype(int))


def region_diff(a: np.ndarray, b: np.ndarray, region) -> float:
    xs, ys = region
    return float(np.abs(a[ys, xs] - b[ys, xs]).mean())


def overlay(boxes) -> dict:
    return {
        "fps": 12,
        "frames": FRAMES,
        "imageSize": [W, H],
        "players": {"o1": {"team": "offense", "boxes": boxes}, "d1": {"team": "defense", "boxes": boxes}},
    }


def test_box_for_frame_holds_the_nearest_box():
    boxes = [[1, 1, 2, 2], None, None, None, None, None, None, None, [5, 5, 6, 6]]
    assert consent.box_for_frame(boxes, 0, 6) == [1, 1, 2, 2]
    assert consent.box_for_frame(boxes, 3, 6) == [1, 1, 2, 2]
    assert consent.box_for_frame(boxes, 5, 6) == [5, 5, 6, 6]  # nearer to the later box
    assert consent.box_for_frame(boxes, 4, 3) is None
    assert consent.box_for_frame(boxes, 20, 6) is None


def test_padded_grows_and_clamps():
    assert consent.padded([20, 20, 40, 40], 100, 100, 0.15) == (17, 17, 43, 43)
    assert consent.padded([0, 0, 40, 40], 42, 100, 0.15) == (0, 0, 42, 46)


@needs_ffmpeg
def test_render_hides_the_box_region_only(tmp_path):
    src, dst = tmp_path / "clip.mp4", tmp_path / "clip-hidden.mp4"
    make_clip(src)
    boxes = [BOX] * 12 + [None] * 12
    result = consent.render_hidden(src, dst, overlay(boxes), ["o1"], FFMPEG)
    assert result == {"frames": FRAMES, "framesMissed": {"o1": 6}}
    before, after = read_frames(src), read_frames(dst)
    assert len(after) == FRAMES and after[0].shape == before[0].shape
    for f in (0, 11, 17):  # boxed and held frames
        assert region_diff(before[f], after[f], INSIDE) > 20
        assert region_diff(before[f], after[f], OUTSIDE) < 4
    for f in (18, 23):  # missed frames are not touched
        assert region_diff(before[f], after[f], INSIDE) < 4
    assert not list(tmp_path.glob("*.tmp.*"))


@needs_ffmpeg
def test_render_fails_cleanly_when_ffmpeg_fails(tmp_path):
    src = tmp_path / "clip.mp4"
    make_clip(src)
    with pytest.raises(RuntimeError):
        consent.render_hidden(src, tmp_path / "x.bad", overlay([BOX] * FRAMES), ["o1"], FFMPEG)
    assert list(tmp_path.glob("*.tmp.*")) == []


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    # The route calls set_hidden with the default "ffmpeg"; point it at the ffmpeg this machine has.
    real = consent.set_hidden
    monkeypatch.setattr(
        "app.api.set_hidden", lambda s, j, o, ids: real(s, j, o, ids, ffmpeg=FFMPEG or "ffmpeg")
    )
    app = create_app(
        store=store,
        start_job=lambda j: None,
        ask=lambda q: {},
        models_reachable=lambda: True,
        probe_duration=lambda p: 1.0,
    )
    return TestClient(app), store


def done_job(store, status="done", with_overlay=True, with_clip=False) -> str:
    jid = new_job_id()
    store.write(
        {
            "jobId": jid,
            "status": status,
            "finishedAt": "2026-10-28T10:00:00Z",
            "video": {"fps": 12, "frames": FRAMES, "width": W, "height": H, "duration": 2.0},
        }
    )
    if with_overlay:
        store.write_overlay(jid, overlay([BOX] * FRAMES))
    if with_clip:
        make_clip(store.dir(jid) / "clip.mp4")
    return jid


def test_hide_400_404_409(ctx):
    client, store = ctx
    jid = done_job(store, with_clip=False)
    url = f"/film/jobs/{jid}/hide"
    assert client.post("/film/jobs/nope/hide", json={"playerIds": []}).status_code == 400
    assert client.post(f"/film/jobs/{new_job_id()}/hide", json={"playerIds": []}).status_code == 404
    for body in ({}, {"playerIds": "o1"}, {"playerIds": [1]}, {"playerIds": ["o1"] * 5}):
        r = client.post(url, json=body)
        assert r.status_code == 400 and "error" in r.json()
    assert client.post(url, content=b"not json").status_code == 400
    r = client.post(url, json={"playerIds": ["zz"]})
    assert r.status_code == 400 and "zz" in r.json()["error"]
    assert client.post(url, json={"playerIds": ["o1"]}).status_code == 409  # no clip.mp4
    running = done_job(store, status="track")
    assert client.post(f"/film/jobs/{running}/hide", json={"playerIds": []}).status_code == 409
    no_overlay = done_job(store, with_overlay=False)
    assert client.post(f"/film/jobs/{no_overlay}/hide", json={"playerIds": ["o1"]}).status_code == 409


@needs_ffmpeg
def test_hide_switches_the_clip_route_and_back(ctx):
    client, store = ctx
    jid = done_job(store, with_clip=True)
    original = (store.dir(jid) / "clip.mp4").read_bytes()
    assert client.get(f"/film/jobs/{jid}/clip").content == original
    assert client.get(f"/film/jobs/{jid}").json()["hidden"] is None

    r = client.post(f"/film/jobs/{jid}/hide", json={"playerIds": ["o1", "o1", "d1"]})
    assert r.status_code == 200
    view = r.json()
    hidden = view["hidden"]
    assert hidden["playerIds"] == ["o1", "d1"] and hidden["frames"] == FRAMES
    assert hidden["framesMissed"] == {"o1": 0, "d1": 0}
    assert view["clipUrl"] == f"http://testserver/film/jobs/{jid}/clip?v={hidden['at']}"
    assert view["overlay"]["players"]["o1"]["boxes"][0] == BOX  # the court view keeps the boxes
    assert (store.dir(jid) / "hidden.json").is_file()
    served = client.get(f"/film/jobs/{jid}/clip")
    assert served.status_code == 200
    assert served.content == (store.dir(jid) / "clip-hidden.mp4").read_bytes()
    assert served.content != original  # the original clip is served by no route now

    r = client.post(f"/film/jobs/{jid}/hide", json={"playerIds": []})
    assert r.status_code == 200 and r.json()["hidden"] is None
    assert r.json()["clipUrl"] == f"http://testserver/film/jobs/{jid}/clip"
    assert not (store.dir(jid) / "hidden.json").exists()
    assert not (store.dir(jid) / "clip-hidden.mp4").exists()
    assert client.get(f"/film/jobs/{jid}/clip").content == original


def test_clip_is_never_the_original_when_the_hidden_file_is_missing(ctx):
    client, store = ctx
    jid = done_job(store)
    (store.dir(jid) / "clip.mp4").write_bytes(b"original")
    store.write_hidden(jid, {"playerIds": ["o1"], "framesMissed": {"o1": 0}, "frames": 1, "at": "t"})
    assert client.get(f"/film/jobs/{jid}/clip").status_code == 404


def test_empty_hidden_list_counts_as_not_hidden(ctx):
    client, store = ctx
    jid = done_job(store)
    (store.dir(jid) / "clip.mp4").write_bytes(b"original")
    store.write_hidden(jid, {"playerIds": [], "framesMissed": {}, "frames": 1, "at": "t"})
    assert client.get(f"/film/jobs/{jid}/clip").content == b"original"
    assert client.get(f"/film/jobs/{jid}").json()["hidden"] is None
