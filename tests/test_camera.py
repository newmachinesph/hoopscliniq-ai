import cv2
import numpy as np
import pytest

from app.camera import CameraMotion, detect_features, drop_masked, estimate_motion
from app.geometry import COURT_CORNERS, apply_h, homography, image_to_court

W, H = 320, 200
STEP = 4  # the camera pans right by 4 px per frame


def scene(width: int = 900, height: int = H, seed: int = 7) -> np.ndarray:
    """A textured 'gym wall and floor': smoothed noise plus random rectangles, rich in corners."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (height, width), dtype=np.uint8), (0, 0), 1.5)
    for _ in range(150):
        x, y = int(rng.integers(0, width - 20)), int(rng.integers(0, height - 20))
        cv2.rectangle(
            img,
            (x, y),
            (x + int(rng.integers(5, 20)), y + int(rng.integers(5, 20))),
            int(rng.integers(0, 255)),
            -1,
        )
    return img


def pan(n: int, world: np.ndarray) -> list[np.ndarray]:
    return [world[:, i * STEP : i * STEP + W].copy() for i in range(n)]


def motion(frames: list[np.ndarray], boxes=lambda i: []) -> CameraMotion:
    return estimate_motion(detect_features(frames, scale=1.0), boxes, (W, H), len(frames), scale=1.0)


def test_pan_is_followed_on_every_frame():
    frames = pan(30, scene())
    cm = motion(frames)
    assert cm.summary() == {"framesTracked": 30, "framesFallback": 0}
    assert np.allclose(cm.to_frame0[0], np.eye(3))
    for i in (5, 17, 29):
        # A pixel of frame i shows the scene point that sat STEP * i px further right in frame 0.
        assert apply_h(cm.to_frame0[i], 100, 80) == pytest.approx((100 + STEP * i, 80), abs=1.0)


def test_moving_player_inside_its_box_does_not_drag_the_camera():
    world = scene()
    frames = pan(20, world)
    patch = scene(60, 80, seed=99)
    boxes = {}
    for i, f in enumerate(frames):
        x = 20 + 9 * i  # the 'player' runs right much faster than the pan
        f[100:180, x : x + 60] = patch
        boxes[i] = [(x, 100, x + 60, 180)]
    cm = motion(frames, lambda i: boxes[i])
    assert cm.frames_fallback == 0
    assert apply_h(cm.to_frame0[19], 160, 50) == pytest.approx((160 + STEP * 19, 50), abs=1.0)


def test_a_frame_with_nothing_in_common_falls_back():
    frames = pan(12, scene())
    frames[6] = scene(W, H, seed=1234)  # a cut to an unrelated picture
    cm = motion(frames)
    assert cm.to_frame0[6] is None
    assert cm.summary() == {"framesTracked": 11, "framesFallback": 1}
    assert apply_h(cm.to_frame0[11], 100, 80) == pytest.approx((100 + STEP * 11, 80), abs=1.0)


def test_missing_decoded_frames_count_as_fallback():
    frames = pan(5, scene())
    cm = estimate_motion(detect_features(frames, scale=1.0), lambda i: [], (W, H), total=8, scale=1.0)
    assert len(cm.to_frame0) == 8 and cm.frames_fallback == 3


def test_drop_masked_keeps_only_background_points():
    pts = np.float32([[10, 10], [50, 50], [90, 90]])
    des = np.arange(3 * 32, dtype=np.uint8).reshape(3, 32)
    kept, kept_des = drop_masked(
        (pts, des), [(40, 40, 60, 60)], scale=0.5
    )  # box covers (20..30) at half size
    assert kept.tolist() == [[10, 10], [50, 50], [90, 90]]
    kept, kept_des = drop_masked((pts, des), [(80, 80, 120, 120)], scale=0.5)
    assert kept.tolist() == [[10, 10], [90, 90]] and len(kept_des) == 2


def test_camera_json_round_trip():
    cm = CameraMotion([np.eye(3), None, np.diag([1.0, 1.0, 1.0]) + 0.5], (1280, 720))
    back = CameraMotion.from_json(cm.to_json())
    assert back.image_size == (1280, 720) and back.to_frame0[1] is None
    assert np.allclose(back.to_frame0[2], cm.to_frame0[2])


def test_homography_chain_keeps_a_court_point_fixed_through_a_synthetic_pan():
    """Court position = H_cal . H_(i->calFrame) . p: a spot painted on the floor stays put while the camera
    pans, whichever frame the coach marked the corners on."""
    world = scene()
    frames = pan(25, world)
    cm = motion(frames)
    corners_world = [(300.0, 40.0), (600.0, 40.0), (620.0, 190.0), (280.0, 190.0)]  # half court, scene px
    spot_world = (450.0, 120.0)
    expected = apply_h(homography(corners_world, COURT_CORNERS), *spot_world)
    for cal_frame in (0, 12, 24):
        shift = STEP * cal_frame
        corners = [(x - shift, y) for x, y in corners_world]  # where the corners appear in the marked frame
        mats = image_to_court(homography(corners, COURT_CORNERS), cm.to_frame0, cal_frame, len(frames))
        for i in (0, 9, 24):
            seen_at = (spot_world[0] - STEP * i, spot_world[1])  # the spot in frame i's pixels
            assert apply_h(mats[i], *seen_at) == pytest.approx(expected, abs=0.3)


def test_image_to_court_without_motion_is_the_static_homography():
    h = homography([(0, 0), (100, 0), (100, 94), (0, 94)], COURT_CORNERS)
    mats = image_to_court(h, None, 0, 3)
    assert len(mats) == 3 and all(m is h for m in mats)
    shift = np.array([[1.0, 0, 10], [0, 1, 0], [0, 0, 1]])
    mats = image_to_court(h, [np.eye(3), None, shift], 0, 3)
    assert mats[1] is h  # frame without a match falls back to the static calibration
    assert apply_h(mats[2], 40, 47) == pytest.approx(apply_h(h, 50, 47))
