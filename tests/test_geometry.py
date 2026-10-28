import pytest

from app.geometry import (
    COURT_CORNERS,
    Calibration,
    apply_h,
    clamp,
    court_tracks,
    fill_gaps,
    foot_point,
    hold_nearest,
    homography,
    load_default_calibration,
    parse_calibration,
    smooth,
    top_tracks,
)
from app.sam import Box

IMAGE = [(850.0, 285.0), (1422.0, 505.0), (265.0, 641.0), (272.0, 302.0)]


def test_homography_maps_the_four_corners_exactly():
    h = homography(IMAGE, COURT_CORNERS)
    for src, dst in zip(IMAGE, COURT_CORNERS, strict=True):
        u, v = apply_h(h, *src)
        assert u == pytest.approx(dst[0], abs=1e-6)
        assert v == pytest.approx(dst[1], abs=1e-6)


def test_homography_identity_like_scale():
    h = homography([(0, 0), (100, 0), (100, 94), (0, 94)], COURT_CORNERS)
    assert apply_h(h, 50, 47) == pytest.approx((25.0, 23.5))


def test_foot_point_is_bottom_center():
    assert foot_point(Box(1, 10, 20, 30, 80)) == (20.0, 80)


def test_clamp_to_court():
    assert clamp((-3, 90)) == (0.0, 84.0)
    assert clamp((60, -1)) == (50.0, 0.0)
    assert clamp((10, 20)) == (10, 20)


def test_fill_gaps_interpolates_short_gaps_only():
    series = [(0.0, 0.0), None, None, (3.0, 6.0)] + [None] * 7 + [(10.0, 10.0)]
    out = fill_gaps(series, max_gap=6)
    assert out[1] == pytest.approx((1.0, 2.0))
    assert out[2] == pytest.approx((2.0, 4.0))
    assert out[4:11] == [None] * 7  # gap of 7 > 6 stays open


def test_hold_nearest_fills_edges_and_long_gaps():
    out = hold_nearest([None, (1.0, 1.0), None, (2.0, 2.0), None])
    assert out == [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 2.0)]


def test_smooth_is_a_centered_moving_average():
    series = [(float(i), 0.0) for i in range(7)]
    out = smooth(series, window=5)
    assert out[3] == pytest.approx((3.0, 0.0))
    assert out[0] == pytest.approx((1.0, 0.0))  # mean of 0,1,2
    spike = smooth([(0.0, 0.0)] * 2 + [(10.0, 0.0)] + [(0.0, 0.0)] * 2, window=5)
    assert spike[2] == pytest.approx((2.0, 0.0))


def test_top_tracks_keeps_the_longest():
    tracks = {i: {f: Box(i, 0, 0, 1, 1) for f in range(i + 1)} for i in range(12)}
    assert top_tracks(tracks, limit=10) == list(range(11, 1, -1))


def test_court_tracks_cover_every_frame_and_count_observed():
    h = homography([(0, 0), (100, 0), (100, 94), (0, 94)], COURT_CORNERS)
    tracks = {7: {0: Box(7, 40, 0, 60, 47), 4: Box(7, 40, 0, 60, 47)}}
    out = court_tracks(tracks, [h] * 6, frames=6)
    assert len(out) == 1
    t = out[0]
    assert t.obj == 7 and t.observed == 2
    assert len(t.positions) == 6
    assert t.positions[0] == pytest.approx((25.0, 23.5))


def test_calibration_scales_to_box_frame_size():
    cal = Calibration([(1920, 1080)] * 4, (1920, 1080), "estimated")
    assert cal.scaled_to(1280, 720)[0] == pytest.approx((1280, 720))


def test_parse_calibration_shapes():
    assert parse_calibration([[0, 0], [1, 0], [1, 1], [0, 1]]).source == "given"
    cal = parse_calibration({"points": [[0, 0], [1, 0], [1, 1], [0, 1]], "imageSize": [1280, 720]})
    assert cal.image_size == (1280, 720)
    with pytest.raises(ValueError):
        parse_calibration([[0, 0], [1, 0]])
    with pytest.raises(ValueError):
        parse_calibration([[0, 0], [1, 0], [1, "x"], [0, 1]])


def test_default_calibration_is_estimated():
    cal = load_default_calibration()
    assert cal.source == "estimated"
    assert cal.image_size == (1920, 1080)
    assert len(cal.points) == 4
