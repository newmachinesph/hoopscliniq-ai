from app.ballpos import RIM, ball_court_track

FPS = 12


def players(n: int):
    return {"o1": [(10.0 + i, 20.0) for i in range(n)], "o2": [(30.0, 20.0 + i) for i in range(n)]}


def test_held_frame_uses_the_holder_position_even_if_projection_is_elsewhere():
    out, counts = ball_court_track(["o1", None], players(2), [(99, 99), None], [], FPS)
    assert out[0] == [10.0, 20.0, 0]
    assert out[1] is None
    assert counts == {"held": 1, "interpolated": 0, "projected": 0, "missing": 1}


def test_gap_between_held_frames_interpolates_linearly():
    holder = ["o1", None, None, "o2"]
    out, counts = ball_court_track(holder, players(4), [None] * 4, [], FPS)
    assert out[1] == [round(10 + (30 - 10) / 3, 2), round(20 + (23 - 20) / 3, 2), 0]
    assert out[2] == [round(10 + 2 * (30 - 10) / 3, 2), round(20 + 2 * (23 - 20) / 3, 2), 0]
    assert counts["interpolated"] == 2


def test_gap_longer_than_two_seconds_is_not_interpolated():
    n = 2 * FPS + 2
    holder = ["o1"] + [None] * (n - 2) + ["o2"]
    out, counts = ball_court_track(holder, players(n), [None] * n, [], FPS)
    assert counts["interpolated"] == 0
    assert out[5] is None


def test_projection_used_inside_court_with_margin_else_null_never_clamped():
    proj = [(25.0, 20.0), (-2.5, 49.0), (60.0, 10.0), (25.0, -5.0)]
    out, counts = ball_court_track([None] * 4, players(4), proj, [], FPS)
    assert out[0] == [25.0, 20.0, 0]
    assert out[1] == [-2.5, 49.0, 0]
    assert out[2] is None
    assert out[3] is None
    assert counts == {"held": 0, "interpolated": 0, "projected": 2, "missing": 2}


def test_interpolation_beats_projection():
    holder = ["o1", None, "o1"]
    out, counts = ball_court_track(holder, players(3), [None, (40.0, 40.0), None], [], FPS)
    assert out[1] == [11.0, 20.0, 0]
    assert counts["interpolated"] == 1 and counts["projected"] == 0


def test_shot_without_later_holder_travels_to_the_rim():
    holder = ["o1", None, None, None, None]
    shot = {"type": "shot", "from": "o1", "frame": 0, "end": 4}
    out, counts = ball_court_track(holder, players(5), [(40.0, 40.0)] * 5, [shot], FPS)
    assert out[0] == [10.0, 20.0, 0]
    assert out[4] == [RIM[0], RIM[1], 0]
    assert out[2] == [round((10 + 25) / 2, 2), round((20 + 5.25) / 2, 2), 0]
    assert counts == {"held": 1, "interpolated": 4, "projected": 0, "missing": 0}


def test_shot_with_next_holder_keeps_pass_interpolation():
    holder = ["o1", None, None, "o2"]
    shot = {"type": "shot", "from": "o1", "frame": 0, "end": 3}
    out, _ = ball_court_track(holder, players(4), [None] * 4, [shot], FPS)
    assert out[3] == [30.0, 23.0, 0]
    assert out[1] == [round(10 + (30 - 10) / 3, 2), round(20 + (23 - 20) / 3, 2), 0]
