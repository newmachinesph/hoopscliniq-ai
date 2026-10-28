from app.events import detect_events, find_holders, hold_segments, near_player, pick_ball
from app.sam import Box

FPS = 12
# Players stand still; boxes in clip pixels.
O1 = [100, 200, 160, 360]
O2 = [400, 200, 460, 360]
D1 = [250, 200, 310, 360]
PLAYERS = {"o1": [O1] * 40, "o2": [O2] * 40, "d1": [D1] * 40}


def ball_at(x: float, y: float, r: float = 8) -> list[float]:
    return [x - r, y - r, x + r, y + r]


def flat_court(b: list[float] | None) -> tuple[float, float] | None:
    """A toy projection: 10 px = 1 ft, far from the rim unless the test says otherwise."""
    if b is None:
        return None
    return ((b[0] + b[2]) / 20, 30 + (b[1] + b[3]) / 200)


def test_pick_ball_prefers_continuity_then_the_longest_object():
    frames = {
        0: [Box(1, 0, 0, 10, 10), Box(2, 100, 100, 110, 110)],
        1: [Box(2, 100, 100, 110, 110), Box(1, 2, 0, 12, 10)],
        2: [Box(3, 104, 100, 114, 110), Box(4, 300, 300, 310, 310)],
        3: [Box(2, 101, 100, 111, 110)],
    }
    picked = pick_ball(frames, 5)
    # Object 2 is seen longest (3 frames), so it starts the track; frames 2 and 3 take the nearest candidate.
    assert [b.obj if b else None for b in picked] == [2, 2, 3, 2, None]


def test_holder_is_the_player_whose_box_holds_the_ball():
    ball = [ball_at(130, 300), ball_at(165, 300), ball_at(205, 300), None]  # reach = 0.25 x 160 px
    assert find_holders(ball, PLAYERS) == ["o1", "o1", None, None]


def test_pass_between_teammates():
    flight = [ball_at(130 + 30 * k, 100) for k in range(1, 10)]  # in the air, over d1's head
    ball = [ball_at(130, 300)] * 10 + flight + [ball_at(430, 300)] * 10
    holder = find_holders(ball, PLAYERS)
    events = detect_events(holder, ball, [flat_court(b) for b in ball], PLAYERS, FPS)
    assert holder[0] == "o1" and holder[-1] == "o2"
    passes = [e for e in events if e["type"] == "pass"]
    assert len(passes) == 1 and len(events) == 1
    p = passes[0]
    assert p["from"] == "o1" and p["to"] == "o2"
    assert p["frame"] < p["end"] and p["detect"] == "heuristic" and p["confidence"] in ("low", "medium")


def test_ball_to_an_opponent_is_not_a_pass():
    holder = ["o1"] * 8 + [None] * 3 + ["d1"] * 8
    ball = [ball_at(130, 300)] * 19
    assert detect_events(holder, ball, [flat_court(b) for b in ball], PLAYERS, FPS) == []


def test_pass_needs_the_catch_within_one_and_a_half_seconds():
    holder = ["o1"] * 8 + [None] * 19 + ["o2"] * 8
    ball = [ball_at(130, 300)] * len(holder)
    assert detect_events(holder, ball, [flat_court(b) for b in ball], PLAYERS, FPS) == []


def test_shot_toward_the_rim():
    held = [ball_at(130, 300)] * 8
    flight = [ball_at(140 + 10 * k, 200 - 15 * k) for k in range(1, 7)]  # rises over o1's head
    ball = held + flight + [None] * 6
    holder = find_holders(ball, PLAYERS)
    rim_side = [(10.0, 30.0)] * 8 + [(10.0 + 2 * k, 30.0 - 4 * k) for k in range(1, 7)] + [None] * 6
    events = detect_events(holder, ball, rim_side, PLAYERS, FPS)
    assert len(events) == 1
    shot = events[0]
    assert shot["type"] == "shot" and shot["from"] == "o1" and shot["made"] is None
    assert shot["frame"] == max(i for i, h in enumerate(holder) if h == "o1")  # release = last held frame
    assert shot["end"] > shot["frame"]
    assert shot["detect"] == "heuristic" and shot["confidence"] == "medium"  # it got within 4 ft of the rim


def test_shot_that_only_rises_and_heads_to_the_rim_is_low_confidence():
    ball = [ball_at(130, 300)] * 8 + [ball_at(140, 150), ball_at(150, 120)] + [None] * 4
    holder = find_holders(ball, PLAYERS)
    court = [(10.0, 40.0)] * 8 + [(14.0, 34.0), (16.0, 30.0)] + [None] * 4  # 6.7 ft closer, never within 4
    events = detect_events(holder, ball, court, PLAYERS, FPS)
    assert [(e["type"], e["confidence"]) for e in events] == [("shot", "low")]


def test_no_event_when_the_ball_is_unseen():
    ball = [None] * 30
    holder = find_holders(ball, PLAYERS)
    assert holder == [None] * 30
    assert detect_events(holder, ball, [None] * 30, PLAYERS, FPS) == []


def test_blips_are_ignored_and_the_same_holder_is_joined():
    holder = ["o1"] * 5 + ["o2"] + ["o1"] * 5
    assert [(s.pid, s.start, s.end) for s in hold_segments(holder)] == [("o1", 0, 10)]


def test_a_new_ball_track_starts_on_a_ball_near_a_player():
    frames = {f: [Box(1, 600, 600, 620, 620), Box(2, 125, 290, 135, 300)] for f in range(3)}
    frames[0].append(Box(3, 900, 50, 910, 60))
    for f in range(3, 6):
        frames[f] = [Box(1, 600, 600, 620, 620)]
    # Object 1 (a ball on the scorer's table) is seen longest, but object 2 is in o1's hands.
    picked = pick_ball(frames, 6, lambda f, b: near_player(PLAYERS, f, (b.x1, b.y1, b.x2, b.y2)))
    assert [b.obj for b in picked] == [2, 2, 2, 1, 1, 1]


def test_hand_off_without_a_ball_in_flight_is_low_confidence():
    holder = ["o1"] * 8 + ["o2"] * 8
    ball = [ball_at(130, 300)] * 8 + [ball_at(430, 300)] * 8
    events = detect_events(holder, ball, [flat_court(b) for b in ball], PLAYERS, FPS)
    assert [(e["type"], e["confidence"]) for e in events] == [("pass", "low")]
