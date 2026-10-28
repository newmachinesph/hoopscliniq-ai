import json

from PIL import Image

from app.geometry import CourtTrack
from app.naming import naming_messages, naming_schema, to_suggested_name, track_summary
from app.playdna import UNNAMED, build_play_dna
from app.sam import Box
from app.teams import assign_ids, jersey_box, mean_rgb, pick_offense, track_colors, two_means

PLAYBOOK = [
    {"id": "horns-twist", "title": "Horns Twist", "description": "Horns ball screen, then a re-screen."},
    {"id": "zone-gut", "title": "Zone Gut", "description": "Gap the top of the zone."},
]


def make_tracks(n_frames: int = 24) -> list[CourtTrack]:
    return [
        CourtTrack(obj=4, observed=24, positions=[(25.0, 30.0)] * n_frames),
        CourtTrack(obj=1, observed=20, positions=[(25.0, 10.0)] * n_frames),
        CourtTrack(obj=9, observed=12, positions=[(10.0, 20.0)] * n_frames),
    ]


def build(name: str = UNNAMED) -> dict:
    tracks = make_tracks()
    return build_play_dna(
        job_id="job-abcdefghijkl",
        fps=12,
        frames=24,
        court_tracks=tracks,
        ids={4: "o1", 1: "d1", 9: "o2"},
        calibration_source="estimated",
        sam_frames=24,
        image_size=(1280, 720),
        name=name,
    )


def test_play_dna_v2_shape():
    play = build()
    assert play["version"] == 2 and play["dictionary"] == "1.5"
    assert play["name"] == UNNAMED
    assert play["source"] == {"clip": "job-abcdefghijkl", "authored": False, "fps": 12, "frames": 24}
    assert play["court"]["rim"] == [25, 5.25]
    assert list(play["players"]) == ["o1", "o2", "d1"]
    assert play["players"]["d1"] == {"team": "defense"}
    assert len(play["tracks"]["players"]["o1"]) == 24
    assert play["tracks"]["players"]["o1"][0] == [25.0, 30.0]
    assert play["tracks"]["ball"] == [] and play["tracks"]["holder"] == [] and play["tracks"]["areas"] == []
    assert play["moments"] == [{"frame": 0, "label": "Start"}]
    assert play["events"] == [] and play["actions"] == [] and play["fixes"] == []
    meta = play["meta"]
    assert meta["playersFound"] == 3
    assert meta["calibration"] == "estimated"
    assert meta["samFrames"] == 24
    assert meta["imageSize"] == [1280, 720]
    assert meta["coverage"]["o2"] == 0.5
    json.dumps(play)  # serialisable


def test_two_means_splits_two_jersey_colors():
    pink = [(230.0, 90.0, 150.0), (225.0, 100.0, 160.0), (235.0, 95.0, 140.0)]
    navy = [(30.0, 40.0, 90.0), (20.0, 35.0, 100.0)]
    labels = two_means(pink + navy)
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert two_means([(1.0, 2.0, 3.0)]) == [0]


def test_jersey_crop_is_top_forty_percent_and_mean_rgb():
    assert jersey_box(Box(0, 10, 100, 30, 200)) == (10, 100, 30, 140)
    im = Image.new("RGB", (40, 40), (200, 10, 10))
    assert mean_rgb(im, (0, 0, 10, 10)) == (200.0, 10.0, 10.0)
    assert mean_rgb(im, (50, 50, 60, 60)) is None


def test_track_colors_reads_first_frames():
    tracks = {1: {f: Box(1, 0, 0, 10, 10) for f in range(15)}}
    seen = []

    def reader(frames):
        seen.extend(frames)
        return {f: Image.new("RGB", (20, 20), (0, 0, 255)) for f in frames}

    colors = track_colors(tracks, [1], reader, sample=10)
    assert seen == list(range(10))
    assert colors[1] == (0.0, 0.0, 255.0)


def test_offense_is_the_team_with_more_players_in_the_attacked_half():
    tracks = {t.obj: t for t in make_tracks()}
    tracks[1] = CourtTrack(1, 20, [(25.0, 60.0)] * 24)  # defender in the backcourt
    labels = {4: 0, 9: 0, 1: 1}
    assert pick_offense(labels, tracks, 12) == 0


def test_offense_tie_goes_to_the_team_farther_from_the_rim():
    tracks = {
        1: CourtTrack(1, 1, [(25.0, 30.0)]),
        2: CourtTrack(2, 1, [(25.0, 25.0)]),
    }
    assert pick_offense({1: 0, 2: 1}, tracks, 0) == 0


def test_ids_follow_track_length_within_each_team():
    ids = assign_ids(make_tracks(), {4: 1, 1: 0, 9: 1}, offense=1)
    assert ids == {4: "o1", 1: "d1", 9: "o2"}


def test_track_summary_is_one_sample_per_second():
    summary = track_summary(build())
    line = next(x for x in summary.splitlines() if x.startswith("o1 "))
    assert "start (25,30) end (25,30)" in line
    assert "0s(25,30)" in line and "1s(25,30)" in line and "2s" not in line


def test_naming_prompt_and_schema():
    msgs = naming_messages(PLAYBOOK, build())
    assert msgs[0]["role"] == "system"
    assert "horns-twist: Horns Twist." in msgs[1]["content"]
    assert "calibration estimated" in msgs[1]["content"]
    schema = naming_schema(["horns-twist", "zone-gut"])
    assert schema["properties"]["playId"]["enum"] == ["horns-twist", "zone-gut", "unknown"]
    assert schema["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


def test_suggested_name_rules():
    s = to_suggested_name({"playId": "zone-gut", "title": "x", "confidence": "medium", "why": "w"}, PLAYBOOK)
    assert s == {
        "playId": "zone-gut",
        "title": "Zone Gut",
        "confidence": "medium",
        "why": "w",
        "needsReview": True,
    }
    u = to_suggested_name(
        {"playId": "unknown", "title": "Weave", "confidence": "low", "why": "y" * 300}, PLAYBOOK
    )
    assert u["title"] == UNNAMED and u["playId"] == "unknown" and len(u["why"]) == 160
    bad = to_suggested_name({"playId": "made-up", "confidence": "sure"}, PLAYBOOK)
    assert bad["playId"] == "unknown" and bad["confidence"] == "low"


def test_naming_prompt_lists_ball_events():
    from app.naming import events_summary

    play = build()
    assert events_summary(play) == "Ball not tracked in this clip."
    play["tracks"]["ball"] = [[25.0, 20.0, 0]] * 24
    assert events_summary(play) == "No passes or shots detected."
    play["events"] = [
        {"type": "pass", "from": "o1", "to": "o2", "frame": 12, "end": 15, "confidence": "low"},
        {"type": "shot", "from": "o2", "frame": 18, "end": 22, "made": None, "confidence": "medium"},
    ]
    assert events_summary(play).splitlines() == [
        "1.0s pass o1 -> o2 (low confidence)",
        "1.5s shot by o2 (medium confidence)",
    ]
    assert "1.0s pass o1 -> o2" in naming_messages(PLAYBOOK, play)[1]["content"]
