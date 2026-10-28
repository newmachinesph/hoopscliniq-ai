from app import jobs
from app.playdna import build_overlay
from app.sam import parse_sam_text
from tests.test_sam_parse import fixture_text


def test_overlay_scales_sam_boxes_to_clip_pixels():
    sam = parse_sam_text(fixture_text())  # 2 frames, SAM size 1920x1080
    overlay = build_overlay(
        tracks=sam.tracks(),
        ids={0: "o1", 1: "d1"},
        fps=12,
        frames=3,
        sam_size=(sam.width, sam.height),
        clip_size=(1280, 720),
    )
    assert overlay["fps"] == 12 and overlay["frames"] == 3 and overlay["imageSize"] == [1280, 720]
    assert list(overlay["players"]) == ["o1", "d1"]
    o1 = overlay["players"]["o1"]
    assert o1["team"] == "offense" and overlay["players"]["d1"]["team"] == "defense"
    # SAM frame 0, object 0: (583, 353, 638, 499) at 1920x1080 -> x 2/3 at 1280x720.
    assert o1["boxes"][0] == [389, 235, 425, 333]
    assert o1["boxes"][1] == [389, 235, 428, 333]
    assert o1["boxes"][2] is None  # clip frame SAM did not report
    for player in overlay["players"].values():
        assert len(player["boxes"]) == overlay["frames"]
        for box in player["boxes"]:
            assert box is None or all(isinstance(v, int) for v in box)


def test_overlay_clamps_to_the_frame_and_skips_frames_past_the_clip():
    sam = parse_sam_text(
        "<0f>7<|box;x1=-5;y1=2;x2=30;y2=12;w=20;h=10|>\n<4f>7<|box;x1=1;y1=1;x2=2;y2=2;w=20;h=10|>"
    )
    overlay = build_overlay(
        tracks=sam.tracks(), ids={7: "o1"}, fps=12, frames=2, sam_size=(20, 10), clip_size=(20, 10)
    )
    assert overlay["players"]["o1"]["boxes"] == [[0, 2, 20, 10], None]


def test_build_from_sam_overlay_matches_play_dna(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "ffmpeg_frame_reader", lambda clip, ffmpeg: lambda frames: {})
    pink, navy = (230.0, 90.0, 150.0), (30.0, 40.0, 90.0)
    monkeypatch.setattr(
        jobs, "track_colors", lambda tracks, objs, reader: {o: (pink if o % 2 else navy) for o in objs}
    )
    job = {
        "jobId": "job-abcdefghijkl",
        "calibration": None,
        "video": {"fps": 12, "frames": 2, "width": 1280, "height": 720, "duration": 0.2},
    }
    play, overlay = jobs.build_from_sam(job, fixture_text(), tmp_path / "clip.mp4", "ffmpeg")
    assert set(overlay["players"]) == set(play["players"])
    assert overlay["frames"] == play["source"]["frames"] == 2
    for pid, player in overlay["players"].items():
        assert player["team"] == play["players"][pid]["team"]
        assert len(player["boxes"]) == len(play["tracks"]["players"][pid]) == overlay["frames"]
    # Frame alignment: the SAM object behind each id has its frame-0 box at boxes[0].
    tracks = parse_sam_text(fixture_text()).tracks()
    for pid, obj in play["meta"]["samObjectIds"].items():
        for frame in range(overlay["frames"]):
            b = tracks[obj].get(frame)
            expected = None if b is None else [round(v * 2 / 3) for v in (b.x1, b.y1, b.x2, b.y2)]
            assert overlay["players"][pid]["boxes"][frame] == expected
