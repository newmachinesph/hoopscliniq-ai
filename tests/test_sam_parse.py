from pathlib import Path

from app.sam import Box, deltas_from_sse, parse_sam_text

FIXTURE = Path(__file__).parent / "fixtures" / "sam-boys-2frames.txt"


def fixture_text() -> str:
    with FIXTURE.open(encoding="utf-8") as f:
        return deltas_from_sse(f)


def test_sse_deltas_concatenate_two_frames():
    text = fixture_text()
    assert text.startswith("<0f>")
    assert "<1f>" in text
    assert "<2f>" not in text


def test_parse_two_frames_from_real_stream():
    sam = parse_sam_text(fixture_text())
    assert sam.frame_count == 2
    assert (sam.width, sam.height) == (1920, 1080)
    assert len(sam.frames[0]) == 8
    assert len(sam.frames[1]) == 8
    assert sam.frames[0][0] == Box(0, 583, 353, 638, 499)
    assert sam.frames[0][1] == Box(1, 1670, 427, 1791, 671)
    assert sam.frames[1][0] == Box(0, 583, 352, 642, 499)


def test_object_ids_are_stable_tracks():
    tracks = parse_sam_text(fixture_text()).tracks()
    assert set(tracks[0]) == {0, 1}
    assert tracks[0][1].x2 == 642


def test_masks_are_ignored_and_empty_frames_kept():
    text = (
        "<0f>3<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=8|><|mask;x=0;y=0;data=1,2,~ab|>,5<|box;x1=5;y1=6;x2=7;y2=8;w=10;h=8|>"
        "<|mask;x=0;y=0;data=9,9,~xy|>\n<1f>\n<2f>3<|box;x1=2;y1=2;x2=4;y2=5;w=10;h=8|><|mask;data=1|>\n"
    )
    sam = parse_sam_text(text)
    assert sam.frame_count == 3
    assert [b.obj for b in sam.frames[0]] == [3, 5]
    assert sam.frames[1] == []
    assert sam.tracks()[3][2] == Box(3, 2, 2, 4, 5)


def test_parse_empty_text():
    sam = parse_sam_text("")
    assert sam.frame_count == 0
    assert sam.tracks() == {}
