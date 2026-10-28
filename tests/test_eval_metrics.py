import json

import pytest

from eval import jobcache
from eval import run as ev

A = (0.0, 0.0, 10.0, 10.0)


def test_iou_identical_disjoint_and_half_overlap():
    assert ev.iou(A, A) == 1.0
    assert ev.iou(A, (20.0, 20.0, 30.0, 30.0)) == 0.0
    # Shifted by half the width: overlap 50, union 150.
    assert ev.iou(A, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1 / 3)
    assert ev.iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)) == 0.0  # degenerate boxes do not divide by zero


def test_match_is_one_to_one_and_needs_iou_of_at_least_half():
    # Two predictions on the same player: only one can match it.
    assert ev.match_frame([A, (0.0, 0.0, 10.0, 9.0)], [A]) == 1
    # IoU exactly 0.5 matches; just under does not.
    assert ev.match_frame([(0.0, 0.0, 10.0, 5.0)], [A]) == 1
    assert ev.match_frame([(0.0, 0.0, 10.0, 4.9)], [A]) == 0
    assert ev.match_frame([], [A]) == 0 and ev.match_frame([A], []) == 0


def test_greedy_matching_takes_the_best_pairs_first():
    t1, t2 = (0.0, 0.0, 10.0, 10.0), (8.0, 0.0, 18.0, 10.0)
    p1, p2 = (8.0, 0.0, 18.0, 10.0), (1.0, 0.0, 11.0, 10.0)
    assert ev.match_frame([p1, p2], [t1, t2]) == 2


def test_score_frames_counts_precision_recall_and_players_per_frame():
    pred = {0: [A, (50.0, 50.0, 60.0, 60.0)], 12: [A], 99: [A]}  # frame 99 is not labelled: ignored
    truth = {0: [A], 12: [A, (30.0, 30.0, 40.0, 40.0)], 24: [A]}  # frame 24: nothing predicted
    s = ev.score_frames(pred, truth)
    assert (s["truePositives"], s["falsePositives"], s["falseNegatives"]) == (2, 1, 2)
    assert s["precision"] == pytest.approx(0.667) and s["recall"] == 0.5
    assert s["labelledFrames"] == 3
    assert s["playersFoundPerFrame"] == 1.0 and s["playersLabelledPerFrame"] == pytest.approx(1.33)


def test_precision_recall_is_none_without_boxes():
    assert ev.precision_recall(0, 0, 0) == {"precision": None, "recall": None}
    assert ev.precision_recall(0, 3, 0) == {"precision": 0.0, "recall": None}


def test_only_done_frames_are_scored_and_labels_scale_to_the_overlay():
    labels = {
        "imageSize": [640, 360],
        "frames": {"0": [[0, 0, 5, 5]], "30": [[10, 10, 20, 20]]},
        "done": [30],
    }
    assert ev.label_boxes(labels, [1280, 720]) == {30: [(20.0, 20.0, 40.0, 40.0)]}
    assert ev.label_boxes({"frames": labels["frames"]}, None).keys() == {0, 30}  # no `done`: all frames
    assert ev.label_boxes({}, None) == {}


def overlay_job():
    return {
        "jobId": "job-abc",
        "overlay": {
            "fps": 12,
            "frames": 3,
            "imageSize": [1280, 720],
            "players": {
                "o1": {"team": "offense", "boxes": [[0, 0, 10, 10], None, [0, 0, 10, 10]]},
                "d1": {"team": "defense", "boxes": [[100, 100, 120, 140], None, None]},
            },
        },
        "playDna": {
            "meta": {
                "samFrames": 240,
                "ballFrames": 200,
                "ballTracked": True,
                "cameraMotion": {"framesTracked": 236, "framesFallback": 4},
                "timings": {"frames": 2.5, "samPlayers": 20, "samBall": 10, "track": 30, "camera": 0.5,
                            "build": 1.0, "name": 15.0},
            }
        },
    }  # fmt: skip


def test_clip_report_reads_the_job_outputs():
    labels = {"frames": {"0": [[0, 0, 10, 10]], "1": [[0, 0, 10, 10]]}, "done": [0, 1]}
    r = ev.clip_report("c.mp4", {"jobId": "job-abc", "wallSeconds": 80.0}, overlay_job(), labels)
    assert (r["truePositives"], r["falsePositives"], r["falseNegatives"]) == (1, 1, 1)
    assert r["cameraFallbackFrames"] == 4 and r["ballFrames"] == 200
    assert r["serverSeconds"] == 49.0 and r["wallSeconds"] == 80.0
    assert r["samCostUsd"] == pytest.approx(0.096)  # 2 calls x 240 frames x $0.20 / 1000


def test_overall_pools_counts_across_clips():
    a = ev.clip_report("a.mp4", {"wallSeconds": 80.0}, overlay_job(), {"frames": {"0": [[0, 0, 10, 10]]}})
    b = ev.clip_report("b.mp4", {"wallSeconds": 100.0}, overlay_job(), {"frames": {"1": [[0, 0, 10, 10]]}})
    o = ev.overall([a, b])
    assert (o["truePositives"], o["falsePositives"], o["falseNegatives"]) == (1, 1, 1)
    assert o["precision"] == 0.5 and o["recall"] == 0.5
    assert o["labelledFrames"] == 2 and o["wallSecondsMean"] == 90.0
    assert o["samCostUsdTotal"] == pytest.approx(0.192)
    md = ev.summary_md({"generatedAt": "2026-10-28", "overall": o, "clips": [a, b], "notLabelled": ["c.mp4"]})
    assert "| **overall** | 2 | 0.5 | 0.5 |" in md and "c.mp4" in md


def test_eval_frames_every_two_and_a_half_seconds():
    assert jobcache.eval_frames(12, 240) == [15, 45, 75, 105, 135, 165, 195, 225]
    assert jobcache.eval_frames(12, 70) == [15, 45]


def test_no_labels_means_nothing_measured(tmp_path, monkeypatch, capsys):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"backend": "", "clips": {"x.mp4": {"jobId": "job-x"}}}), encoding="utf-8")
    monkeypatch.setattr(jobcache, "JOBS_FILE", jobs)
    monkeypatch.setattr(jobcache, "LABELS", tmp_path / "labels")
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    assert ev.run_summary(None, False) == 0
    assert "nothing measured" in capsys.readouterr().out
    assert not (tmp_path / "eval" / "results" / "summary.json").exists()
