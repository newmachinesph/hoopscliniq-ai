# Eval harness

Measures the pipeline's player boxes against checked reference boxes. Each label file says who checked it
(`labelledBy`; absent means a person). The 2026-10-28 labels were checked by an AI vision model, not a person.
Nothing is measured until labels exist.

## Eval set

Five 20-second live-play clips (half-court offense, game clock running) from the three Hazen Union games, in
`../clips/eval/` (why each was chosen: `../clips/README.md`). Each ran once through the deployed backend with
`scripts/run_clip.py`, without a coach calibration (calibration moves court positions, not boxes).
`eval/jobs.json` maps each clip to its job id and the client wall time of that run.

## Label (about 30 minutes)

```
python -m eval.label_server --clips ../clips/eval      # then open http://127.0.0.1:8765
```

- On first start it fetches each job view and `clip.mp4` from the backend once and cuts 8 frames per clip (one
  every 2.5 s, at 1.25 s, 3.75 s, ... 18.75 s; frame 0 is skipped because SAM is still picking players up there)
  to JPEG under `eval/cache/` (gitignored). After that it runs offline from the cache.
- Every frame starts with the job's overlay player boxes. Fix them: drag a box to move it, drag a corner to resize,
  click a box and press Delete to remove a false one, press N and drag to add a missed player, Enter to mark the
  frame done and go to the next. Arrow keys move between frames (left/right) and clips (up/down); Ctrl+Z undoes.
- Box every on-court player, both teams. Not referees, coaches or bench players.
- Every change is saved to `eval/labels/<clip>.json` at once. Only frames marked done count.
- Standard library only (no install); ffmpeg comes from `imageio-ffmpeg` (dev extra) or PATH for the one-time cut.

## Measure

```
python -m eval.run           # every clip in eval/jobs.json with done frames -> eval/results/summary.{json,md}
python -m eval.run --clip ../clips/<file>.mp4 --stream ../clips/sam-girls-stream.txt --fps 60   # one raw stream
```

- Reads each job's stored outputs from the cache (`--refresh` fetches them again): the overlay player boxes (the
  10 kept tracks the demo draws, so a person SAM saw but the pipeline dropped counts as a miss) and `playDna.meta`.
- Per clip and overall: player box precision and recall at IoU >= 0.5 (greedy one-to-one matching, best IoU
  first, pooled over all done frames), players found vs labelled per frame, camera fallback frames
  (`cameraMotion.framesFallback`), ball frames (`meta.ballFrames`), wall time (upload to `done`, from
  `eval/jobs.json`), server time (sum of `meta.timings` steps) and a SAM cost estimate (players + ball calls at
  $0.20 per 1,000 frames; Muse Spark naming and Modal compute not included).
- With no done frames it prints "nothing measured", writes nothing and exits 0.
- `--clip` mode: boxes from `--stream` (a saved raw SAM event stream, or a job's `sam.txt`), or from a fresh SAM
  call on the clip re-encoded to 12 fps (needs `META_API_KEY` in `.env`). Results go to `eval/results/<clip>.json`.

## Label format (`eval/labels/<clip>.json`)

```json
{
  "clip": "hazen-girls-lakeregion-2026-02-11-t2940.mp4",
  "jobId": "job-...",
  "fps": 12,
  "imageSize": [1280, 720],
  "frames": { "15": [[x1, y1, x2, y2], ...], "45": [[...]] },
  "done": [15, 45]
}
```

- `frames` keys are frame indices at `fps` (the rate SAM saw: 12 for the pipeline, 60 for the raw reference
  streams in `../clips/`). Boxes are pixels in `imageSize` (clip.mp4 pixels for the label server).
- `done` lists the frames a person checked. A file without `done` (older format) scores all its frames.
- `example.json` is an empty template.
