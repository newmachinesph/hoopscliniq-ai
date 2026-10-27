# Eval harness

Measures SAM 3.1 player boxes against hand labels. Nothing is measured until labels exist.

```
python -m eval.run --clip ../clips/<file>.mp4 --labels eval/labels/<clip>.json
python -m eval.run --clip ../clips/<file>.mp4 --stream ../clips/sam-girls-stream.txt --fps 60   # no API call
```

- Boxes come from `--stream` (a saved raw SAM event stream, or a job's `sam.txt`), or from a fresh SAM call on the
  clip re-encoded to 12 fps (needs `META_API_KEY` in `.env`; about $0.05 per 20 s clip).
- With labels: precision and recall of player boxes at IoU >= 0.5 (greedy one-to-one matching) over the
  labelled frames. Without labels: players found per frame (min / median / max) and the number of SAM object ids,
  and it prints "no labels yet, nothing measured".
- Results go to `eval/results/<clip>.json`; `scripts/run_clip.py` also saves `eval/results/<clip>.play.json`.

## Label format (`eval/labels/<clip>.json`)

```json
{
  "clip": "hazen-girls-lakeregion-2026-02-11_t1200.mp4",
  "fps": 12,
  "imageSize": [1280, 720],
  "frames": { "0": [[x1, y1, x2, y2], ...], "12": [[...]] }
}
```

- `frames` keys are frame indices at `fps` (the rate SAM saw: 12 for the pipeline, 60 for the raw reference
  streams in `../clips/`). Label only on-court players (not referees, coaches or bench).
- Boxes are pixels in `imageSize`; they are scaled to SAM's frame size before matching.
- `example.json` is an empty template: no clip is labelled yet.
