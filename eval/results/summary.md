# Eval summary

Generated 2026-10-28. 5 clips, 40 checked frames, labelled by an AI vision model correcting the pipeline's pre-filled boxes with a zoom tool, not checked by a person. Player boxes from the job overlay (the kept tracks), matched one-to-one to labels at IoU >= 0.5.

| Clip | Frames | Precision | Recall | TP / FP / FN | Found / labelled per frame | Camera fallback | Ball frames | Wall s | Server s | SAM $ |
|---|---|---|---|---|---|---|---|---|---|---|
| hazen-boys-stowe-2025-02-10-t1975.mp4 | 8 | 0.763 | 0.784 | 58 / 18 / 16 | 9.5 / 9.25 | 0 of 240 | 219 of 240 | 98.4 | 63.05 | 0.096 |
| hazen-boys-stowe-2025-02-10-t4375.mp4 | 8 | 0.871 | 0.938 | 61 / 9 / 4 | 8.75 / 8.12 | 0 of 240 | 201 of 240 | 98.3 | 55.8 | 0.096 |
| hazen-girls-lakeregion-2026-02-11-t2940.mp4 | 8 | 0.85 | 0.883 | 68 / 12 / 9 | 10.0 / 9.62 | 0 of 240 | 207 of 240 | 100.4 | 71.86 | 0.096 |
| hazen-girls-lakeregion-2026-02-11-t4860.mp4 | 8 | 0.85 | 0.861 | 68 / 12 / 11 | 10.0 / 9.88 | 0 of 240 | 197 of 240 | 82.2 | 66.21 | 0.096 |
| hazen-girls-randolph-2025-12-27-t4855.mp4 | 8 | 0.808 | 0.851 | 63 / 15 / 11 | 9.75 / 9.25 | 0 of 240 | 189 of 240 | 164.4 | 126.02 | 0.096 |
| **overall** | 40 | 0.828 | 0.862 | 318 / 66 / 51 | 9.6 / 9.22 | 0 of 1200 | 1013 of 1200 | 108.74 | 76.59 | 0.48 |

- Wall s: upload to `done` as seen by `scripts/run_clip.py` (overall: mean). Server s: sum of the server steps (overall: mean).
- SAM $: SAM 3.1 frames (players + ball calls) at $0.20 per 1,000 frames; Muse Spark naming and Modal compute are not included (overall: total).
- Camera fallback and ball frames count every clip frame, not only the labelled ones.
