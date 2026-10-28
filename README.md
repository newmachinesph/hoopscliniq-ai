# hoopscliniq-ai

Backend for the Hoops ClinIQ hackathon demo (`../hoopscliniq-demo`): a coach's phone video of a basketball
possession goes in, tracked players and a suggested play name come out, and ClinIQ answers questions about the
team's film with a citation for every claim. Meta Global AI Developer Hackathon 2026, Indie developer track.

- **Modal** runs it: one FastAPI web endpoint and one CPU function per film job (ffmpeg in the image).
- **SAM 3.1** (Meta Model API) tracks the players; boxes become court positions in feet (play-dna v2 JSON).
- **Muse Spark 1.3** (Meta Model API) names the play from the coach's playbook and answers ClinIQ questions
  with tool calls (`fetch_play`, `fetch_clip`, `fetch_stat`). No source, no answer.

The full spec is [SPEC.md](SPEC.md).

## Run it

```bash
pip install -e ".[dev]"            # or: pip install fastapi openai pydantic numpy pillow python-multipart sentry-sdk modal opencv-python-headless pytest ruff httpx python-dotenv imageio-ffmpeg
pytest -q && ruff check .

python -m modal serve app/modal_app.py    # dev server with a temporary URL, hot reload
python -m modal deploy app/modal_app.py   # the deployed app `hoopscliniq-ai`

python scripts/run_clip.py ../clips/hazen-girls-lakeregion-2026-02-11_t1200.mp4 --url <web url>
python scripts/smoke.py "What did we run against Westview's zone and what worked?" --url <web url>
python -m eval.run --clip ../clips/<clip>.mp4 [--labels eval/labels/<clip>.json] [--stream <saved SAM stream> --fps 60]
python scripts/build_world.py            # rebuild data/world.json from the demo's mocks (needs Node 22.6+)
```

Deployed: `https://del-rosario-llino-g--hoopscliniq-ai-web.us-east.modal.run`

## Environment

| Variable | Where | Use |
|---|---|---|
| `META_API_KEY` | Modal Secret `hoopscliniq-keys`, local `.env` | Meta Model API (`https://api.meta.ai/v1`): SAM 3.1 and Muse Spark 1.3 |
| `BASETEN_API_KEY`, `BASETEN_MODEL_ID` | `hoopscliniq-keys`, `.env` | Optional ClinIQ provider (Muse Spark Open on Baseten) |
| `CLINIQ_PROVIDER` | function env | `meta` (default) or `baseten` |
| `SENTRY_DSN` | Modal Secret `hoopscliniq-sentry`, `.env` | Errors from the web endpoint and the job function |
| `HOOPSCLINIQ_API_URL` | local shell | Deployed URL for `scripts/run_clip.py` and `scripts/smoke.py` |

Storage: Modal Volume `hoopscliniq-film`, `jobs/<jobId>/{clip.mp4, job.json, sam.txt, sam-ball.txt, camera.json,
calibration.json, play.json, overlay.json}`.

## API

Every response carries `Cache-Control: no-store` (except the clip) and `x-demo: live`. Errors are `{ "error": "..." }`.

| Route | Request | Response |
|---|---|---|
| `GET /health` | | `{ ok: true, version, modelsReachable }` (models check cached 60 s) |
| `POST /film/upload` | multipart `file` (mp4/mov/webm, ≤ 200 MB, ≤ 60 s), optional `calibration` JSON string | `202 { jobId, status: "queued", statusUrl: "/film/jobs/<jobId>" }` |
| `GET /film/jobs/{jobId}` | | `{ jobId, status, finishedAt, steps[], suggestedName, playDna, error, clipUrl, overlay, calibration, cameraMotion }`; `400` malformed id, `404` unknown |
| `POST /film/jobs/{jobId}/calibrate` | `{ frame, points: [[x,y] x4] }` in clip.mp4 pixels | `200` job view; `400` bad input, `404` unknown, `409` not done |
| `GET /film/jobs` | | `{ jobs: [{ jobId, finishedAt, title, playersFound }] }`: status `done` only, newest first, max 10; `title` = `suggestedName.title` |
| `GET /film/jobs/{jobId}/clip` | | `clip.mp4` (`video/mp4`), HTTP Range (206), `Cache-Control: public, max-age=3600`, `Access-Control-Allow-Origin: *`; `404 { error }` when missing |
| `POST /cliniq/ask` | `{ question }` (≤ 500 chars) | `{ question, answer: { body[], citations[], confidence, credits }, noSource }`; `400` bad body |

`calibration` is `[[x,y] x4]` or `{ "points": [[x,y] x4], "imageSize": [w,h] }`: the half-court corners in the
order baseline-left, baseline-right, halfcourt-right, halfcourt-left, mapped to court feet (0,0), (50,0),
(50,47), (0,47). Without `imageSize` the points are in the uploaded video's pixels. Without a calibration the
pipeline uses `data/calibration/hazen-union.json`, read by eye off one frame of the Hazen Union gym and marked
`"calibration": "estimated"` in the result.

Version 0.3.0 adds (all additive; older jobs return `null` or omit them):

- `overlay.ball`: the chosen ball box per frame in clip.mp4 pixels, or null.
- `calibration`: `{ source: "estimated" | "coach", frame, points }` in clip.mp4 pixels. The built-in estimate's
  baseline-right corner sits past the right edge (x ≈ 1422 on a 1280 px clip) because that corner is off-screen in
  the gym camera and was extrapolated from the lines.
- `cameraMotion`: `{ framesTracked, framesFallback }` (also `playDna.meta.cameraMotion`).
- `playDna.tracks.ball` (`[x, y, 0]` per frame, null when unseen), `tracks.holder` (player id or null), `events`.
- `POST /film/jobs/{jobId}/calibrate`: the coach's four corners (baseline-left, baseline-right, half-court-line
  right sideline, half-court-line left sideline) on frame `frame`. It re-projects the stored tracks with the stored
  camera motion (no new SAM call), rebuilds play.json and the overlay, re-runs naming and returns the job view.
  Points must form a convex quadrilateral in that order and lie inside the image or at most half a frame outside it
  (for a corner the camera does not show). Naming inside calibrate is capped at 40 s; past that the previous name
  stays.

## Pipeline

1. **frames** (Modal): ffmpeg re-encodes to 12 fps H.264, ≤ 1280 px wide, no audio, `+faststart`.
2. **track** (SAM 3.1): `POST /v1/files` then a streamed `POST /v1/responses` with the prompt "basketball player";
   the raw text is saved as `sam.txt` and parsed into boxes per frame (masks ignored). SAM omits frames with no
   match at all, so a missing frame tag counts as zero players. In parallel, on the same file id: a second SAM
   call with the prompt "basketball" (`sam-ball.txt`; re-upload only if the file expired; one retry; a failure
   leaves the ball empty) and the camera features (ORB on half-size grey frames).
   **Camera motion**: keypoints inside any SAM player or ball box (+10 %) are dropped; the rest are matched to a
   keyframe (ratio test, RANSAC homography, ≥ 30 inliers). The keyframe moves every 12 frames (1 s) or when inliers
   drop under 80; a frame that cannot match the keyframe is chained from the previous frame. A frame with no good
   match falls back to the static calibration and is counted in `framesFallback`. `camera.json` stores frame i →
   frame 0 for every frame, so calibrating later only re-projects: court = `H_cal · H_(i→calFrame) · p`.
3. **geometry**: box foot point → per-frame homography → court feet, clamped to the court, 5-frame moving average, gaps
   ≤ 6 frames interpolated (longer gaps and the edges hold the nearest position; `meta.coverage` says how much
   of each track SAM actually saw). The 10 longest tracks are kept.
4. **teams**: mean RGB of the top 40 % of each box over its first 10 frames, 2-means into two teams; offense is
   the team with more players in the attacked half at the clip midpoint.
5. **play.json**: play-dna v2 plus a backend-only `meta` block (`playersFound`, `calibration`, `samFrames`,
   `imageSize`, `coverage`, `samObjectIds`, `timings`, and since 0.3.0 `calibrationFrame`, `cameraMotion`,
   `ballTracked`, `ballFrames`, `holderFrames`).
6. **name** (Muse Spark 1.3, structured output): the playbook, a per-second track summary and the detected passes
   and shots (with time stamps) in, `{ playId, title, confidence, why }` out. `needsReview` is always true.

## Ball and events

Simple, deterministic heuristics in `app/events.py` (thresholds in `app/settings.py`, unit-tested on synthetic
tracks). Every event carries `detect: "heuristic"` and a confidence of `low` or `medium`, never `high`.

- **Ball per frame**: follow the same SAM object as the last ball seen (up to 6 frames back), else the nearest
  candidate if it moved at most 6 % of the frame width per frame. Otherwise, or when the followed ball has been away
  from every player for 12 frames while another ball is near a player, start again on a ball near a player (not one
  on the scorer's table or a rack), then the one SAM saw longest. Court position = box centre projected to the
  floor (`z = 0`), so a ball in the air lands at the wrong spot.
- **Holder**: the kept player whose box contains or is nearest the ball centre in the image, if the gap is at most
  0.25 × that player's box height; else null. Unseen ball = no holder.
- **Segments**: runs of one holder (gaps of ≤ 2 frames allowed), runs shorter than 3 frames dropped, same-holder runs
  ≤ 5 frames apart joined.
- **Shot**: after a segment, before anyone holds the ball again (max 1.5 s), the seen ball comes within 4 ft of the
  rim (25, 5.25) → `medium`; or it rises above the holder's box top (by 10 % of the box height) and ends ≥ 3 ft closer
  to the rim → `low`. `made` is always null.
- **Pass**: otherwise, the next segment is a different player of the same team within 1.5 s. `medium` when the ball
  was seen in between, the gap is ≤ 0.75 s and both held it ≥ 6 frames; else `low` (instant hand-offs are `low`).
- No ball seen = no holder and no events.

ClinIQ answers only from `data/world.json` (the demo's fictional Lakeshore season, generated from the demo's
mocks so every id and href matches the app). The post-check drops citations whose href is not in the world,
renumbers the `[n]` markers, and returns `noSource: true` when any paragraph is left without a marker.

## Measured

Only what the deploy-and-verify runs on 2026-10-28 produced (one run each; nothing here is averaged). Version 0.3.0.

**Clip runs** (`scripts/run_clip.py`, deployed endpoint, server step times from `meta.timings`):

| Step | Girls `job-wrpgt63evkhe` | Boys `job-brnjmhuduw6a` |
|---|---|---|
| frames (ffmpeg re-encode to 12 fps, 1280×720) | 2.72 s | 2.63 s |
| track, wall time of the parallel block (upload + SAM players + SAM ball + camera features) | 32.09 s | 26.26 s |
| — SAM players / SAM ball / camera features (run side by side) | 26.56 / 13.18 / 2.00 s | 23.54 / 18.56 / 2.28 s |
| camera motion (matching, after SAM) | 0.49 s | 0.48 s |
| geometry + teams + ball + events + play.json | 1.75 s | 1.14 s |
| name (Muse Spark 1.3) | 14.23 s | 22.63 s |
| Wall clock, upload to `done`, seen by the client | 95.7 s | 79.8 s |

- Camera motion: 240 of 240 frames followed, 0 fallback, on both clips. Not checked against ground truth: no
  labelled camera poses exist.
- Ball: girls 193 of 240 frames with a ball, 138 frames with a holder, 1 event (shot by o4, medium). Boys (pregame
  warm-up, many balls on court and on the scorer's table) 240 of 240 frames with a ball, 203 with a holder,
  5 passes (2 medium, 3 low). Events are heuristic and unverified against the video.
- `suggestedName`: girls "Primary Transition" (medium); boys "Primary Transition" (medium) after the coach
  calibration below. The playbook belongs to the fictional demo team, so these are model guesses, not accuracy.
- One earlier girls run lost the ball to a SAM `server_error` mid-stream; the ball call now retries once.

**Coach calibration** (boys job, `POST /calibrate`, frame 0). The corners were read by eye off frame 0 of clip.mp4
(rendered with ffmpeg), using the lane and three-point arc to check them: `[[838,281], [1620,495], [258,652],
[271,305]]`. The baseline-right corner is off-screen and was extrapolated from the baseline and the near sideline.

| Frame 0, 7 players SAM saw | Estimated calibration | Coach calibration |
|---|---|---|
| Foot points inside the court (0–50 × 0–84 ft) | 6 of 7 (o1 projected 2.4 ft behind the baseline) | 7 of 7 |
| The two players standing in the painted lane (o1, o5) projected inside the lane (x 19–31, y 0–19) | 1 of 2 | 2 of 2 |

Calibrate latency, end to end over HTTP: 36.6 s (re-projection 0.41 s, naming 31.83 s). Two earlier calls, before
naming was capped at 40 s, took 54.5 s and 109.6 s; almost all of it is the Muse Spark call.

**ClinIQ** (`scripts/smoke.py`, deployed endpoint, 0.2.0 run, unchanged code path):

- "What did we run against Westview's zone and what worked?" → 200 in 22.1 s, `noSource: false`, 9 citations,
  all `/film/...` or `/plays/...`; every number in the answer matches `data/world.json`.
- "What is the best recipe for banana bread?" → 200 in 9.4 s, `noSource: true`, body "I have no source for that
  in your film.", 0 credits.

**Eval harness**: no hand labels exist yet, so precision and recall are not measured. Per-frame players found on
the saved 60 fps reference streams: girls clip min 0 / median 9 / max 12, boys pregame min 6 / median 7 / max 12
(`eval/results/`).

## Not working yet

- No hand-labelled frames, so SAM box precision/recall, ball accuracy and event accuracy are unmeasured.
- The ball is projected as if it were on the floor; a ball in the air lands at the wrong court spot, so "within
  4 ft of the rim" rarely fires for real jump shots and the rise test does most of the work.
- Shots never say made or missed. Screens, cuts, actions and areas stay empty.
- In warm-ups with many balls, "the" ball is whichever one the continuity rule follows.
- Camera motion drifts a little at every keyframe change (about 1 s apart) and does not handle cuts beyond falling
  back; broadcast overlays (score bug, captions) that stay fixed on screen can pull the estimate toward "no motion".
- Without a coach calibration, both sample clips use the Hazen boys' estimate, which does not fit the girls'
  broadcast camera.
- Team split is jersey color only; referees, coaches or bench players caught in the top 10 tracks get a team anyway
  (the boys clip's two coaches on the floor are tracked).
- Naming inside `POST /calibrate` is cut off at 40 s; when that happens the previous name stays.

## Attribution

Test clips: Hardwick Community Television (HCTV), Hazen Union, Vermont, via Wikimedia Commons, **CC BY 4.0**:
[Boys Varsity Basketball v. Stowe - Feb. 10, 2025](https://commons.wikimedia.org/wiki/File:Boys_Varsity_Basketball_v._Stowe_-_Feb._10,_2025.webm)
and [Girls Varsity Basketball v. Lake Region - February 11, 2026](https://commons.wikimedia.org/wiki/File:Girls_Varsity_Basketball_v._Lake_Region_-_February_11,_2026.webm).
20-second cuts, re-encoded. See `../clips/README.md`.
