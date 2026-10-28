# hoopscliniq-ai: backend spec (hackathon build)

Hoops ClinIQ turns a coach's phone video of a basketball game into tagged plays and answers that
cite the exact clip. This repo is the backend for the demo in `../hoopscliniq-demo` (Next 16).
Meta Global AI Developer Hackathon 2026, track Indie developer, due Nov 4 2026 9 PM ET.

## Non-negotiables

- **Python 3.12, FastAPI, deployed on Modal** as one app `hoopscliniq-ai`. No database. No auth.
- **Meta Model API** (OpenAI-compatible, base `https://api.meta.ai/v1`, key `META_API_KEY`):
  `sam-3.1` for tracking, `muse-spark-1.3` for play naming and ClinIQ answers. Use the `openai`
  Python SDK with `base_url` set. Never call Anthropic or OpenAI models.
- **Secrets** come from the Modal Secret `hoopscliniq-keys` (already exists: `META_API_KEY`,
  `BASETEN_API_KEY`, `BASETEN_MODEL_ID`) plus a new Modal Secret `hoopscliniq-sentry` with
  `SENTRY_DSN` (create it from `.env` with `python -m modal secret create hoopscliniq-sentry SENTRY_DSN=...`).
  Locally, `.env` (gitignored) holds the same keys for `scripts/` and tests. Never print keys.
- **Storage**: Modal Volume `hoopscliniq-film`, layout `jobs/<jobId>/{clip.mp4,job.json,sam.txt,sam-ball.txt,camera.json,calibration.json,play.json,overlay.json}`.
- **Every response** carries `Cache-Control: no-store` and `x-demo: live`. JSON shapes below are a
  contract with the frontend: keep the keys exactly.
- **Honesty**: anything not measured is reported as such. No invented numbers anywhere.
- Code style: ruff (line length 110), type hints, small functions, docstrings only where the why is
  not obvious. Tests with pytest for the pure functions (parsing, geometry, naming prompt builders);
  network calls are mocked in tests.

## Routes (FastAPI, mounted at `/`)

| Route | Request | Response |
|---|---|---|
| `GET /health` | | `{ ok: true, version, modelsReachable: bool }` (models check cached 60 s) |
| `POST /film/upload` | multipart `file` (mp4/mov/webm, ≤ 200 MB, ≤ 60 s), optional `calibration` (JSON string, see Geometry) | `202 { jobId, status: "queued", statusUrl: "/film/jobs/<jobId>" }` |
| `GET /film/jobs/{jobId}` | | `{ jobId, status, finishedAt, steps[], suggestedName, playDna, error, clipUrl, overlay, calibration, cameraMotion }` (`400` on a malformed id, `404` unknown) |
| `POST /film/jobs/{jobId}/calibrate` | `{ frame: int, points: [[x, y] x4] }` in clip.mp4 pixels | `200` job view; `400` bad input, `404` unknown job, `409` job not done (0.3.0) |
| `GET /film/jobs` | | `{ jobs: [{ jobId, finishedAt, title, playersFound }] }`: status `done` only, newest first, max 10; `title` = `suggestedName.title` |
| `GET /film/jobs/{jobId}/clip` | | `clip.mp4` (`video/mp4`), HTTP Range (206), `Cache-Control: public, max-age=3600`, `Access-Control-Allow-Origin: *`; `404 { error }` when missing |
| `POST /cliniq/ask` | `{ question: string ≤ 500 }` | `{ question, answer: { body: string[], citations: Citation[], confidence, credits }, noSource }` (`400` on bad body) |

`jobId` = `job-` + 12 lowercase base32 chars. `status` ∈ `queued | frames | track | name | done | failed`.
`steps` = `[{ id, label, engine, status }]` for ids `upload, frames, track, name, review, cite` with
engine labels exactly `Phone or file`, `Modal`, `SAM 3.1`, `Muse Spark`, `Coach confirms`, `ClinIQ`;
per-step `status` ∈ `done | running | queued | failed | planned` (`review` and `cite` are always `planned`
from the backend's view; the frontend owns them).
`suggestedName` = `{ playId, title, confidence: "high"|"medium"|"low", why, needsReview: true }` or null until named.
`clipUrl` = absolute URL of `GET /film/jobs/{jobId}/clip`, null until the frames step is done.
`overlay` = `{ fps, frames, imageSize: [w, h], players: { <id>: { team, boxes: ([x1,y1,x2,y2] | null)[] } } }` or null
until tracking is done: SAM boxes of the kept players in clip.mp4 pixels (integers), one entry per clip frame, null
where SAM did not see that player; same ids and frame index as `playDna.tracks.players`. Stored as `overlay.json`.
Since 0.3.0 `overlay.ball` = `([x1,y1,x2,y2] | null)[]`: the chosen ball box per frame in clip.mp4 pixels (missing
on older jobs and when the ball call failed).
`calibration` (0.3.0) = `{ source: "estimated" | "coach", frame, points: [[x,y] x4] }` in clip.mp4 pixels, or null
on older jobs. The built-in estimate's baseline-right corner lies past the right edge (x ≈ 1422 on a 1280 clip): it
is off-screen in the gym camera and was extrapolated. `cameraMotion` (0.3.0) = `{ framesTracked, framesFallback }`
or null; the same object is in `playDna.meta.cameraMotion`.

### `POST /film/jobs/{jobId}/calibrate` (0.3.0)

Body `{ frame, points }`: the four corners of the attacked half as seen in clip.mp4 frame `frame`, in the order
baseline-left, baseline-right, half-court-line right sideline, half-court-line left sideline. Validation: `frame`
is an integer in `[0, frames)`; 4 numeric points; a convex quadrilateral in that order (either winding); every
point inside the image or at most 50 % of the frame size outside it (a side-on camera often hides the near
baseline corner, which the coach can only extrapolate; the demo UI only taps inside the video). The backend
re-projects the stored SAM tracks with the stored camera motion (no SAM call), keeps the stored team split,
rebuilds `play.json` and `overlay.json`, saves `calibration.json` (`source: "coach"`), re-runs naming (on a naming
error it keeps the previous name) and returns the job view. `meta.timings.calibrate` records the latency.
`Citation` = `{ kind: "play"|"clip"|"stat", label, href }` where `href` is a route in the demo app
(`/plays/<id>`, `/film/<id>`, `/study/results`). Only hrefs that exist in `data/world.json` may be emitted.

## Pipeline (one Modal function per job, GPU not required; CPU with 4 cores is enough)

1. **frames** (Modal): ffmpeg (installed in the image with `apt`) re-encodes the upload to
   `clip.mp4` at **12 fps**, H.264, `-movflags +faststart`, max 1280 px wide, no audio. Record
   `fps`, `frames`, `width`, `height`.
2. **track** (SAM 3.1): upload `clip.mp4` with `POST /v1/files` (`purpose=user_data`,
   `expires_after={"anchor":"created_at","seconds":86400}`), then `POST /v1/responses` with
   `model: "sam-3.1"`, content `[{type:"input_text", text:"basketball player"}, {type:"input_video", file_id}]`,
   `stream: true`, `Accept: text/event-stream`. Concatenate every `response.output_text.delta`.
   Save the raw text as `sam.txt`. Parse: frames are split by `<Nf>`; each match is
   `idx<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;...|>` in source pixels; `idx` is the object id,
   stable across frames. Ignore masks (do not store them in `play.json`). Verified on 2026-10-28:
   a 20 s 60 fps clip = 1200 frames, ~70 s, output 120–230 MB; at 12 fps expect ~15 s and ~30 MB.
   Reference streams and clips: `../clips/` (CC BY 4.0, attribution in `../clips/README.md`).
2b. **ball** (SAM 3.1, 0.3.0): a second streamed call with the prompt `"basketball"` on the same uploaded file id,
   run in parallel with the player call (re-upload only if the API says the file is gone; one retry on a stream
   error; a failure leaves the ball empty and the job continues). Raw text saved as `sam-ball.txt`.
2c. **camera motion** (OpenCV, 0.3.0): ORB features (1000 per frame, half-size grey frames, detected while SAM
   runs), keypoints inside any SAM player or ball box (+10 %) dropped, ratio-test matches to the current keyframe,
   RANSAC homography. The keyframe moves every 12 frames or when inliers fall under 80; when the keyframe match
   fails the frame is chained from the previous frame. Fewer than 30 inliers (or a degenerate fit) = fallback for
   that frame (no homography). Stored once per job as `camera.json` (`toFrame0`: frame i -> frame 0, clip.mp4
   pixels). Court position in frame i = `H_cal · inv(G_calFrame) · G_i · p`; a fallback frame uses `H_cal`.
3. **geometry** (pure Python, tested): foot point of each box = `((x1+x2)/2, y2)`. Map to court feet
   with a homography from `calibration` = 4 image points (px) of the half-court corners in order
   `[baseline-left, baseline-right, halfcourt-right, halfcourt-left]`, mapped to court feet
   `[(0,0), (50,0), (50,47), (0,47)]` (NFHS 84×50, x across 0–50, y out from the attacked baseline,
   rim at (25, 5.25)). If no calibration is given, use `data/calibration/hazen-union.json` (derive it
   once by reading the 4 corners off `../clips/sam-boys-frame0.png`, which is 1280 px wide: scale ×1.5
   to source pixels; it is approximate and must be marked `"source": "estimated"`). Clamp to the court,
   smooth each track with a 5-frame moving average, fill gaps ≤ 6 frames by linear interpolation.
   Keep the 10 tracks with most frames; mark `playersFound` = that count (may be < 10).
4. **teams**: mean RGB of the top 40 % of each box over its first 10 frames (read frames with
   ffmpeg → PNG → PIL); 2-means on those colors → team A/B. Offense = the team with more players
   in the attacked half (y < 47) at the clip midpoint. Ids `o1..o5`, `d1..d5` by track length.
5. **play.json** (play-dna v2, same schema as `../../frontends/hoopscliniq-app/src/lib/play-dna/horns-twist.play.json`):
   `version: 2, dictionary: "1.5", name, source: { clip: jobId, authored: false, fps, frames },
   court: { units:"ft", x:"across, 0-50", y:"out from the attacked baseline", rim:[25,5.25], dimensions:"NFHS" },
   players: { o1: {team:"offense"}, ... }, tracks: { players: { id: [[x,y], ...] per frame }, ball: [], holder: [], areas: [] },
   events: [], actions: [], moments: [{frame:0,label:"Start"}], fixes: []`. Add a backend-only block
   `meta: { playersFound, calibration: "given"|"estimated"|"coach", samFrames, imageSize:[w,h] }`.
   0.3.0 fills `tracks.ball` (`[x, y, 0]` per frame, the ball box centre projected to the floor, null when unseen),
   `tracks.holder` (player id or null per frame) and `events` (`{type:"pass", from, to, frame, end}` and
   `{type:"shot", from, frame, end, made:null}`, each with `detect:"heuristic"` and `confidence:"low"|"medium"`;
   rules in README "Ball and events"), and adds `meta.calibrationFrame`, `meta.cameraMotion`, `meta.ballTracked`,
   `meta.ballFrames`, `meta.holderFrames`.
6. **name** (Muse Spark 1.3, structured output; 0.3.0 adds the detected passes and shots with time stamps to the
   prompt): prompt = the coach's playbook (titles + one-line
   descriptions from `data/world.json`) and a compact track summary (each player's court position
   every second, rounded to 1 ft, plus start/end). `response_format` json_schema
   `{ playId: enum of playbook ids + "unknown", title, confidence: high|medium|low, why: ≤ 160 chars }`.
   `needsReview` is always true. If the model says `unknown`, `title` = "Unnamed play".
7. Write `job.json` after every step so `GET /film/jobs/{id}` is always consistent. On any exception:
   `status: "failed"`, `error` = a one-line friendly message, details to Sentry and logs.

## ClinIQ ask (Muse Spark 1.3 with tools)

- `data/world.json`: the demo's fictional world, hand-written from
  `../hoopscliniq-demo/src/mocks/{plays,film,insights,cliniq-answers}.ts`: ~10 plays
  (`id, title, description, timesRun, pointsPerPossession`), ~8 films
  (`id, opponent, date, result, possessions: [{ clipId, playId, outcome, note }]`), ~10 stats
  (`id, label, value, href:"/study/results"`). Opponents: Westview, Ridgeline, Cedar Park, Lakemont,
  Fairhaven, Oak Hollow, Brookside, Pinecrest, Harbor City, Northfield Prep. Keep ids identical to the
  demo's film and play ids so citation hrefs resolve (`/film/cedar-park`, `/plays/horns-twist`, ...).
- Tools: `fetch_play(playId)`, `fetch_clip(filmId, playId?)`, `fetch_stat(query)`. Run the tool loop
  (max 6 calls). System prompt: answer only from tool results; every paragraph ends with `[n]`
  markers that index the citations list; if nothing relevant is returned, say "I have no source for
  that in your film" and nothing else. Final turn uses structured output
  `{ body: string[], citations: [{kind,label,href}], confidence: high|medium|low|null }`.
- Post-check in code: drop citations whose `href` is not in the world; if a paragraph has no `[n]`
  or the citations list is empty → `noSource: true`, `body: ["I have no source for that in your film."]`,
  `citations: []`. `credits` = 1 per answer with sources, 0 otherwise.
- Baseten (Muse Spark Open) is an optional second provider behind `CLINIQ_PROVIDER=baseten`
  (`https://inference.baseten.co/v1`, model from `BASETEN_MODEL_ID`, same OpenAI-compatible calls).
  Default is `meta`.

## Eval harness (`eval/`)

- `eval/labels/<clip>.json`: `{ clip, fps, frames: { "<frameIdx>": [[x1,y1,x2,y2], ...] } }` hand labels
  (none exist yet; ship the format, a README and one empty example).
- `python -m eval.run --clip ../clips/<file>.mp4 --labels eval/labels/<clip>.json` → precision and
  recall of player boxes at IoU ≥ 0.5 on labelled frames, and players-found per frame stats. Prints a
  markdown table; writes `eval/results/<clip>.json`. With no labels it prints the per-frame stats only
  and says "no labels yet, nothing measured".

## Repo layout

```
app/            modal_app.py (Modal App, image, volume, web app mount), api.py (FastAPI), jobs.py (pipeline),
                sam.py (upload + stream + parse), geometry.py, teams.py, playdna.py, naming.py, cliniq.py,
                world.py (load data/world.json), models.py (pydantic), settings.py
data/           world.json, calibration/hazen-union.json
eval/           run.py, labels/, results/, README.md
scripts/        run_clip.py (local end-to-end on one clip against the deployed URL), smoke.py (health + ask)
tests/          test_sam_parse.py (uses a 2-frame fixture cut from ../clips/sam-boys-stream.txt), test_geometry.py,
                test_playdna.py, test_cliniq_postcheck.py
pyproject.toml  (uv or pip; ruff + pytest config), README.md, SPEC.md (this file), .env.example, .gitignore
```

## Deploy and verify (the implementer does all of this and reports the results)

1. `pytest -q` green, `ruff check .` clean.
2. `python -m modal deploy app/modal_app.py` → note the web URL. `GET /health` returns ok and `modelsReachable: true`.
3. `python scripts/run_clip.py ../clips/hazen-girls-lakeregion-2026-02-11_t1200.mp4` → upload, poll until
   `done`, print the step timings, `playersFound`, `suggestedName`, and save `play.json` to `eval/results/`.
4. `python scripts/smoke.py "What did we run against Westview's zone and what worked?"` → an answer with
   ≥ 1 citation whose href starts with `/film/` or `/plays/`; and a nonsense question → `noSource: true`.
5. README: what it is, how to run locally (`modal serve`), env vars, the API table, the attribution line
   for the Hazen clips, and a "Measured" section that only lists what step 3 and 4 produced.
6. Write the report to the path given in the task, then return `DONE|<path>`.

## Next.js side (separate task, `../hoopscliniq-demo`)

The three route handlers become proxies when `HOOPSCLINIQ_API_URL` is set: forward the request to
`<url>/film/upload`, `<url>/film/jobs/<id>`, `<url>/cliniq/ask`, pass the status and JSON through,
set `x-demo: live`; on network error or 5xx, fall back to the mock with `x-demo: mock` and a
`x-demo-fallback: <reason>` header. Unset → mock, unchanged. No other frontend change.
