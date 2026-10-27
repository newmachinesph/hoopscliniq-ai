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
pip install -e ".[dev]"            # or: pip install fastapi openai pydantic numpy pillow python-multipart sentry-sdk modal pytest ruff httpx python-dotenv imageio-ffmpeg
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

Storage: Modal Volume `hoopscliniq-film`, `jobs/<jobId>/{clip.mp4, job.json, sam.txt, play.json}`.

## API

Every response carries `Cache-Control: no-store` and `x-demo: live`. Errors are `{ "error": "..." }`.

| Route | Request | Response |
|---|---|---|
| `GET /health` | | `{ ok: true, version, modelsReachable }` (models check cached 60 s) |
| `POST /film/upload` | multipart `file` (mp4/mov/webm, ≤ 200 MB, ≤ 60 s), optional `calibration` JSON string | `202 { jobId, status: "queued", statusUrl: "/film/jobs/<jobId>" }` |
| `GET /film/jobs/{jobId}` | | `{ jobId, status, finishedAt, steps[], suggestedName, playDna, error }`; `400` malformed id, `404` unknown |
| `POST /cliniq/ask` | `{ question }` (≤ 500 chars) | `{ question, answer: { body[], citations[], confidence, credits }, noSource }`; `400` bad body |

`calibration` is `[[x,y] x4]` or `{ "points": [[x,y] x4], "imageSize": [w,h] }`: the half-court corners in the
order baseline-left, baseline-right, halfcourt-right, halfcourt-left, mapped to court feet (0,0), (50,0),
(50,47), (0,47). Without `imageSize` the points are in the uploaded video's pixels. Without a calibration the
pipeline uses `data/calibration/hazen-union.json`, read by eye off one frame of the Hazen Union gym and marked
`"calibration": "estimated"` in the result.

## Pipeline

1. **frames** (Modal): ffmpeg re-encodes to 12 fps H.264, ≤ 1280 px wide, no audio, `+faststart`.
2. **track** (SAM 3.1): `POST /v1/files` then a streamed `POST /v1/responses` with the prompt "basketball player";
   the raw text is saved as `sam.txt` and parsed into boxes per frame (masks ignored). SAM omits frames with no
   match at all, so a missing frame tag counts as zero players.
3. **geometry**: box foot point → homography → court feet, clamped to the court, 5-frame moving average, gaps
   ≤ 6 frames interpolated (longer gaps and the edges hold the nearest position; `meta.coverage` says how much
   of each track SAM actually saw). The 10 longest tracks are kept.
4. **teams**: mean RGB of the top 40 % of each box over its first 10 frames, 2-means into two teams; offense is
   the team with more players in the attacked half at the clip midpoint.
5. **play.json**: play-dna v2 plus a backend-only `meta` block (`playersFound`, `calibration`, `samFrames`,
   `imageSize`, `coverage`, `samObjectIds`, `timings`).
6. **name** (Muse Spark 1.3, structured output): the playbook plus a per-second track summary in, `{ playId,
   title, confidence, why }` out. `needsReview` is always true.

ClinIQ answers only from `data/world.json` (the demo's fictional Lakeshore season, generated from the demo's
mocks so every id and href matches the app). The post-check drops citations whose href is not in the world,
renumbers the `[n]` markers, and returns `noSource: true` when any paragraph is left without a marker.

## Measured

Only what the deploy-and-verify run on 2026-10-28 produced (one run of each; nothing here is averaged).

**Clip run** (`scripts/run_clip.py ../clips/hazen-girls-lakeregion-2026-02-11_t1200.mp4`, job `job-vgloz5z3jtki`):

| Step | Server time |
|---|---|
| frames (ffmpeg re-encode to 12 fps, 1280×720) | 3.69 s |
| track (SAM 3.1 upload + stream) | 31.55 s |
| geometry + teams + play.json | 1.69 s |
| name (Muse Spark 1.3) | 47.67 s |
| Wall clock, upload to `done`, seen by the client | 108.1 s |

- 240 frames sent to SAM, 240 frames back; `playersFound` 10 (5 offense, 5 defense by jersey color).
- Share of frames SAM saw each kept track: 0.99, 0.98, 0.93, 0.81, 0.53 (offense); 0.98, 0.88, 0.76, 0.75, 0.24
  (defense).
- `suggestedName`: `overload-131` "1-3-1 Overload", confidence medium, needsReview true. This clip is a real
  Hazen Union girls game and the playbook belongs to the fictional demo team, so the name is a model guess with
  nothing to check it against; it is not a measured accuracy.
- Court positions use the estimated Hazen calibration from a different camera position (the boys' game), and the
  girls' broadcast camera pans, so positions in feet are not reliable for this clip.

**ClinIQ** (`scripts/smoke.py`, deployed endpoint):

- "What did we run against Westview's zone and what worked?" → 200 in 22.1 s, `noSource: false`, 9 citations,
  all `/film/...` or `/plays/...`; every number in the answer matches `data/world.json`.
- "What is the best recipe for banana bread?" → 200 in 9.4 s, `noSource: true`, body "I have no source for that
  in your film.", 0 credits.

**Eval harness**: no hand labels exist yet, so precision and recall are not measured. Per-frame players found on
the saved 60 fps reference streams: girls clip min 0 / median 9 / max 12, boys pregame min 6 / median 7 / max 12
(`eval/results/`).

## Not working yet

- No hand-labelled frames, so SAM box precision/recall is unmeasured.
- Court mapping is a single fixed homography: wrong whenever the camera pans or sits somewhere other than the
  spot the calibration was read from. No per-frame court registration.
- Ball, holder, events, actions and areas in play-dna are empty (not tracked).
- Team split is jersey color only; referees or bench players caught in the top 10 tracks get a team anyway.

## Attribution

Test clips: Hardwick Community Television (HCTV), Hazen Union, Vermont, via Wikimedia Commons, **CC BY 4.0**:
[Boys Varsity Basketball v. Stowe - Feb. 10, 2025](https://commons.wikimedia.org/wiki/File:Boys_Varsity_Basketball_v._Stowe_-_Feb._10,_2025.webm)
and [Girls Varsity Basketball v. Lake Region - February 11, 2026](https://commons.wikimedia.org/wiki/File:Girls_Varsity_Basketball_v._Lake_Region_-_February_11,_2026.webm).
20-second cuts, re-encoded. See `../clips/README.md`.
