# Gesture / Sign Language AAC — agent handoff

This file is the source of truth for **what this project is, what we already built, and how to keep going**. Another Cursor account will **not** see the original chat. Share this repo (and this file) instead.

Do **not** commit `backend/.env`. It contains API keys and the database password.

## What we are building

A local app for people who use sign language as AAC:

1. Webcam sees the hands (MediaPipe landmarks).
2. The app recognizes a sign (Yes, Hello, Clear, …).
3. Signs stack into a **sentence**.
4. The sentence is **spoken** out loud.

There is also an older **aviation** practice mode (exit pointing, seatbelt). Sign language Talk is the active product.

## How to run

- Backend: FastAPI on `http://127.0.0.1:8000` (`backend/.venv`, `uvicorn main:app --reload --host 0.0.0.0 --port 8000`)
- Frontend: Next.js on `http://localhost:3000`
- Postgres: local server, database **`gestures`**, port **`5433`** (this machine). Credentials live only in `backend/.env` (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`). Password may contain `@` — `app/db.py` URL-encodes it.

If History returns 503, Postgres login failed. Talk / Train can still run only if samples are already in the DB; new samples must save to Postgres.

## Product rules the user cares about

- **Closed fist (all fingers + thumb in) = Clear** — wipes the sentence and starts over. Not Yes.
- **Thumbs up = Yes.** Do not confuse with fist.
- **Still open palm = Hello.** **Side-to-side wave = Goodbye.** Wave needs real motion (wag), not a still palm.
- **Signs panel is on the left** so the camera stays large.
- User wants **real training**, not a fake progress UI. Add samples → Train now → Promote.
- After promote, Talk should use the **trained model** for still signs. Wave/Goodbye stays on motion rules.
- **All training data must live in Postgres**, per spoken name (Hello, Yes, …), so it can be reused later. Do not write new gold holds to `samples.jsonl`.

## Dual path: rules now, model when promoted

Until the user clicks **Promote**, Talk uses **hardcoded heuristics** in `backend/app/scoring/engine.py` (finger-extension rules + FastDTW for wave).

| Step | What happens | Talk changes? |
|---|---|---|
| Add samples | One confirmed hold is inserted into `training_samples` immediately | No |
| Train now | Softmax classifier (NumPy, not sklearn — Windows blocked sklearn DLLs) fits on gold holds with ≥3 samples on ≥2 signs | No (shadow model only) |
| Promote | `ml_runtime.source = model`. Still signs use the model if confidence ≥ 0.55, else rules. Wave always rules | Yes |

A Yes-only dataset cannot train (need 2 classes). A Yes-heavy dataset (e.g. 100 Yes + 6 Clear) will bias toward Yes if promoted. Collect similar counts per sign before promote.

## Signs (gesture key → spoken name)

| Key | Spoken |
|---|---|
| `thumbs_up` | Yes |
| `thumbs_down` | No |
| `open_palm` | Hello |
| `fist` | Clear |
| `pointing` | Help |
| `peace_sign` | Thank you |
| `please` | Please |
| `you` | You |
| `want` | Want |
| `okay` | Okay |
| `i_love_you` | I love you |
| `wave` | Goodbye |

Meanings: `backend/app/scoring/sign_meanings.py`.

## Where data lives (Postgres database `gestures`)

| Table | Purpose |
|---|---|
| `training_samples` | Every Add samples row. Filter `name = 'Yes'` or `gesture = 'thumbs_up'`. Includes `features`, wave `sequence`, and `landmarks` |
| `trained_models` | Each Train now bundle (`version` like `v7`, weights JSON) |
| `ml_runtime` | Singleton: rules vs promoted model |
| `recorded_references` | Optional 1-shot overlays (Record reference). Empty means hardcoded defaults |
| `attempts` | Live practice log for History |
| `session_recordings` | Session metadata. **Video files still on disk** under `backend/recordings/` |

Old file caches (`samples.jsonl`, `runtime.json`, `sign_clf_vN.json`, `recorded_references.json`) were imported once and archived as `*.imported`. Do not treat those files as the source of truth.

List/export samples: `GET /ml/samples?name=Yes`.

## Important files

- `backend/main.py` — HTTP + websocket
- `backend/app/scoring/engine.py` — heuristic classifier
- `backend/app/ml/dataset.py` — gold holds in/out of Postgres
- `backend/app/ml/train.py` / `numpy_clf.py` — real training
- `backend/app/ml/infer.py` — promote + live predict
- `backend/app/models.py` — SQLAlchemy tables
- `frontend/src/components/SignTrainer.tsx` — Train panel
- `frontend/src/components/SignCommunicator.tsx` — sentence + speak
- `frontend/src/components/SignRulebook.tsx` — left Signs overlay
- `frontend/src/lib/signSentence.ts` / `signVocab.ts` — sentence builder / vocab

## Stack (libraries vs hardcoded)

- **Libraries:** MediaPipe Hand/Pose landmarker (browser), FastAPI, Next.js, SQLAlchemy/Postgres, FastDTW, Web Speech API, Gemini coaching (optional)
- **Hardcoded until promote:** which finger shapes mean which sign (`engine.py`)
- **Not used for the 12-sign vocab:** MediaPipe GestureRecognizer canned poses (too few labels)
- Training is a **NumPy softmax** classifier because `sklearn` failed to load on this Windows machine (Application Control / DLL)

## Open / known issues

- History needed a working Postgres password; it is connected when `/sessions` returns 200 (empty list `[]` until someone practices).
- Do not guess or commit DB passwords.
- Promoting a 2-class model (fist vs thumbs_up) will force other still poses into those two labels when confident.
- Webcam video blobs are not in Postgres yet (paths only).
- Frontend History page previously 503 when DB was down.

## How to hand this to another Cursor login

1. Push or zip this git repo (include `AGENTS.md`).
2. Do **not** send `.env`. The other person copies `backend/.env.example` → `.env` and fills their own keys/password.
3. They open the folder in Cursor. New chats should read this `AGENTS.md`.
4. Optional: attach this file in the first message: “Read `AGENTS.md` and continue from there.”
