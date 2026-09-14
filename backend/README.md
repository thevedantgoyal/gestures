# Gesture Recognition Backend

Minimal FastAPI server for the AI Gesture Recognition Platform.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

- `GET /health` — health check (`{"status": "ok"}`)
- `GET /modes` — available modes (`["aviation", "sign_language"]`)
- `WS /ws/landmarks` — real-time landmark stream; aviation mode returns gesture scoring results
- `GET /reference/status` — which gestures use a recorded reference vs the defaults
- `POST /reference/record` — save a recorded reference from a captured landmark sequence
- `GET /reference/details` — recorded payloads plus the references currently in use
- Interactive docs: `http://localhost:8000/docs`

## Gesture references

Scoring compares live landmarks against a reference per gesture. References come
from one of two places:

1. **Recorded** — captured through the frontend's "Record reference" panel and
   persisted to `app/scoring/recorded_references.json`. Loaded on startup and
   applied immediately after recording (no restart needed).
2. **Default** — the hardcoded values in `app/scoring/reference_gestures.py`,
   used only for gestures that have not been recorded yet.

For `exit_pointing` the recording is averaged into a target elbow angle (the
first/last 0.5s are dropped so the transition into the pose isn't averaged in).
For `seatbelt_demo` the wrist-relative-to-waist path becomes the FastDTW
reference sequence.

Delete `app/scoring/recorded_references.json` to fall back to the defaults.

## Coaching text

Incorrect gesture attempts get a short AI tip via `app/services/coaching/`.

1. Copy `.env.example` → `.env` and set `GEMINI_API_KEY`
2. Provider is selected in `app/config.py` (`COACHING_PROVIDER = "gemini"`; switch to `"bedrock"` later)
3. On `correct=false`, `/ws/landmarks` sends the verdict first (`coaching_pending`), then a follow-up with `coaching_text`
4. If the provider fails, clients still get: `"Adjust your arm position and try again."`

## PostgreSQL session logging

Practice attempts are stored in Postgres via SQLAlchemy (`app/db.py`, `app/models.py`).

1. Ensure Postgres is running and a database exists (`gestures` or `gestures_db`)
2. Set `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` in `.env`
3. On startup the API runs `Base.metadata.create_all` (creates the `attempts` table)
4. `/ws/landmarks` inserts an `Attempt` whenever `gesture != "none"` (debounced ~2s while a pose is held)
5. Browse history:
   - `GET /sessions` — session list + summary
   - `GET /sessions/{session_id}` — full attempt timeline
   - Frontend: **History** link → `/history`
