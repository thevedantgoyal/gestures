# Existing Architecture Audit

Date: 28 September 2026  
Scope: study of the repository as it exists. No source code was changed to produce this document.  
Runtime facts the code cannot prove are marked **UNKNOWN — NEEDS RUNTIME VERIFICATION**.

This document is a handoff. It describes what the project actually does, where that differs from a continuous sign-language conversation product, and the order in which fixes should be considered. It is not a redesign.

---

# Existing Architecture

## Executive summary

The current system is a local AAC (augmentative and alternative communication) phrase builder with a closed vocabulary of 13 gestures. The browser reads the webcam with MediaPipe Hand Landmarker, sends named landmarks to a FastAPI WebSocket, and the backend classifies **one frame** with finger-extension rules. A short vote window plus a hold dwell commits **one word**. After the hand leaves, a template builder turns those words into a short English sentence. Ollama may tidy the grammar. The browser then speaks **the user’s own sentence** and clears the word list.

A NumPy softmax classifier can replace the still-pose rules after the user trains and clicks Promote. That model is still a static multi-class classifier over the same labels. Wave / Goodbye stays on motion rules.

The intended product — continuous signing of an open sentence such as “I want to go to the market tomorrow,” then an AI reply such as “Sure. What would you like to buy?” — is not implemented. The vocabulary has no signs for market, tomorrow, go, or a free pronoun sequence. The language model is instructed not to answer. The UI has no reply state.

## What the product actually is

1. Webcam sees the hands (MediaPipe landmarks; up to two hands; backend picks the primary hand).
2. The app recognizes one sign from a fixed list (Yes, Hello, Clear, and the rest below).
3. Signs stack into a sentence. Undo removes the last word. Fist clears the whole sentence.
4. The sentence is polished via Ollama when reachable, otherwise a template, then spoken.

There is also an older aviation practice mode (exit pointing, seatbelt). Sign-language Talk is the active product. Aviation uses body pose, not hands, and optional Gemini coaching. That coaching path is not the Talk conversation.

## Sign vocabulary (gesture key → spoken word)

| Key | Spoken | Role |
|---|---|---|
| `thumbs_up` | Yes | word |
| `thumbs_down` | No | word |
| `open_palm` | Hello | word |
| `fist` | Clear | control: wipe the sentence |
| `four` | Undo | control: drop the last word |
| `pointing` | Help | word |
| `peace_sign` | Thank you | word |
| `please` | Please | word |
| `you` | You | word |
| `want` | Want | word |
| `okay` | Okay | word |
| `i_love_you` | I love you | word |
| `wave` | Goodbye | word; motion only |

Source of truth: `backend/app/scoring/sign_meanings.py` and `frontend/src/lib/signVocab.ts`.

## Dual path: rules now, model after Promote

Until Promote, Talk uses hardcoded heuristics in `backend/app/scoring/engine.py` plus temporal stabilization in `backend/app/scoring/sign_stability.py`. No personal training is required for that path.

| Step | What happens | Talk changes? |
|---|---|---|
| Add samples | One confirmed hold is inserted into Postgres `training_samples` | No |
| Train now | Softmax classifier fits gold holds with at least 3 samples on at least 2 signs | No (shadow model only) |
| Promote | `ml_runtime.source = model`. Still signs use the model if confidence ≥ 0.55, else rules. Wave always uses rules | Yes |

A Yes-only dataset cannot train (need 2 classes). A heavily unbalanced dataset biases the promoted model toward the majority class.

## Actual pipeline (modules)

```
CAMERA
  navigator.mediaDevices.getUserMedia
  frontend/src/app/page.tsx  startCamera
        │
        ▼
VIDEO + requestAnimationFrame predict()
  frontend/src/app/page.tsx
        │
        ├─ mode aviation
        │    PoseLandmarker.detectForVideo
        │    pose_landmarker_lite.task (Google CDN)
        │
        └─ mode sign_language
             HandLandmarker.detectForVideo
             numHands: 2
             hand_landmarker.task (Google CDN)
        │
        ▼
extractAllHandLandmarks()
  named points: x, y, z, visibility, optional handedness
        │
        ▼
WebSocket  ws://localhost:8000/ws/landmarks
  landmarks_ws()  backend/main.py
        │
        ▼
select_primary_hand()          backend/app/scoring/engine.py
deque landmark buffer          maxlen 16
        │
        ├─ classify_sign_attempt()      one frame + wave motion test
        ├─ best_sign_pose()             heuristic scores
        ├─ SignStabilizer.update()      vote window + hysteresis
        └─ apply_model_to_sign_result() only if runtime source == "model"
        │
        ▼
WebSocket JSON
  gesture, correct, meaning, score, reason, stability, model, ...
        │
        ▼
page.tsx handleMessage
  dwell 550 ms → appendSignToken()
  fist         → resetSentenceBuilder()
  four         → undoLastSignToken()
        │
        ▼
idle 2200 ms after the hand leaves and the lock drops
  composeSignSentence()                  frontend/src/lib/signSentence.ts
  POST /ml/polish-sentence
  polish_sentence_with_ollama()          backend/app/services/sentence_polish.py
        │
        ▼
speakSentence()  window.speechSynthesis
SignCommunicator shows that sentence
word list is cleared

NOT PRESENT
  face landmarks
  continuous gloss decoder
  open-vocabulary translation
  conversation endpoint
  reply text in the UI
```

---

# Repository Map

There is no Docker, no CI, no Web Worker, no Zustand/Redux, and no ONNX/TFLite model file in the repo. Frontend state lives in one page component. Training weights live in Postgres as JSON, not as files committed to git.

## Frontend

Framework: Next.js 16.3.4, React 19, TypeScript, Tailwind 4.  
Dependencies that matter: `@mediapipe/tasks-vision`. No other ML library in the browser.

| File | Purpose | Who calls it | What it calls | Input | Output |
|---|---|---|---|---|---|
| `frontend/src/app/page.tsx` | Talk UI, camera loop, WebSocket client, sentence state, training buttons, aviation mode | Next.js route `/` | MediaPipe, `/ws/landmarks`, REST under `http://localhost:8000`, `signSentence`, `SignCommunicator`, `SignTrainer`, `SignRulebook` | Webcam frames, WS JSON, button clicks | Rendered Talk screen, spoken sentence |
| `frontend/src/app/layout.tsx` | Root layout | Next.js | — | children | HTML shell |
| `frontend/src/app/history/page.tsx` | Session list | route `/history` | `GET /sessions` | — | History list |
| `frontend/src/app/history/[sessionId]/page.tsx` | One session | route `/history/:id` | `GET /sessions/{id}`, recording video URL | session id | Attempt log and video |
| `frontend/src/components/SignCommunicator.tsx` | Sentence panel, Speak / Undo / Clear | `page.tsx` | callbacks only | tokens, sentence, last spoken, live gesture | Text the user said. No reply slot |
| `frontend/src/components/SignRulebook.tsx` | Left signs overlay | `page.tsx` | vocab | sign list | How to form each shape |
| `frontend/src/components/SignTrainer.tsx` | Add samples / Train / Promote | `page.tsx` | REST via callbacks in `page.tsx` | hold recording | Train status UI |
| `frontend/src/lib/signSentence.ts` | Token list and template sentences | `page.tsx` | `signVocab` | ordered meanings | One spoken string, max 10 tokens |
| `frontend/src/lib/signVocab.ts` | Labels, how-to steps, meaning types | sentence, rulebook, page | — | — | 13-sign catalog |
| `frontend/src/lib/signTrain.ts` | Types/guards for `/ml/stats` | `page.tsx` | — | JSON | Typed train stats |
| `frontend/src/lib/handOverlay.ts` | Canvas hand skeleton and motion trails | `page.tsx` | MediaPipe hand connections | landmarks, canvas | Pixels |

State management: React `useState` and `useRef` inside `page.tsx` only.  
Camera: `getUserMedia` in `startCamera`, drawn to a `<video>`, detected in `predict()`.  
Browser inference: MediaPipe Hand Landmarker and Pose Landmarker. The sign classifier does not run in the browser.  
Web Workers: none.  
API client: hardcoded `API_BASE = "http://localhost:8000"` and `WS_URL = "ws://localhost:8000/ws/landmarks"` in `page.tsx` and the history pages.  
WebSocket: one client in `page.tsx`, reconnect up to 5 times.

## Backend

Framework: FastAPI (`backend/main.py`), Uvicorn.  
Database: PostgreSQL via SQLAlchemy (`backend/app/db.py`, `backend/app/models.py`). Database name used by this project is `gestures`.  
Authentication: none.  
LLM: Ollama (OpenAI-compatible chat completions) for sentence polish. Gemini or Bedrock for aviation coaching only.

| File | Purpose | Who calls it | What it calls | Input | Output |
|---|---|---|---|---|---|
| `backend/main.py` | HTTP routes and `/ws/landmarks` | Frontend | scoring, ML, polish, sessions, coaching | WS landmark JSON, REST bodies | Score JSON, sentence JSON, history |
| `backend/app/config.py` | Env: coaching provider, Ollama, Gemini, Bedrock | polish, coaching | `backend/.env` via python-dotenv | env names | constants |
| `backend/app/db.py` | Engine, sessions, URL-encoded password | models, dataset, runtime | Postgres | `DB_*` | SQLAlchemy sessions |
| `backend/app/models.py` | Tables | db, services | — | rows | ORM |
| `backend/app/sessions.py` | Attempt and recording queries | `main.py` | `Attempt`, `SessionRecording` | session id | history JSON |
| `backend/app/persist.py` | One-time import of old JSON/JSONL into Postgres | startup migration path | files on disk | archived local files | DB rows |
| `backend/app/scoring/engine.py` | Heuristic sign and aviation classifiers, features, hand selection, wave motion | `main.py`, ML | NumPy, FastDTW for motion score | landmark dict, optional buffer | gesture label, scores, 15-d features |
| `backend/app/scoring/sign_stability.py` | Vote window, margin, hysteresis | `_evaluate_sign_language` | — | label, score, score map | locked / not locked |
| `backend/app/scoring/sign_meanings.py` | Key → spoken word | engine, infer, dataset | — | gesture key | spoken string |
| `backend/app/scoring/calibration.py` | Normalize and trim recorded holds; build references | dataset, reference record | engine features | landmark sequence | feature rows or wave sequence |
| `backend/app/scoring/reference_gestures.py` | Hardcoded and user-recorded pose/motion templates | scoring, `main.py` | Postgres `recorded_references` | gesture name | reference used by DTW / pose match |
| `backend/app/services/sentence_polish.py` | Grammar polish. Rejects meaning drift | `POST /ml/polish-sentence` | Ollama HTTP | token list + template | one sentence, source tag |
| `backend/app/services/coaching/*` | Aviation correction text (Gemini default, Bedrock stub) | aviation branch of the WebSocket | Gemini or Bedrock | deviation payload | coaching string |
| `backend/app/ml/dataset.py` | Gold holds in and out of Postgres | train routes, samples routes | `hand_pose_features`, `hand_wave_features` | landmark clip | `training_samples` row |
| `backend/app/ml/train.py` | Fit softmax on static holds | `POST /ml/train` | dataset, `numpy_clf`, registry | DB holds | new `trained_models` version, shadow only |
| `backend/app/ml/numpy_clf.py` | Multinomial logistic regression | train, infer | NumPy | `(n, 15)` features | class label and probabilities |
| `backend/app/ml/infer.py` | Load, promote, predict one frame | WebSocket after heuristic score | `hand_pose_features`, runtime | one landmark dict | optional label override |
| `backend/app/ml/runtime.py` | Singleton: heuristic vs promoted model | infer, `main.py` | `ml_runtime` table | source, version | runtime dict |
| `backend/app/ml/registry.py` | Save and load weight bundles | train, infer | `trained_models` | classifier + meta | version string |

### HTTP and WebSocket surface

| Method | Path | Role |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/modes` | aviation vs sign language |
| GET | `/sessions`, `/sessions/{id}` | history |
| POST | `/sessions/{id}/recording` | upload webcam video metadata/file |
| GET | `/sessions/{id}/recording`, `.../video` | playback |
| GET/POST/DELETE | `/reference/...` | optional 1-shot overlays |
| GET/POST | `/ml/runtime` | rules vs model |
| GET | `/ml/stats`, `/ml/models`, `/ml/samples` | train status and export |
| POST | `/ml/train` | fit shadow model |
| POST | `/ml/models/{version}/promote` | make that model live for still signs |
| POST | `/ml/samples` | insert one gold hold |
| POST | `/ml/polish-sentence` | rewrite the user’s tokens. Not a reply |
| WS | `/ws/landmarks` | live scoring |

CORS allowlist: `http://localhost:3000` and `http://127.0.0.1:3000` only.

## ML artifacts

| Item | Where it lives |
|---|---|
| Live hand detector | MediaPipe `hand_landmarker.task`, downloaded by the browser from Google storage |
| Live pose detector | MediaPipe `pose_landmarker_lite.task`, aviation only |
| Sign classifier | NumPy softmax. Format: JSON weights in `trained_models.weights`. Not a neural sequence model |
| Labels | Gesture keys in `SIGN_MEANINGS`, minus `wave` for the static trainer |
| Training rows | Postgres `training_samples` |
| Runtime switch | Postgres `ml_runtime` |

## Infrastructure

| Item | Status |
|---|---|
| Docker | absent |
| CI/CD | absent |
| Deployment config | absent. Local processes only |
| Frontend env | absent. URLs are string literals |
| Backend env | `backend/.env` (do not commit). Template: `backend/.env.example` |
| Tests | no project test suite in the repo |

## Database tables

| Table | Purpose |
|---|---|
| `training_samples` | Every Add-samples row. Filter by `name` (Yes, Hello, …) or `gesture` |
| `trained_models` | Each Train-now bundle (`version` like `v7`, weights JSON) |
| `ml_runtime` | Singleton: rules vs promoted model |
| `recorded_references` | Optional recorded overlays. Empty means hardcoded defaults |
| `attempts` | Live practice log for History |
| `session_recordings` | Session metadata. Video files stay on disk under `backend/recordings/` |

Old file caches (`samples.jsonl`, `runtime.json`, and similar) were import-once archives. Postgres is the source of truth for new data.

---

# Actual Data Flow

One Talk interaction, from camera to speech, using the functions that exist.

1. User allows the camera. `startCamera` in `frontend/src/app/page.tsx` calls `navigator.mediaDevices.getUserMedia`. The `<video>` element plays the stream.
2. `predict()` runs on `requestAnimationFrame`. When `mode === "sign_language"`, `HandLandmarker.detectForVideo(video, performance.now())` returns up to two hands, 21 landmarks each, normalized `x`, `y`, `z`.
3. `extractHandLandmarks` / `extractAllHandLandmarks` copy those into a named object (`wrist`, `thumb_tip`, `index_tip`, …) plus optional MediaPipe handedness. `z` is included here.
4. Every 120 ms (`WS_SEND_INTERVAL_MS`), if the socket is open, the page sends:

   `{ mode, session_id, landmarks, hands, timestamp }`

   to `ws://localhost:8000/ws/landmarks`.
5. `landmarks_ws` in `backend/main.py` parses JSON. If `hands` is a non-empty list, `select_primary_hand` replaces `landmarks` with the chosen hand. The frame is appended to a `deque` of maxlen 16 when it contains `wrist` and `index_tip`.
6. `_evaluate_sign_language` calls `classify_gesture_attempt(..., mode="sign_language")`, which calls `classify_sign_attempt`. That function inspects finger-extension flags on **this frame** and wrist motion on the buffer. It returns one of the 13 keys or `none`.
7. `best_sign_pose` scores every still sign. Wave is handled separately with `buffer_looks_like_wave` and `score_motion`.
8. `SignStabilizer.update` (window 7, 3 votes to lock, score at least 0.55, top-two margin 0.07) decides whether the label is stable. `correct: true` is returned only when the label is locked.
9. If `ml_runtime.source == "model"`, `apply_model_to_sign_result` may replace the still-sign label with `predict_sign` when probability ≥ 0.55. Wave is never replaced. If the runtime source is `heuristic` (the default), the model field may be filled for display but the spoken label stays the heuristic.
10. The WebSocket sends JSON back. `handleMessage` in `page.tsx` stores live feedback.
11. A word is committed only when `correct` is true, `meaning` is a string, the same gesture has been pending for at least 550 ms (`SPEAK_DWELL_MS`), and it is not the gesture already committed for this hold. `appendSignToken` pushes the spoken word. `fist` / Clear calls `resetSentenceBuilder`. `four` / Undo calls `undoLastSignToken`. The same sign cannot be committed twice until the label changes or the hand leaves the frame (`releasedAfterCommitRef`).
12. While `gestureFeedback.correct` is true, the auto-speak effect does not start. When the hand leaves, the detect loop sets feedback to `none`. After 2200 ms (`SENTENCE_SPEAK_IDLE_MS`), `speakWithGrammarPolish` runs, and only if voice was unlocked.
13. `composeSignSentence(tokens)` builds the fallback string. The page POSTs `{ tokens, fallback }` to `/ml/polish-sentence`.
14. `polish_sentence_with_ollama` either returns a curated template, an Ollama rewrite that still contains every signed stem, or the template again if Ollama is down or the rewrite is rejected.
15. `speakSentence` uses `SpeechSynthesisUtterance` (en-US). `speakComposedSentence` then clears `sentenceTokens`. `SignCommunicator` keeps the line on screen via `lastSpoken`.

There is no step that stores an assistant reply, and no step that speaks a second sentence.

### What a real sign does

Signing the pose for `want` commits the single word `Want`. After the pause, the template is `I want that.` Ollama may polish that sentence. The app speaks `I want that.` (or the polished form). It does not produce “I want to go to the market tomorrow,” because those words are not labels.

---

# ML Pipeline

## Problem class

**A. Static gesture classification**, plus one hand-crafted motion rule for wave.

It is not:

- B. Isolated sign recognition over a learned time window
- C. A temporal sign model (no sequence network)
- D. Continuous sign-language translation

Calling the softmax model a translation model would be wrong. It maps one still hand shape to one of a few trained class names.

## Model

| Field | Value |
|---|---|
| Algorithm | `SoftmaxClassifier` in `backend/app/ml/numpy_clf.py` (multinomial logistic regression, L2, 400 epochs, learning rate 0.25) |
| Format | JSON: `classes` list and `weights` matrix, stored in Postgres `trained_models` |
| Loader | `registry.load_bundle` → `SoftmaxClassifier.from_dict` |
| Default Talk path | Heuristic. Model is live only after Promote sets `source` to `model` |
| Confidence gate | `MODEL_MATCH_THRESHOLD = 0.55` in `runtime.py` |
| Wave | Always heuristic, even when a model is promoted |

`infer.py`’s module docstring still says “sklearn”. The implementation is NumPy. sklearn is not in `backend/requirements.txt`.

## Training versus production

| | Training | Production Talk |
|---|---|---|
| Who | `train_sign_classifier` | `classify_sign_attempt`, then maybe `predict_sign` |
| Samples | Each gold hold, trimmed, evenly sampled to at most 10 frames. Each frame is its own row with the hold’s label (`collect_static_examples`) | The current WebSocket frame only |
| Features | `hand_pose_features` → length 15 | same function → length 15 |
| Time | Frames are i.i.d. rows. The model never sees a sequence | Label votes over 7 classifications. Features are not stacked |
| Wave | Excluded from `STATIC_GESTURES` | Motion rules + optional FastDTW against a wave reference |
| When it affects speech | Never, until Promote | After Promote, still signs only |

Shape matches. The learning problem does not: the trainer pretends frames inside a hold are independent, and live inference never feeds a window of features to the model.

## Known model risks already implied by the code

- Promoting a 2-class model forces other still poses into those two labels whenever confidence ≥ 0.55.
- `apply_model_to_sign_result` runs **after** `SignStabilizer`. A confident model label replaces the locked heuristic label without a new vote.
- Class set and whether any version is promoted: **UNKNOWN — NEEDS RUNTIME VERIFICATION** (`GET /ml/runtime`, `GET /ml/models`).

---

# Model Input/Output

## What the camera produces

MediaPipe hand: 21 landmarks. Each landmark sent by the frontend has `x`, `y`, `z`, and `visibility` (visibility defaulted to 1 when MediaPipe omits it).

21 × 3 coordinates = 63 coordinate values, plus visibility and an optional handedness string. They are sent as a **named dict**, not a flat tensor.

## What the classifier consumes

`hand_pose_features` in `engine.py`:

- 5 fingertips (`thumb_tip`, `index_tip`, `middle_tip`, `ring_tip`, `pinky_tip`)
- For each tip: `(tip.x - wrist.x) / scale` and `(tip.y - wrist.y) / scale`
- Then 5 extension ratios: distance(wrist, tip) / distance(wrist, mcp)
- `scale` = 2D distance from wrist to `middle_mcp`

5 × 2 + 5 = **15 features**.

`FEATURE_DIM = 15` in `dataset.py`. Live code builds `np.asarray([vector], dtype=float)`, shape **`(1, 15)`**. Weights are shape **`(16, n_classes)`** because of a bias column. `n_classes` is whatever was in that training run (minimum 2, maximum 12 still signs).

`predict_sign` returns one label string and one probability for that label. It does not return a sequence.

## Wave features (not the softmax input)

`hand_wave_features`: 4 numbers per frame (`wrist.x`, `wrist.y`, index tip offset x, index tip offset y). Stored sequences are downsampled to at most 10 frames. The live buffer holds at most 16 frames.

## Audit checklist

| Check | Result |
|---|---|
| Expected model input | `(1, 15)` float |
| Actual application input to the model | `(1, 15)` from the same function, when a bundle is loaded and `feature_dim` matches |
| Shape mismatch | **No** |
| Semantic mismatch | **Yes.** Training rows are many frames of one hold. Inference is one frame. `z` is dropped. Handedness is stored on the sample and is not a feature, so left and right hands mirror the signed x offsets |
| x/y/z order | x then y for tips. z unused. Image-normalized MediaPipe coordinates, not camera-millimetre coordinates |
| Missing hand | `hand_pose_features` returns `None`. Predict returns no label. Heuristic returns `none` |
| dtype | Python floats, then NumPy `float` (float64) |
| Batch | Implicit batch of 1 |

**MODEL INPUT MISMATCH:** none on vector length. The mismatch that matters is that this vector cannot represent a signed sentence.

---

# Temporal Pipeline

Still signs: **single-frame inference**, then a vote over labels.

Wave: **short buffer test**, not a learned sequence model.

| Parameter | Value | Where |
|---|---|---|
| Window (votes) | 7 classifications | `SignStabilizer` default |
| Lock / unlock votes | 3 | `SignStabilizer` |
| Min score | 0.55 | stabilizer and `SIGN_RECOGNITION_THRESHOLD` |
| Top-2 margin | 0.07 | stabilizer |
| Landmark buffer | 16 frames | `BUFFER_MAX_FRAMES` in `main.py` |
| Stride | none (every accepted WS frame is appended) | `landmarks_ws` |
| Send interval | 120 ms | `WS_SEND_INTERVAL_MS` |
| Approx. buffer time | about 1.9 s if every send is buffered | 16 × 120 ms |
| Prediction frequency | once per WebSocket message | `landmarks_ws` |
| Commit dwell | 550 ms of the same `correct` gesture | `SPEAK_DWELL_MS` |
| Sentence idle | 2200 ms after the lock drops | `SENTENCE_SPEAK_IDLE_MS` |
| Wave min samples | 5 wrist-x values | `WAVE_MIN_SAMPLES` |
| Wave min amplitude | 0.085 (normalized x) | `WAVE_MIN_AMPLITUDE` |
| Wave reversals | at least 2 | `WAVE_MIN_REVERSALS` |
| Motion energy gate | mean wrist step ≥ 0.014 marks the hand as moving | `HAND_MOTION_ACTIVE_THRESHOLD` |
| Sign boundary model | none | commit is dwell + “hand left or label changed” |
| Smoothing | majority vote, hysteresis, top-2 margin | `sign_stability.py` |

This is enough to lock a held finger shape and to reject a still palm as Goodbye. It is not enough for continuous signing. There is no boundary detector, no CTC/alignment, no gloss lattice, and no model that reads a second of motion as one sign except the wave wag test.

---

# Sentence Pipeline

## Implementation

| | |
|---|---|
| File | `frontend/src/lib/signSentence.ts` |
| Function | `composeSignSentence` |
| Model | none |
| API | optional polish after the template exists |
| Input | ordered spoken meanings, for example `["Please", "Want", "Help"]` |
| Output | one string, for example `Please, I want help.` |

`clauseFor` rewrites a few single words (`Help` → `I need help`, `Want` → `I want that`). `PAIR_CLAUSES` and `TRIPLE_CLAUSES` cover a small set of everyday combinations (`Hello|You` → `Hello, how are you`). Anything else is the clauses joined with `. `. Clear and Undo produce empty clauses and are not spoken as words. Maximum 10 tokens.

## What it cannot emit

There is no gloss `I`, `MARKET`, `TOMORROW`, or `GO`. The system cannot produce “I want to go to the market tomorrow.” The nearest live output for the want-pose is “I want that.”

## Polish layer

| | |
|---|---|
| File | `backend/app/services/sentence_polish.py` |
| Function | `polish_sentence_with_ollama` |
| Route | `POST /ml/polish-sentence` in `backend/main.py` |
| Request | `{ "tokens": string[], "fallback": string }` |
| Response | `{ status, sentence, source, detail, tokens }` |
| `source` | `template`, `ollama`, or `fallback` |
| Prompt rule | One natural sentence. Same meaning. Do not invent people, places, or actions. Do not drop signed ideas |
| Rejection | `_reply_keeps_meaning` discards a reply that drops stems or grows far past the template |

If the template is already in the curated key set, Ollama is not called.

**Language reconstruction beyond these templates: partial, closed-vocabulary only.**  
**Open sign-language translation: missing.**

---

# Conversation Pipeline

## Does a conversation system exist?

**No.**

Trace of the only LLM call on the Talk path:

```
composed sentence
  → POST /ml/polish-sentence
  → polish_sentence_with_ollama
  → Ollama chat completions (or immediate template)
  → same meaning, user's sentence
  → speakSentence (user's sentence)
  → UI lastSpoken
```

There is no endpoint that takes the sentence and returns an assistant answer. There is no chat history. The polish system prompt forbids the behavior you would want from a partner (“do not invent … new actions”).

Gemini (`get_coaching_text`) runs only on the aviation branch when exit-pointing or seatbelt is incorrect. It does not see sign sentences.

| Check | Result |
|---|---|
| Endpoint | `POST /ml/polish-sentence` only |
| Schema | tokens + fallback in, sentence + source out |
| API key | Ollama: none in code. Gemini key is `GEMINI_API_KEY` and is not used for Talk polish |
| Env | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS` |
| CORS | localhost:3000 only, so a different origin cannot call the API |
| Auth | none |
| Timeout | default 45 s around the Ollama HTTP call |
| Streaming | none. Full body, then one string |
| Error handling | HTTP errors and meaning-drift return the template with HTTP 200. The UI treats that as success and speaks it |
| Why no AI answer appears | Nothing requests one, nothing stores one, `SignCommunicator` has no reply prop |

---

# Frontend State Flow

All Talk state is inside `frontend/src/app/page.tsx`.

| State | Role |
|---|---|
| `sentenceTokens` / `sentenceTokensRef` | Words committed so far |
| `lastSpokenSentence` | Line kept on screen after tokens are cleared |
| `gestureFeedback` | Latest WS gesture. `correct` gates auto-speak |
| `pendingSpeakRef` | Start time of the current gesture dwell |
| `lastCommittedSignRef` | Prevents committing the same hold twice |
| `releasedAfterCommitRef` | Set when gesture becomes `none`, so the next hold can commit |
| `speaking`, `polishing` | Block undo and a second polish while busy |
| `voiceUnlocked` | Auto-speak and Speak stay off until the user taps to allow voice |
| `wsRef` | Socket. Reconnect max 5 |
| `handLandmarkerRef` | MediaPipe instance |

Effects that matter:

- WebSocket connect depends on `isStreaming`. Malformed messages are swallowed by an empty `catch`. `onerror` is empty; `onclose` reconnects.
- Auto-speak depends on `sentenceTokens`, mode, streaming, voice, speaking, polishing, and `gestureFeedback.correct`. It returns immediately while the current sign is still locked, so the sentence waits until the hand leaves.
- Mode change clears the sentence, feedback, and canvas.
- After speech starts, tokens are cleared at once. The visible line is `sentence || lastSpoken` in `SignCommunicator`.

Race notes grounded in code:

- Commit and polish use refs for the token list, so a stale React render is less likely to drop a word than the state value alone would be.
- `speakWithGrammarPolish` bails out if `polishing` or `speaking` is already true in that effect closure. The effect re-runs when those flags change, which is the intended guard.
- A dead Ollama does not surface as an error banner. `console.warn` and the template path still speak.
- Predictions with `correct: false` update the badge but do not append a word. That is deliberate, not a dropped commit.
- There is no subscription bug hiding an assistant message, because no assistant message is ever written to state.

---

# Backend Flow

Startup (`lifespan` in `main.py`): `init_db()` (failure is logged and the API still comes up), `load_recorded_references()`, `load_runtime()`, `load_live_model()`.

Each sign-language WebSocket frame:

1. Parse JSON or return `invalid_json`.
2. Select primary hand.
3. Reset buffer and stabilizer when `mode` changes.
4. If the payload is not a hand frame, send `waiting_for_hand_landmarks`.
5. `_evaluate_sign_language`.
6. `apply_model_to_sign_result`.
7. Attach `ws_recognizer_fields` (`recognizer`, `model` stub).
8. Best-effort `log_attempt` into `attempts`, debounced by 2 s. Database errors are printed and do not stop the socket.
9. `send_json`.

`_evaluate_sign_language` prints framing and scores on every frame. That is noisy, not a functional break.

Polish is a separate request, not part of the socket. It does not read `attempts` or prior sentences.

---

# Environment Configuration

Variable **names** only. Do not copy secret values into this file or into git.

From `backend/.env.example` and `backend/app/config.py`:

| Name | Used for | Code default if unset |
|---|---|---|
| `GEMINI_API_KEY` | Aviation coaching | empty |
| `GEMINI_MODEL` | Aviation coaching | `gemini-3.6-flash` |
| `COACHING_PROVIDER` | `gemini` or `bedrock` | `gemini` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID` | Bedrock coaching stub | Bedrock model id has a Haiku default |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | Postgres | example file shows host `localhost`, port `5432`, database `gestures`, user `postgres` |
| `OLLAMA_BASE_URL` | Sentence polish | code default `http://127.0.0.1:11434/v1/` |
| `OLLAMA_MODEL` | Sentence polish | `llama3.1:latest` |
| `OLLAMA_TIMEOUT_SECONDS` | Sentence polish | `45` |

Conflicts to verify on a machine, not assumed here:

- `.env.example` sets `OLLAMA_BASE_URL` to `http://172.16.200.30:11434/v1/`. The code default is localhost. Which value `backend/.env` actually has is **UNKNOWN — NEEDS RUNTIME VERIFICATION**.
- `.env.example` uses `DB_PORT=5432`. Project notes for this workstation say Postgres is on port **5433**. A wrong port makes History return 503. Talk scoring still runs. New samples fail until the database accepts writes.
- Frontend always calls `localhost:8000` over HTTP and `ws://`. There is no HTTPS URL in the client. Opening the UI from another host will fail CORS and the hardcoded host.
- MediaPipe wasm and `.task` files load from public CDNs (`cdn.jsdelivr.net`, `storage.googleapis.com`). Offline use fails model load. **UNKNOWN — NEEDS RUNTIME VERIFICATION** whether those URLs are reachable from the browser in use.
- No model path env var exists, because weights are rows in Postgres.

`backend/.env` must not be committed.

---

# Fault Analysis

Status is against the **intended** product (open signed sentence, then an AI reply), not against “does the AAC phrase builder run.”

| Stage | Expected | Actual | Status | Evidence |
|---|---|---|---|---|
| Camera | frames | `getUserMedia` → `<video>` → `detectForVideo` | PASS in code | `page.tsx` `startCamera`, `predict` |
| MediaPipe hands | landmarks | Hand Landmarker, `numHands: 2`, 21 named points | PASS in code | `page.tsx` HandLandmarker options |
| Body for signs | body features in the sign path | Pose Landmarker only when `mode === "aviation"` | FAIL for signs | `predict` mode branch |
| Face | face landmarks | no Face Landmarker | FAIL | repo search |
| Preprocessing | features that can encode a signed phrase | 15-d still-pose vector; `z` dropped | FAIL for translation | `hand_pose_features` |
| Model loading | a temporal translation model | optional softmax JSON; default runtime source `heuristic` | FAIL | `runtime.py`, `infer.load_live_model` |
| Inference | a gloss sequence | one label per frame from rules, or one static class | FAIL | `classify_sign_attempt`, `predict_sign` |
| Temporal logic | continuous signing | 16-frame buffer for wave; 7-vote lock for labels | PARTIAL | `BUFFER_MAX_FRAMES`, `SignStabilizer` |
| Sign commit | a stable gloss | 550 ms dwell, one closed-vocab word | PASS for this vocab | `handleMessage`, `appendSignToken` |
| Sentence | free grammatical sentence | templates over 11 content words | FAIL for an open sentence | `composeSignSentence` |
| API | a conversation request | polish endpoint rewrites the user line | FAIL | `ml_polish_sentence` |
| LLM | an answer | prompt forbids a new action; fallback is the user line | FAIL | `_SYSTEM`, `_reply_keeps_meaning` |
| UI | answer displayed | user’s sentence and last spoken line only | FAIL | `SignCommunicator` props |

Camera permission, Postgres reachability, whether a model is promoted, and whether Ollama is up are **UNKNOWN — NEEDS RUNTIME VERIFICATION**.

Places that fail quietly:

- WebSocket `onmessage` `catch` ignores bad payloads with no log (`page.tsx`).
- `ws.onerror` is empty.
- Ollama failure returns HTTP 200 and `source: "fallback"` (`sentence_polish.py`).
- `init_db` failure does not stop the API (`main.py` lifespan).
- Attempt logging failure is printed and ignored (`_persist_attempt`).
- Frontend `fetch` for reference status uses `.catch(() => {})`.

---

# Root Causes

## Primary

The repository was built as a 13-label AAC stack: detect a finger shape, commit a word, template a short sentence, optionally polish grammar, speak that sentence. Continuous sign-language translation and a conversational reply were never implemented.

Severity: **CRITICAL**  
Files: `backend/app/scoring/sign_meanings.py`, `frontend/src/lib/signSentence.ts`, `backend/app/services/sentence_polish.py`, `frontend/src/components/SignCommunicator.tsx`  
What is wrong: the pipeline ends by speaking the user.  
Why the observed behavior happens: there is no function whose job is to answer.  
What should happen instead, for the intended product: an open vocabulary or a real sign-to-text model, then a separate reply turn with its own UI state.

## Secondary

1. **CRITICAL — no reply path.** `polish_sentence_with_ollama` returns the user’s sentence. The communicator has no reply field. An answer cannot appear.
2. **HIGH — closed vocabulary.** `want` is one static pose meaning the word Want. There is no market, tomorrow, or go.
3. **HIGH — the learned model is optional and static.** Default `source` is `heuristic`. After Promote the input is still one 15-d vector.
4. **MEDIUM — model override skips stability.** `apply_model_to_sign_result` can replace a locked heuristic label. A 2-class model can absorb every other still pose above 0.55.
5. **MEDIUM — handedness and `z`.** Features use image x/y relative to the wrist only. Left and right hands mirror x. Handedness is saved on the training row and unused.
6. **MEDIUM — silent polish fallback.** A down Ollama still looks like a successful speak. It still does not converse.
7. **LOW — `page.tsx` is one coupled page** (camera, aviation, Talk, train, history upload). Socket parse errors are swallowed. Per-frame `print` in `_evaluate_sign_language` hides useful logs.

## By area

| Area | Problem | Severity |
|---|---|---|
| Architecture | Talk, aviation, training, and speech share `page.tsx` and one WebSocket. The sentence step cannot grow into dialogue without a new response type | HIGH |
| ML | Static 15-d softmax plus rules. Not a temporal recognizer | CRITICAL for the intended product |
| Frontend | No reply state. Speech clears tokens on purpose | CRITICAL for conversation; correct for current AAC |
| Backend | Polish route is the only LLM on Talk, and it rejects new meaning | CRITICAL |
| Configuration | Hardcoded localhost, example Ollama host may not match the code default, DB port notes disagree (5432 vs 5433) | MEDIUM |
| Data | Labels are the 13 shapes. No sentence-level or gloss-sequence dataset | CRITICAL for translation |

---

# Gap Analysis

```
INTENDED

Camera
  → hand / body / face landmarks
  → feature sequence
  → temporal sign recognition
  → gloss sequence
  → language reconstruction
  → grammatical sentence
  → conversational model
  → reply text and optional speech

CURRENT

Camera                          page.tsx getUserMedia
  → hand landmarks only         HandLandmarker.detectForVideo
  → 15-d still features         hand_pose_features
  → finger rules or softmax     classify_sign_attempt / predict_sign
  → one word after 550 ms       appendSignToken
  → template                    composeSignSentence
  → optional grammar polish     polish_sentence_with_ollama
  ████████████████████████
  MISSING: assistant reply
  ████████████████████████
  → browser speaks the user     speakSentence
```

| Intended | Current |
|---|---|
| Temporal sign model | Static classifier + wave rules |
| Gloss sequence from fluent signing | One committed word per hold, from 13 labels |
| Language translation | `composeSignSentence` templates |
| Conversation | Polish API exists and is wired to the user line only |
| Face and body in the sign path | Hands only. Body is aviation |
| Reply on screen | `SignCommunicator` shows tokens, sentence, last spoken |

---

# Recommended Fix Order

Do not start with a new network. The running AAC path already classifies its 13 shapes without training.

## Priority 0 — things that stop the app

| File | Change | Reason | Depends on | Risk |
|---|---|---|---|---|
| `backend/.env` | Confirm `DB_PORT` matches the local Postgres (notes say 5433; example says 5432) | History 503 and failed sample inserts | local Postgres | Wrong port only affects DB features |
| processes | Backend on port 8000, frontend on port 3000, camera permission granted | Hardcoded URLs | — | Low |
| Ollama | Only required for grammar polish | Talk speaks templates without it | `OLLAMA_BASE_URL` | None for recognition |

## Priority 1 — sign recognition

Only if the current 13 shapes miss in practice. Rules already run with no trained model. A new softmax does not add words that are not labels. If Promote is used, collect similar counts per sign and do not promote a 2-class model onto a 12-sign UI.

## Priority 2 — sentence formation

Templates cannot become “I want to go to the market tomorrow” until those words exist as signs. Each new word needs a gesture key, a meaning, a rule or a training class, and a clause. That is still isolated classification, not continuous translation.

## Priority 3 — conversation

| File | Change | Reason | Depends on | Risk |
|---|---|---|---|---|
| New reply function next to `sentence_polish.py` | Separate prompt that answers the sentence. Do not reuse `_SYSTEM` | Current prompt forbids replies | Ollama or another chat API | Low if polish stays as the user line |
| `backend/main.py` | New route, for example `POST /ml/reply`, with tokens or the final sentence plus a short turn list | UI needs a contract | reply function | Timeout should not block the socket |
| `page.tsx`, `SignCommunicator.tsx` | State for turns. Show You and Reply. Speak the reply. Fist clears turns. Undo only edits words before send | This is the missing UI | reply route | Do not clear the user line before it is visible |

## Priority 4 — structure and model class

Splitting `page.tsx` reduces coupling. Do not put a sequence model inside `SoftmaxClassifier`. That class has no time dimension. A temporal recognizer is a new subsystem (new features, new labels, new inference buffer), not a patch on Train now.

## Priority 5 — hardening

- Move `API_BASE` / `WS_URL` to configuration.
- Log WebSocket parse failures.
- Remove or gate per-frame prints in `_evaluate_sign_language`.
- Add tests for `composeSignSentence` and for the reply contract (user line unchanged, reply is a different field).
- Keep `backend/.env` out of git.

---

# Known Limitations

- Vocabulary is the 13 rows above. Controls Clear and Undo are not spoken words.
- Goodbye requires a real wag. A still palm is Hello. A moving palm without a wag is not committed.
- Both hands may be visible. Talk scores the selected primary hand, not a two-hand sign.
- Auto-speak waits until voice is unlocked and until the hand is no longer locked, then 2.2 s.
- After speech, the token list is cleared. Repeat uses the last spoken string.
- Webcam video blobs are not in Postgres. `session_recordings` stores a path under `backend/recordings/`.
- Training is NumPy because sklearn DLL loading failed on this Windows machine. That constraint is environmental, not a reason to call the softmax a sequence model.
- Promoting an unbalanced model biases Talk for still signs. Wave stays on rules.
- Offline browsers cannot download MediaPipe wasm or the `.task` files.
- No authentication on the API. It is a localhost tool.

---

# Questions / Unknowns

UNKNOWN — NEEDS RUNTIME VERIFICATION:

- Whether `backend/.env` points Ollama at localhost or at `172.16.200.30`.
- Whether Postgres is listening on 5432 or 5433 on this machine, and whether `/sessions` returns 200.
- Whether `ml_runtime.source` is `heuristic` or `model`, and which `trained_models.version` is live.
- How many holds exist per sign in `training_samples`.
- Whether the browser can load the MediaPipe CDN files.
- Whether the user has tapped to unlock speech. Without that, the sentence can still be composed on screen and will not be spoken.
- Live accuracy of each finger rule on this camera and this user. The code paths exist; recognition quality is not proven by reading them.

---

# If you sign in front of the camera right now

1. The page captures the webcam and runs MediaPipe Hand Landmarker on each animation frame.
2. About 8 times a second it sends the hand points to the backend.
3. The backend checks which fingers are extended and, for Goodbye, whether the wrist is wagging.
4. It will not emit a word until the same label wins a short vote and you have held it for about half a second.
5. That word is one entry from the table in this document (Yes, Hello, Help, …).
6. You can stack up to 10 words. Four fingers removes the last one. A fist deletes all of them.
7. When you drop your hand and wait about two seconds, the app turns those words into a short sentence with fixed templates, optionally cleans the grammar, speaks **that sentence**, and clears the list.
8. It does not answer you.

Signing an English sentence that is not in the 13-word list does nothing more than whichever single shape the rules recognize, one word at a time.
