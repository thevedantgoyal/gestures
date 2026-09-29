# ISL Implementation Plan

Date: 28 September 2026  
Companion to: `docs/EXISTING_ARCHITECTURE_AUDIT.md`  
Scope: research and planning only. No source code was changed to produce this document.

This plan is repository-specific. It preserves the existing AAC Talk path and adds conversation first, then temporal ISL recognition, then continuous translation.

---

## Sign-language target (important)

The repository does **not** currently name Indian Sign Language as its target.

- `AGENTS.md` and the UI describe a local AAC phrase builder.
- The 13 shapes (thumbs up, peace, ILY, open palm, etc.) are generic / AAC-style, with some ASL-adjacent shapes, not an ISL lexicon.
- Choosing ISL as the product language is a **new goal**, not what the code implements today.

Phase 1 must not be labeled “ISL translation” in the UI. Label AAC vs ISL modes clearly when temporal models arrive.

---

## 1. Current architecture

```
Webcam
  → MediaPipe HandLandmarker (Talk) / PoseLandmarker (aviation)
  → WS /ws/landmarks
  → classify_sign_attempt (1 frame) + SignStabilizer
  → optional SoftmaxClassifier (15-d, after Promote)
  → appendSignToken (13 words)
  → composeSignSentence (templates)
  → POST /ml/polish-sentence (Ollama rewrite, no reply)
  → speak user's sentence
```

**Keep:** camera loop, MediaPipe, WebSocket scoring, stabilizer, Train/Promote personalization, Postgres samples, template sentence fallback, Ollama wiring, Speak UI, Clear/Undo, aviation coaching.

---

## 2. Target architecture

```
Webcam
  → MediaPipe Holistic-style features
       (2 hands + pose + optional face subset)
  → floating temporal buffer (landmarks, not pixels)
  → temporal recognizer / translator
       (isolated CTC/Transformer early; continuous Pose→Text later)
  → gloss sequence and/or English draft
  → language reconstruction (dataset-backed or LLM gloss→EN)
  → POST /ml/reply  (conversation; not polish)
  → assistant reply in UI + optional speech
```

Expand `ml_runtime.source` conceptually to:

`heuristic | model | temporal | translate`

Keep AAC (`heuristic` / `model`) as a fallback mode while `temporal` / `translate` are added. Do not delete SoftmaxClassifier or aviation.

---

## 3. Dataset recommendation

| Dataset | URL | Type | Annotations | Training fit | License / access | Fit for this repo |
|---|---|---|---|---|---|---|
| **INCLUDE** | [GitHub AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE), [HF mirror](https://huggingface.co/datasets/manojkumarcs/INCLUDE_Dataset), [paper](https://doi.org/10.1145/3394171.3413528) | **Isolated** word signs | 4,287 videos, 263 words, INCLUDE-50 subset; MediaPipe Hands + BlazePose pipeline in their code | Yes for word-level temporal SLR | Public research distribution via category zips + train/test CSVs; confirm terms on download | **Best Phase 2** bootstrap. Same landmark idea as this app. Not continuous sentences |
| **CISLR** | [HF Exploration-Lab/CISLR](https://huggingface.co/datasets/Exploration-Lab/CISLR), [ACL paper](https://aclanthology.org/2022.emnlp-main.707/) | **Isolated** / word-level (~4.7k words, ~7k videos) | English word labels | Yes for large vocab recognition | **CC BY-NC 4.0** (no commercial). HF gated (accept conditions) | Good Phase 2 vocab expansion. Not continuous |
| **ISLTranslate** | [GitHub Exploration-Lab/ISLTranslate](https://github.com/Exploration-Lab/ISLTranslate), [ACL Findings 2023](https://aclanthology.org/2023.findings-acl.665/), [HTML](https://arxiv.org/html/2307.05440) | **Continuous** sentence/phrase | ~31k ISL–English pairs; **gloss-free**; ships `ISLTranslate.csv` + videos + **pre-extracted MediaPipe Holistic poses** | Yes for Pose→English SLT | Research; check download terms on their link | **Primary Phase 3** dataset. Aligns with landmark WS pipeline |
| **iSign** | [Website](https://exploration-lab.github.io/iSign/), [HF Exploration-Lab/iSign](https://huggingface.co/datasets/Exploration-Lab/iSign), [arXiv](https://arxiv.org/abs/2407.05404) | Large continuous ISL benchmark | Video–sentence pairs + poses; multiple NLP-style tasks | Yes, but huge (~150GB video / ~50GB poses in community mirrors) | **Research free, not commercial** | Phase 3–4 scale-up after ISLTranslate |
| **ISL-CSLTR** | [Mendeley](https://data.mendeley.com/datasets/kcmpdxky7p/1), [Kaggle copy](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset) | Continuous, small (700 videos, 100 sentences, 7 signers) | Spoken sentences; signer/time boundaries; secondary sources claim frame gloss + start/end | Prototype continuous SLR with boundaries | Kaggle lists **CC BY-SA 4.0**; verify on Mendeley | Useful **pilot** for CTC + boundaries. Too small alone for production vocab |
| **PoseStich-SLT / BPCC-ISL** | [GitHub Exploration-Lab/PoseStich-SLT](https://github.com/Exploration-Lab/PoseStich-SLT) | Synthetic continuous pose–text | Stitched poses from iSign/CISLR etc. | Augmentation / scale | Research; inherits source licenses | Phase 4 augmentation only |
| Community realtime demo | [djdhairya/…Translator](https://github.com/djdhairya/Real-Time-Indian-Sign-Language-ISL-to-English-Translator) | Demo stack on iSign poses | Pose→Transformer→English | Reference architecture, not a dataset | Depends on iSign license | Study inference patterns; evaluate yourself |

**“RTISLT”:** No authoritative public dataset under that exact name was found in research for this plan. Treat community “real-time ISL” demos as **code references**, not a named corpus.

### Recommended stack by phase

| Phase | Data |
|---|---|
| Phase 1 | Existing 13 AAC signs + Postgres `training_samples` (no new ISL download) |
| Phase 2 | **INCLUDE-50** first, then INCLUDE / CISLR for isolated temporal word recognition |
| Phase 3 | **ISLTranslate** (MediaPipe Holistic poses + English) as main continuous set; ISL-CSLTR as small CTC pilot if gloss boundaries are needed |
| Phase 4 | iSign (+ optional PoseStich) once storage/GPU allow |

### A. Sign language data (answers)

**Use:** INCLUDE (isolated) + ISLTranslate (continuous), with CISLR for vocab growth and ISL-CSLTR as a small continuous pilot.

**Why:** INCLUDE’s official code already extracts MediaPipe Hands + pose keypoints — closest to this app’s WebSocket landmark path. ISLTranslate is the largest continuous ISL–English set with **precomputed Holistic poses**, so training can start without re-encoding all video first.

**Formats:**
- INCLUDE: category video zips + `Train_Test_Split` CSVs; their `generate_keypoints.py` writes landmark files.
- ISLTranslate: `ISLTranslate.csv` (uid → English), video tar, `mediapipe_holistic_poses*.tar.gz`.
- CISLR: HF gated word videos + labels.
- ISL-CSLTR: videos + sentence CSVs / frames; boundary detail varies by release — verify in the downloaded package.

**Continuous sentence-level?** INCLUDE/CISLR: **no**. ISLTranslate/iSign: **yes**. ISL-CSLTR: **yes**, small.

**Labels:** INCLUDE/CISLR = word glosses. ISLTranslate/iSign = English text (**no official gloss sequence**). ISL-CSLTR = sentences (+ claimed time boundaries).

**Preprocessing:** resample to fixed FPS (e.g. 15–25), run or load MediaPipe Holistic, body-center normalize, scale by shoulder width, mirror augment, pad/truncate sequences, filter missing hands, map left/right consistently.

**Commercial warning:** CISLR and iSign are **non-commercial**. Shipping a commercial product requires a license plan or self-collected data.

---

## 4. Model recommendation

### SoftmaxClassifier — keep, do not “upgrade”

Keep `backend/app/ml/numpy_clf.py` for the **personal AAC** Train → Promote path (user’s Hello/Yes shapes). It cannot become continuous ISL.

### B. Input representation

**Recommend: MediaPipe Holistic-style landmarks (hands + upper-body pose + optional face subset), not raw video in production.**

| Option | Verdict |
|---|---|
| Raw video | Best accuracy potential; heavy GPU, privacy, bandwidth; mismatches current WS | Use offline training only if needed |
| Hands only (current Talk) | Enough for AAC; **weak for ISL** (non-manuals, body shift) | Keep for AAC mode |
| Pose only | Misses finger detail | Incomplete |
| Hands + pose + face | Matches ISLTranslate/INCLUDE pipelines; fits browser MediaPipe | **Production choice** |

The repo already loads Hand + Pose landmarkers separately. Target: Holistic (or Hand+Pose+Face Mesh) in the browser, send one structured frame per tick — same WS pattern as today.

Suggested live feature vector per frame (starting point):

- Hands: 2 × 21 × 3 = 126  
- Pose (upper body subset, e.g. 25 keypoints × 3) ≈ 75  
- Face (optional subset, e.g. 20–68 × 3) ≈ 60–204  
- **Start without full face** if latency hurts; add face if ablation on ISLTranslate poses helps.

Normalize: subtract mid-hip or mid-shoulder; divide by shoulder width; zero-fill missing hand; keep handedness flag.

### C. Temporal model

**Phase 2 — isolated word (INCLUDE):**  
BiLSTM or small Transformer encoder + softmax over word classes.

| Spec | Phase 2 recommendation |
|---|---|
| Input | `(B, T, F)` e.g. `T=64`, `F=126` (hands) or `F≈200+` with pose |
| FPS | Sample landmarks at **15 FPS** from 30 FPS cam (stride 2) |
| Window | 64 frames ≈ 4.3 s at 15 FPS |
| Stride (live) | 8–16 frames (~0.5–1 s) |
| Norm | Body-relative as above |
| Arch | 2–3 layer BiLSTM (hidden 256) + FC, or 4-layer Transformer encoder |
| Output | Softmax over INCLUDE / CISLR word IDs |
| Boundaries | Idle / low motion energy between windows (reuse `hand_motion_energy` idea) |

**Phase 3 — continuous (ISLTranslate):**  
**Pose → English Transformer encoder–decoder** (gloss-free), as in ISLTranslate / Pose-SLT style demos.

| Spec | Phase 3 recommendation |
|---|---|
| Input | `(B, T, F)` variable `T` (pad/pack); typical clip 32–256 frames |
| Output | English token sequence (BPE / SentencePiece) |
| Arch | Transformer Enc–Dec with cross-attention; optionally CTC Sign2Gloss **only if** ISL-CSLTR-style glosses are adopted |
| Variable length | Padding + attention mask; inference until utterance end |
| Boundaries | **Utterance-level** first: user pause / “Send” / low motion for N ms — not frame-perfect sign cuts |
| Gloss path | Optional later: CTC BiLSTM → glosses → LLM English. ISLTranslate alone does **not** give official glosses |

CTC is appropriate when time-aligned glosses exist. Sequence-to-sequence / Transformer is appropriate for ISLTranslate-style video→English. Do not force CTC onto a gloss-free dataset.

**Do not** replace SoftmaxClassifier with a Transformer inside `numpy_clf.py`. New code lives under a new module (e.g. `backend/app/ml/temporal/` or `backend/app/ml/isl/`).

---

## 5. API design

### Keep
`POST /ml/polish-sentence` — user-line grammar only.

### Add
`POST /ml/reply` — conversational answer.

**Request**

```json
{
  "sentence": "I want help.",
  "tokens": ["Want", "Help"],
  "history": [
    { "role": "user", "content": "Hello, how are you." },
    { "role": "assistant", "content": "Hello. I am here to help." }
  ],
  "session_id": "optional-uuid",
  "locale": "en-IN"
}
```

**Response**

```json
{
  "status": "ok",
  "reply": "Sure. What do you need help with?",
  "source": "ollama",
  "detail": null,
  "history": []
}
```

**On failure**

```json
{
  "status": "ok",
  "reply": "I heard you. Please sign that again.",
  "source": "fallback",
  "detail": "ConnectionError"
}
```

Still HTTP 200 with `source: fallback` (same pattern as polish), but the **frontend must show** fallback vs live.

**System prompt duties:** You are an AAC assistant. Answer the user’s meaning in 1–2 short sentences. Do not invent medical/legal advice. Do not ask for PII. Stay in English. Do not rewrite the user’s sentence — that is polish’s job.

**History:** last 6–10 turns in the request; optionally persist later in Postgres (`conversation_turns`). No auth today — keep `session_id` client-generated like scoring.

**Streaming:** optional Phase 4 (`text/event-stream`). Phase 1: one shot.

**TTS:** keep Web Speech on the client for the reply; do not block the reply API on TTS.

Env vars (names only): reuse `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS`. Optional later: separate `OLLAMA_REPLY_MODEL` if polish and reply should use different models.

---

## 6. Frontend changes

Files likely touched: `frontend/src/app/page.tsx`, `frontend/src/components/SignCommunicator.tsx`, optionally a small `frontend/src/lib/conversation.ts`.

| UI / state | Behavior |
|---|---|
| `userSentence` | Translated / polished user line (keep showing after tokens clear) |
| `assistantReply` | Latest reply string |
| `conversation` | `{ role, content }[]` |
| `replyLoading` / `replyError` | After speak/send |
| Speak reply | Button + auto-speak after reply arrives |
| Clear / new chat | Fist or “New conversation” clears tokens **and** history |
| Signing state | Existing lock / motion / “Ready” |
| Mode toggle (later) | AAC (13 gestures) vs ISL temporal |

**Phase 1 flow:** signs → sentence → polish (optional) → call `/ml/reply` → show You / Assistant → speak reply.

Do not put reply text into `lastSpoken` alone; keep separate fields so Repeat User vs Repeat Reply stays clear.

---

## 7. Backend changes

| Module | Action |
|---|---|
| `backend/app/services/sentence_polish.py` | **Keep** |
| New `backend/app/services/conversation_reply.py` | **Create** — Ollama chat with history |
| `backend/main.py` | **Modify** — add `POST /ml/reply`; later WS fields for temporal predictions |
| `backend/app/scoring/engine.py`, `sign_stability.py`, Softmax, Train/Promote | **Keep** for AAC |
| New `backend/app/ml/temporal/` (or `isl/`) | **Create** Phase 2–3 — preprocess, train scripts, ONNX/Torch export, infer |
| `backend/app/ml/runtime.py` | **Modify** — new sources without removing heuristic/model |
| `backend/app/models.py` / Postgres | **Extend** Phase 3–4 — optional `conversation_turns`, temporal model metadata |
| Coaching / aviation | **Keep** unchanged |
| `backend/requirements.txt` | **Extend** when training: `torch` (or `onnxruntime` for infer-only). Avoid forcing sklearn |

---

## 8. Training pipeline

```
Dataset (INCLUDE / ISLTranslate)
  → download + license check
  → extract or load MediaPipe Holistic poses
  → normalize (body center, scale, handedness)
  → train / val / test (use official splits)
  → augment (mirror, time warp, jitter, drop keypoints)
  → train PyTorch model
  → evaluate (top-1 / WER / BLEU as appropriate)
  → export ONNX (preferred for Windows FastAPI serve)
  → store artifact + meta in Postgres or backend/app/ml/models/
  → production: onnxruntime.InferenceSession on WS buffer
```

### Artifacts that should eventually exist

| Artifact | Role |
|---|---|
| `include50_bilstm.onnx` + `label_map.json` | Phase 2 isolated |
| `isltranslate_pose2text.onnx` + `tokenizer.json` | Phase 3 continuous |
| `feature_schema.json` | F, FPS, keypoint order — must match browser |
| `ml_runtime` row pointing at temporal/translate version | Promote switch |
| Eval reports (WER/BLEU, confusion) | Gate promote |

Personal `training_samples` Softmax path stays parallel for AAC personalization.

---

## 9. Phased implementation plan

### Phase 1 — Prove conversation on the existing AAC path

| | |
|---|---|
| **Goal** | signs → sentence → chatbot reply → reply on screen (and spoken) |
| **Likely files** | `conversation_reply.py` (new), `main.py`, `SignCommunicator.tsx`, `page.tsx` |
| **Dependencies** | Ollama already used for polish; reuse `OLLAMA_*` |
| **Tests** | API contract; UI shows You/Reply; polish still does not answer; reply failure shows fallback |
| **Expected result** | Demo works with Hello / Want / Help without any ISL dataset |
| **Risks** | Users may think recognition is “ISL” when it is still 13 AAC shapes — label the mode clearly |

### Phase 2 — Better / temporal recognition (isolated)

| | |
|---|---|
| **Goal** | Landmark window → ISL **word** from INCLUDE-50 (then INCLUDE/CISLR) |
| **Likely files** | Holistic (or hand+pose) features on WS; `app/ml/temporal/*`; `runtime.py`; frontend feature payload |
| **Dependencies** | INCLUDE download, PyTorch train machine, ONNXRuntime in backend |
| **Tests** | Offline top-1 on INCLUDE-50 test; live smoke on 10 words; schema version check |
| **Expected result** | Continuous signing not solved, but real ISL words replace finger heuristics for a subset |
| **Risks** | Domain gap (studio vs webcam); NC license if CISLR; Softmax vs temporal confusion in UI |

### Phase 3 — Continuous ISL sentence translation

| | |
|---|---|
| **Goal** | Pause-delimited utterance → English via ISLTranslate-style Pose→Text |
| **Likely files** | Longer buffer / “Send utterance”; translate model; optional polish on model English; then `/ml/reply` |
| **Dependencies** | ISLTranslate poses, GPU training, tokenizer, latency budget |
| **Tests** | BLEU/chrF on held-out ISLTranslate; latency p95; no duplicate utterance fire on same pause |
| **Expected result** | Closest to “I want to go to the market tomorrow” → English → “Sure. What would you like to buy?” |
| **Risks** | Gloss-free model may invent; domain shift; need clear utterance end (button + idle) |

### Phase 4 — Accuracy, latency, deployment

| | |
|---|---|
| **Goal** | iSign scale, face keypoints, streaming reply, schema CI, offline MediaPipe assets, commercial data plan |
| **Dependencies** | Disk/GPU, license review |
| **Tests** | Load/latency budgets; reconnect; silent LLM failure UX; regression on AAC mode |
| **Risks** | Storage cost; license conflicts; scope creep |

---

## Production inference controls

| Problem | Mitigation |
|---|---|
| Duplicate signs | Stabilizer + commit cooldown; for temporal, emit gloss only on rising edge / max in window |
| Repeated predictions | Stride windows; suppress same class until motion reset |
| Premature sentence | Require idle ≥ N ms **or** explicit Send; Phase 1 already waits ~2.2 s after lock drop |
| False boundaries | Motion energy + min utterance length; do not cut on single low-confidence frame |
| Latency | Landmarks not pixels; 15 FPS; ONNX; keep T≤64 for live; heavy SLT on utterance end only |

---

## Language reconstruction (F)

**From INCLUDE-style glosses (learned):**  
`WANT HELP` → template/LLM: “I want help.”  
`HELLO YOU` → “Hello, how are you.”

**From ISLTranslate (learned end-to-end):**  
Pose sequence → `"Let's discuss"` directly. No official gloss middle.

**Not inventable without more data:** Full ISL grammar (non-manuals, classifier constructions). Separate clearly:

- **Learned from dataset:** word IDs (INCLUDE) or English strings (ISLTranslate).
- **Needs extra rules/data:** Gloss→English grammar if building a CTC gloss path; ISL-specific facial grammar.

**Phase 1 AAC example (current labels):**  
`Want, Help` → “I want help.” → reply “What kind of help do you need?”

**Phase 3 example (ISLTranslate-style):**  
Signed continuous clip → model English “I want to go to the market tomorrow.” → reply “Sure. What would you like to buy?”

---

## 10. Risks / limitations

- Current app is **not** ISL; calling Phase 1 “ISL translation” would be misleading.
- Softmax 15-d path cannot scale to continuous ISL — coexistence only.
- ISLTranslate is gloss-free — no reliable gloss sequence without another corpus or weak unsupervised segmentation.
- CISLR / iSign **non-commercial** — check before product use.
- INCLUDE is isolated — continuous fluency needs Phase 3 data.
- Domain gap: dataset lighting/framing vs laptop webcam.
- Windows already blocked sklearn; prefer ONNXRuntime + PyTorch train elsewhere.
- Two-hand ISL and face non-manuals need Holistic; current Talk is hands-primary.
- Conversation quality depends on Ollama model; keep fallbacks.
- Large iSign downloads (~hundreds of GB) are not a Phase 1–2 requirement.

---

## 11. External sources with URLs

| Resource | URL |
|---|---|
| INCLUDE paper | https://doi.org/10.1145/3394171.3413528 |
| INCLUDE code | https://github.com/AI4Bharat/INCLUDE |
| INCLUDE HF mirror | https://huggingface.co/datasets/manojkumarcs/INCLUDE_Dataset |
| CISLR paper | https://aclanthology.org/2022.emnlp-main.707/ |
| CISLR HF | https://huggingface.co/datasets/Exploration-Lab/CISLR |
| ISLTranslate paper HTML | https://arxiv.org/html/2307.05440 |
| ISLTranslate ACL | https://aclanthology.org/2023.findings-acl.665/ |
| ISLTranslate GitHub | https://github.com/Exploration-Lab/ISLTranslate |
| iSign site | https://exploration-lab.github.io/iSign/ |
| iSign HF | https://huggingface.co/datasets/Exploration-Lab/iSign |
| iSign paper | https://arxiv.org/abs/2407.05404 |
| ISL-CSLTR Mendeley | https://data.mendeley.com/datasets/kcmpdxky7p/1 |
| ISL-CSLTR Kaggle | https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset |
| PoseStich-SLT | https://github.com/Exploration-Lab/PoseStich-SLT |
| Example Pose→Text realtime stack | https://github.com/djdhairya/Real-Time-Indian-Sign-Language-ISL-to-English-Translator |
| Existing repo audit | `docs/EXISTING_ARCHITECTURE_AUDIT.md` |

---

## Bottom line for this repository

1. **Phase 1:** Wire conversation (`/ml/reply` + UI) on the current 13-gesture AAC path — proves the product loop without new ML.
2. **Phase 2:** Add a **temporal landmark model** trained on INCLUDE; keep Softmax for personal AAC.
3. **Phase 3:** Add **ISLTranslate Pose→English** for continuous sentences; then reply.
4. Do **not** reinvent Softmax into a translator; do **not** rebuild the whole app.

Suggested next implementation step: implement Phase 1 only (`POST /ml/reply` + You/Reply UI), then revisit Phase 2 when INCLUDE data and a training machine are ready.
