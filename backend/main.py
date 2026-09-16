from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db, init_db, session_scope
from app.scoring.calibration import (
    CalibrationError,
    build_motion_reference,
    build_pose_reference,
    normalize_frames,
    trim_transition_frames,
)
from app.scoring.engine import (
    AVIATION_HOLD_FRAMES,
    HAND_MOTION_ACTIVE_THRESHOLD,
    SEATBELT_MIN_CLOSING_DELTA,
    SIGN_ATTEMPT_FEEDBACK_THRESHOLD,
    SIGN_RECOGNITION_THRESHOLD,
    assess_hand_framing,
    best_sign_pose,
    buffer_looks_like_wave,
    classify_aviation_attempt,
    classify_gesture_attempt,
    hand_motion_energy,
    has_low_wrist_visibility,
    is_aviation_idle,
    score_motion,
    score_pose,
    wrist_visibility,
)
from app.scoring.reference_gestures import (
    ALL_GESTURE_KEYS,
    active_references,
    gesture_type,
    get_exit_pointing,
    get_reference,
    get_seatbelt_demo,
    is_known_gesture,
    load_recorded_references,
    recorded_reference_details,
    reference_status,
    save_recorded_reference,
    clear_all_recorded_references,
    clear_recorded_reference,
)
from app.ml.dataset import (
    DatasetError,
    DatasetStoreError,
    append_gold_hold,
    build_hold_from_sequence,
    get_dataset_stats,
    list_gold_holds,
)
from app.ml.infer import (
    apply_model_to_sign_result,
    demote_to_rules,
    load_live_model,
    promote_model,
    set_shadow_model,
)
from app.ml.registry import RegistryError, latest_meta, list_versions, read_meta
from app.ml.runtime import get_runtime, load_runtime, ws_recognizer_fields
from app.ml.train import TrainError, train_sign_classifier
from app.scoring.sign_meanings import SIGN_MEANINGS
from app.services.coaching import get_coaching_text
from app.sessions import (
    get_session_attempts,
    get_session_recording_meta,
    get_session_recording_path,
    list_sessions,
    log_attempt,
    save_session_recording,
)
from app.config import COACHING_FALLBACK_TEXT


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_db()
    except Exception:
        # init_db already printed a clear console error — keep the API up for
        # live scoring even when history logging is unavailable.
        pass
    load_recorded_references()
    load_runtime()
    load_live_model()
    yield


app = FastAPI(title="Gesture Recognition API", lifespan=lifespan)

# The calibration UI calls these endpoints from the Next.js dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend sends ~every 200ms; keep ~2 seconds of history for motion (DTW) only.
BUFFER_MAX_FRAMES = 16
# Avoid hammering the LLM on every incorrect frame while the pose is held.
COACHING_COOLDOWN_SECONDS = 3.0
# Hard cap so a hung Gemini call never leaves the client spinning forever.
COACHING_TIMEOUT_SECONDS = 5.0
# Holding a pose streams many identical frames — only log a new row periodically.
ATTEMPT_LOG_COOLDOWN_SECONDS = 2.0

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

GestureTypeLiteral = Literal["pose", "motion"]
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class ReferenceRecordRequest(BaseModel):
    gesture: str = Field(..., min_length=1)
    landmark_sequence: list[dict[str, Any]]
    # Optional — inferred from the gesture registry when omitted.
    gesture_type: GestureTypeLiteral | None = None


class SampleRecordRequest(BaseModel):
    gesture: str = Field(..., min_length=1)
    landmark_sequence: list[dict[str, Any]]
    session_id: str | None = None


class RuntimeUpdateRequest(BaseModel):
    source: Literal["heuristic", "model"]
    version: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/modes")
def modes():
    return ["aviation", "sign_language"]


@app.get("/sessions")
def get_sessions(db: Session = Depends(get_db)):
    """List past practice sessions with a short summary per session."""
    try:
        return list_sessions(db)
    except Exception as err:  # noqa: BLE001
        print(f"[db] GET /sessions failed: {err}")
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — check PostgreSQL and DB_* in .env",
        ) from err


@app.get("/sessions/{session_id}")
def get_session_detail(session_id: str, db: Session = Depends(get_db)):
    """Full chronological attempt list for one session."""
    try:
        detail = get_session_attempts(db, session_id)
    except Exception as err:  # noqa: BLE001
        print(f"[db] GET /sessions/{session_id} failed: {err}")
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — check PostgreSQL and DB_* in .env",
        ) from err
    if detail is None:
        # Still allow fetching recording-only metadata if attempts were empty.
        try:
            recording = get_session_recording_meta(db, session_id)
        except Exception:  # noqa: BLE001
            recording = None
        if recording:
            return {
                "session_id": session_id,
                "mode": recording.get("mode", "aviation"),
                "started_at": recording.get("created_at"),
                "total_attempts": 0,
                "summary": {"text": "Recording available"},
                "attempts": [],
                "recording": recording,
            }
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


@app.post("/sessions/{session_id}/recording")
async def upload_session_recording(
    session_id: str,
    video: UploadFile = File(...),
    timeline: str = Form("[]"),
    mode: str = Form("aviation"),
    duration_ms: int | None = Form(None),
    db: Session = Depends(get_db),
):
    """Store webcam capture + event timeline for later replay."""
    if not _SAFE_SESSION_ID.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session_id")

    try:
        events = json.loads(timeline) if timeline else []
        if not isinstance(events, list):
            raise ValueError("timeline must be a JSON array")
    except (json.JSONDecodeError, ValueError) as err:
        raise HTTPException(status_code=422, detail=f"Invalid timeline: {err}") from err

    content_type = video.content_type or "video/webm"
    ext = ".webm"
    if "mp4" in content_type:
        ext = ".mp4"
    elif "ogg" in content_type:
        ext = ".ogg"

    dest = RECORDINGS_DIR / f"{session_id}{ext}"
    payload = await video.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Empty video upload")
    dest.write_bytes(payload)
    print(
        f"[recording] saved session={session_id} bytes={len(payload)} "
        f"events={len(events)} path={dest}"
    )

    try:
        meta = save_session_recording(
            db,
            session_id=session_id,
            mode=mode if mode in ("aviation", "sign_language") else "aviation",
            video_path=str(dest),
            content_type=content_type,
            timeline=events,
            duration_ms=duration_ms,
        )
        db.commit()
    except Exception as err:  # noqa: BLE001
        db.rollback()
        print(f"[recording] db save failed: {err}")
        raise HTTPException(
            status_code=503,
            detail="Could not save recording metadata — check Postgres",
        ) from err

    return {"status": "saved", "recording": meta}


@app.get("/sessions/{session_id}/recording")
def get_session_recording(session_id: str, db: Session = Depends(get_db)):
    try:
        meta = get_session_recording_meta(db, session_id)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=503, detail="Database unavailable") from err
    if meta is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return meta


@app.get("/sessions/{session_id}/recording/video")
def get_session_recording_video(session_id: str, db: Session = Depends(get_db)):
    try:
        located = get_session_recording_path(db, session_id)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=503, detail="Database unavailable") from err
    if located is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    path, content_type = located
    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video file missing on disk")
    return FileResponse(
        path=file_path,
        media_type=content_type,
        filename=file_path.name,
    )


@app.get("/reference/status")
def get_reference_status():
    return reference_status()


@app.get("/reference/gestures")
def list_gestures():
    """Catalog of recordable gestures with type, domain, and meanings."""
    from app.scoring.reference_gestures import GESTURE_REGISTRY

    return {
        name: {
            **spec,
            "meaning": SIGN_MEANINGS.get(name),
            "status": reference_status().get(name, "default"),
        }
        for name, spec in GESTURE_REGISTRY.items()
    }


@app.post("/reference/record")
def record_reference(request: ReferenceRecordRequest):
    gesture = request.gesture.strip()
    if not is_known_gesture(gesture):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown gesture '{gesture}'. "
                f"Supported: {', '.join(ALL_GESTURE_KEYS)}"
            ),
        )

    expected_type = gesture_type(gesture)
    resolved_type = request.gesture_type or expected_type
    if resolved_type != expected_type:
        raise HTTPException(
            status_code=422,
            detail=(
                f"gesture_type '{resolved_type}' does not match "
                f"registered type '{expected_type}' for '{gesture}'."
            ),
        )

    frames = normalize_frames(request.landmark_sequence)
    if not frames:
        raise HTTPException(
            status_code=422, detail="landmark_sequence contained no usable frames."
        )

    trimmed = trim_transition_frames(frames)

    try:
        if expected_type == "pose":
            payload = build_pose_reference(trimmed, gesture=gesture)
        else:
            payload = build_motion_reference(trimmed, gesture=gesture)
    except CalibrationError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    payload["frames_submitted"] = len(frames)
    save_recorded_reference(gesture, payload)

    features = payload.get("reference_features")
    sequence = payload.get("reference_sequence")
    print(
        f"[reference] saved {gesture} type={expected_type} "
        f"features_len={len(features) if isinstance(features, list) else None} "
        f"sequence_len={len(sequence) if isinstance(sequence, list) else None} "
        f"frames_used={payload.get('frames_used')} "
        f"tolerance={payload.get('feature_tolerance')}"
    )
    if isinstance(features, list):
        print(f"[reference] {gesture} reference_features={features}")
    if isinstance(sequence, list):
        print(
            f"[reference] {gesture} reference_sequence_len={len(sequence)} "
            f"first_row={sequence[0] if sequence else None}"
        )

    return {
        "status": "saved",
        "gesture": gesture,
        "gesture_type": expected_type,
        "reference": payload,
        "reference_status": reference_status(),
    }


@app.delete("/reference/{gesture}")
def delete_recorded_reference(gesture: str):
    name = gesture.strip()
    if not is_known_gesture(name):
        raise HTTPException(status_code=422, detail=f"Unknown gesture '{name}'.")
    cleared = clear_recorded_reference(name)
    return {
        "status": "cleared" if cleared else "already_default",
        "gesture": name,
        "reference_status": reference_status(),
    }


@app.post("/reference/reset")
def reset_recorded_references():
    cleared = clear_all_recorded_references()
    return {
        "status": "reset",
        "cleared": cleared,
        "reference_status": reference_status(),
    }


@app.get("/reference/details")
def get_reference_details():
    """Full recorded payloads plus the references currently in use."""
    return {
        "status": reference_status(),
        "recorded": recorded_reference_details(),
        "active": active_references(),
    }


@app.get("/ml/runtime")
def ml_runtime():
    """Talk-path recognizer contract."""
    return get_runtime()


@app.post("/ml/runtime")
def ml_set_runtime(request: RuntimeUpdateRequest):
    """Switch Talk between hardcoded rules and a promoted model."""
    try:
        if request.source == "heuristic":
            runtime = demote_to_rules()
        else:
            version = request.version
            if not version:
                meta = latest_meta()
                if not meta:
                    raise HTTPException(
                        status_code=409, detail="Train a model before promoting."
                    )
                version = str(meta["version"])
            runtime = promote_model(version)
    except RegistryError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return {
        "status": "updated",
        "runtime": runtime,
        "stats": get_dataset_stats(),
    }


@app.get("/ml/stats")
def ml_stats():
    """Gold-hold counts per sign plus overall training progress."""
    try:
        return get_dataset_stats()
    except DatasetStoreError as err:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — check PostgreSQL and DB_* in .env",
        ) from err


@app.get("/ml/models")
def ml_models():
    versions = list_versions()
    models: list[dict[str, Any]] = []
    for version in versions:
        try:
            models.append(read_meta(version))
        except RegistryError:
            continue
    runtime = get_runtime()
    return {
        "live_model_id": runtime.get("live_model_id"),
        "source": runtime.get("source"),
        "models": models,
    }


@app.post("/ml/train")
def ml_train():
    """Fit a softmax classifier on gold holds. Talk stays on rules until promote."""
    try:
        meta = train_sign_classifier()
    except TrainError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    except DatasetStoreError as err:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — check PostgreSQL and DB_* in .env",
        ) from err
    except RegistryError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except Exception as err:  # noqa: BLE001
        print(f"[ml] train failed: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err
    set_shadow_model(str(meta["version"]))
    try:
        stats = get_dataset_stats()
    except DatasetStoreError as err:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — check PostgreSQL and DB_* in .env",
        ) from err
    return {
        "status": "trained",
        "model": meta,
        "stats": stats,
    }


@app.post("/ml/models/{version}/promote")
def ml_promote(version: str):
    """Make this trained model the live Talk recognizer for static signs."""
    try:
        runtime = promote_model(version)
    except RegistryError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return {
        "status": "promoted",
        "runtime": runtime,
        "stats": get_dataset_stats(),
    }


@app.post("/ml/samples")
def ml_add_samples(request: SampleRecordRequest):
    """Append one confirmed hold to PostgreSQL. Does not change Talk."""
    try:
        hold = build_hold_from_sequence(
            gesture=request.gesture,
            landmark_sequence=request.landmark_sequence,
            session_id=request.session_id,
            source="user",
        )
    except DatasetError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    try:
        append_gold_hold(hold)
        stats = get_dataset_stats()
    except DatasetStoreError as err:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — sample was not saved.",
        ) from err
    gold_holds = 0
    for row in stats["gestures"]:
        if row["gesture"] == hold["gesture"]:
            gold_holds = int(row["gold_holds"])
            break

    print(
        f"[ml] saved hold {hold['id']} gesture={hold['gesture']} "
        f"name={hold['name']} frames={hold['frame_count']} gold_holds={gold_holds}"
    )
    return {
        "status": "saved",
        "gesture": hold["gesture"],
        "name": hold["name"],
        "hold_id": hold["id"],
        "frame_count": hold["frame_count"],
        "gold_holds": gold_holds,
        "stats": stats,
    }


@app.get("/ml/samples")
def ml_list_samples(
    gesture: str | None = None,
    name: str | None = None,
    limit: int = 200,
):
    """Return gold holds from Postgres, optionally filtered by sign key or spoken name."""
    try:
        holds = list_gold_holds(gesture=gesture, name=name, limit=limit)
    except DatasetStoreError as err:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — check PostgreSQL and DB_* in .env",
        ) from err
    return {
        "status": "ok",
        "count": len(holds),
        "samples": holds,
    }


def _none_result(timestamp: Any, **extra: Any) -> dict[str, Any]:
    return {
        "status": "received",
        "timestamp": timestamp,
        "gesture": "none",
        "correct": False,
        "score": 0.0,
        "deviations": [],
        **extra,
    }


def _evaluate_aviation(
    landmarks: dict[str, Any],
    buffer: deque[dict[str, Any]],
    hold_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_vis, right_vis = wrist_visibility(landmarks)
    visibility = {
        "left_wrist_visibility": round(left_vis, 4),
        "right_wrist_visibility": round(right_vis, 4),
    }

    if has_low_wrist_visibility(landmarks):
        return {
            "gesture": "none",
            "correct": False,
            "score": 0.0,
            "reason": "low_visibility",
            "deviations": [],
            **visibility,
        }

    frames = list(buffer)
    previous = None
    if hold_state and hold_state.get("gesture") in (
        "exit_pointing",
        "seatbelt_demo",
    ):
        previous = hold_state["gesture"]

    attempt, classify_debug = classify_aviation_attempt(
        landmarks, previous=previous
    )

    # Hold last non-idle gesture briefly against MediaPipe flicker.
    if hold_state is not None:
        if attempt in ("exit_pointing", "seatbelt_demo"):
            hold_state["gesture"] = attempt
            hold_state["frames"] = AVIATION_HOLD_FRAMES
        elif int(hold_state.get("frames") or 0) > 0 and hold_state.get("gesture"):
            hold_state["frames"] = int(hold_state["frames"]) - 1
            if hold_state["frames"] > 0:
                attempt = hold_state["gesture"]
                classify_debug = {
                    **classify_debug,
                    "held": True,
                    "hold_frames_left": hold_state["frames"],
                }
            else:
                hold_state["gesture"] = None
        else:
            hold_state["gesture"] = None
            hold_state["frames"] = 0

    print(
        f"[classify] mode=aviation attempt={attempt} "
        f"debug={classify_debug}"
    )

    # Standing still with no held training gesture → stay silent.
    if attempt == "none":
        if is_aviation_idle(frames):
            return {
                "gesture": "none",
                "correct": False,
                "score": 0.0,
                "reason": "idle",
                "classify_debug": classify_debug,
                "deviations": [],
                **visibility,
            }
        return {
            "gesture": "none",
            "correct": False,
            "score": 0.0,
            "reason": "no_attempt",
            "classify_debug": classify_debug,
            "deviations": [],
            **visibility,
        }

    if attempt == "exit_pointing":
        pose_result = score_pose(landmarks, get_exit_pointing())
        return {
            "gesture": "exit_pointing",
            "correct": bool(pose_result["correct"]),
            "score": pose_result["score"],
            "left_arm_angle": pose_result["left_arm_angle"],
            "right_arm_angle": pose_result["right_arm_angle"],
            "target_angle": pose_result["target_angle"],
            "tolerance": pose_result["tolerance"],
            "pass_band_deg": pose_result.get("pass_band_deg"),
            "left_arm_score": pose_result.get("left_arm_score"),
            "right_arm_score": pose_result.get("right_arm_score"),
            "horizontal_score": pose_result.get("horizontal_score"),
            "deviations": pose_result["deviations"],
            "hints": pose_result.get("hints", []),
            "classify_debug": classify_debug,
            **visibility,
        }

    motion_result = score_motion(frames, get_seatbelt_demo())
    if not motion_result.get("ready"):
        return {
            "gesture": "none",
            "correct": False,
            "score": 0.0,
            "reason": "motion_buffering",
            "attempted_gesture": "seatbelt_demo",
            "buffer_progress": motion_result.get("buffer_progress", 0.0),
            "buffer_frames": motion_result.get("buffer_frames"),
            "buffer_needed": motion_result.get("buffer_needed"),
            "deviations": motion_result.get("deviations", []),
            "hints": motion_result.get("hints", []),
            "classify_debug": classify_debug,
            **visibility,
        }

    # Static hands near waist without closing motion → stay silent.
    closing = motion_result.get("closing_delta")
    if (
        not motion_result.get("correct")
        and closing is not None
        and float(closing) < SEATBELT_MIN_CLOSING_DELTA
        and is_aviation_idle(frames)
    ):
        return {
            "gesture": "none",
            "correct": False,
            "score": 0.0,
            "reason": "idle",
            "classify_debug": classify_debug,
            "deviations": [],
            **visibility,
        }

    return {
        "gesture": "seatbelt_demo",
        "correct": bool(motion_result["correct"]),
        "score": motion_result["score"],
        "dtw_distance": motion_result.get("dtw_distance"),
        "deviations": motion_result["deviations"],
        "hints": motion_result.get("hints", []),
        "classify_debug": classify_debug,
        **visibility,
    }


def _evaluate_sign_language(
    landmarks: dict[str, Any],
    buffer: deque[dict[str, Any]],
) -> dict[str, Any]:
    """Recognize a sign — always return live feedback; speak only when matched."""
    framing = assess_hand_framing(landmarks)
    print(
        f"[hand_framing] ok={framing.get('ok')} reason={framing.get('reason')} "
        f"in_frame={framing.get('points_in_frame')}/{framing.get('points_total')} "
        f"bbox_area={framing.get('bbox_area')} "
        f"sample_visibility={framing.get('sample_visibility')!r}"
    )

    frames = list(buffer)
    motion = hand_motion_energy(frames)
    motion_active = motion >= HAND_MOTION_ACTIVE_THRESHOLD
    hand_style = "motion" if motion_active else "hold"
    motion_meta = {
        "motion_energy": round(motion, 4),
        "motion_active": motion_active,
        "hand_style": hand_style,
    }

    attempt = classify_gesture_attempt(
        landmarks, mode="sign_language", buffer=frames
    )
    print(f"[classify] mode=sign_language attempt={attempt} motion={motion:.4f}")

    best_gesture, best_score, pose_scores, pose_detail = best_sign_pose(
        landmarks,
        classified_attempt=attempt if attempt not in ("wave", "none") else None,
    )
    print(f"[sign_scores] {pose_scores} best={best_gesture}:{best_score}")

    if not framing.get("ok"):
        return {
            "gesture": "none",
            "correct": False,
            "score": float(best_score or 0.0),
            "reason": "low_visibility",
            "framing": framing,
            "scores": pose_scores,
            "attempted_gesture": best_gesture or attempt,
            "deviations": pose_detail.get("deviations", []),
            **motion_meta,
        }

    if attempt == "wave":
        motion_result = score_motion(frames, get_reference("wave"))
        wave_score = float(motion_result.get("score") or 0.0)
        scores = {**pose_scores, "wave": wave_score}
        wag = buffer_looks_like_wave(frames)
        # A held palm is Hello. Goodbye only after a real left-right wag.
        if wag:
            return {
                "gesture": "wave",
                "correct": True,
                "meaning": SIGN_MEANINGS["wave"],
                "score": max(wave_score, 0.72),
                "reason": None,
                "attempted_gesture": "wave",
                "hint": None,
                "dtw_distance": motion_result.get("dtw_distance"),
                "buffer_progress": 1.0,
                "scores": scores,
                "deviations": motion_result.get("deviations", []),
                "framing": framing,
                **motion_meta,
            }
        progress = motion_result.get("buffer_progress", 0.0)
        if motion_active:
            progress = max(float(progress or 0.0), 0.4)
        return {
            "gesture": "wave",
            "correct": False,
            "score": wave_score,
            "meaning": None,
            "reason": "motion_in_progress" if motion_active else "motion_buffering",
            "attempted_gesture": "wave",
            "buffer_progress": progress,
            "buffer_frames": motion_result.get("buffer_frames"),
            "buffer_needed": motion_result.get("buffer_needed"),
            "scores": scores,
            "deviations": motion_result.get("deviations", []),
            "framing": framing,
            "hint": "Wag your hand left and right twice for Goodbye",
            **motion_meta,
        }

    display_gesture = best_gesture
    if attempt == "fist":
        display_gesture = "fist"
        best_score = max(
            float(pose_scores.get("fist") or 0.0),
            SIGN_RECOGNITION_THRESHOLD + 0.08,
        )
    elif attempt == "thumbs_up":
        display_gesture = "thumbs_up"
        best_score = max(
            float(pose_scores.get("thumbs_up") or 0.0),
            SIGN_RECOGNITION_THRESHOLD + 0.08,
        )
    elif (
        attempt in pose_scores
        and attempt not in ("none", "wave")
        and float(pose_scores.get(attempt, 0.0))
        >= float(pose_scores.get(best_gesture or "", 0.0)) - 0.05
    ):
        display_gesture = attempt
        best_score = float(pose_scores.get(attempt, best_score))

    matched = bool(display_gesture and best_score >= SIGN_RECOGNITION_THRESHOLD)

    if matched:
        return {
            "gesture": display_gesture,
            "correct": True,
            "meaning": SIGN_MEANINGS.get(display_gesture or ""),
            "score": best_score,
            "scores": pose_scores,
            "classified_as": attempt,
            "deviations": pose_detail.get("deviations", []),
            "framing": framing,
            **motion_meta,
        }

    if display_gesture and best_score >= SIGN_ATTEMPT_FEEDBACK_THRESHOLD:
        return {
            "gesture": display_gesture,
            "correct": False,
            "meaning": None,
            "score": best_score,
            "reason": "not_recognized",
            "attempted_gesture": display_gesture,
            "scores": pose_scores,
            "classified_as": attempt,
            "deviations": pose_detail.get("deviations", []),
            "framing": framing,
            **motion_meta,
        }

    return {
        "gesture": "none",
        "correct": False,
        "score": best_score,
        "reason": "not_recognized" if display_gesture else "no_attempt",
        "attempted_gesture": display_gesture or attempt,
        "scores": pose_scores,
        "classified_as": attempt,
        "deviations": pose_detail.get("deviations", []),
        "framing": framing,
        **motion_meta,
    }


def _coaching_payload(scored: dict[str, Any]) -> dict[str, Any]:
    """Shape the metrics blob passed into the coaching provider."""
    gesture = scored.get("gesture")
    payload: dict[str, Any] = {
        "gesture": gesture,
        "score": scored.get("score"),
        "deviations": scored.get("deviations", []),
    }

    if gesture == "exit_pointing":
        payload.update(
            {
                "left_arm_angle": scored.get("left_arm_angle"),
                "right_arm_angle": scored.get("right_arm_angle"),
                "target_angle": scored.get("target_angle"),
                "tolerance": scored.get("tolerance"),
                "pass_band_deg": scored.get("pass_band_deg"),
                "hints": scored.get("hints", []),
            }
        )
    elif gesture == "seatbelt_demo":
        payload["dtw_distance"] = scored.get("dtw_distance")
        payload["hints"] = scored.get("hints", [])

    return payload


def _persist_attempt(
    *,
    session_id: str,
    mode: str | None,
    response: dict[str, Any],
    last_log_key: list[Any],
    last_log_mono: list[float],
) -> None:
    """Insert an Attempt when a real gesture was detected (skips gesture=\"none\")."""
    gesture = response.get("gesture")
    if not gesture or gesture == "none":
        return
    if not isinstance(mode, str) or not mode:
        return

    loop = asyncio.get_running_loop()
    now = loop.time()
    key = (
        session_id,
        mode,
        gesture,
        response.get("correct"),
        response.get("meaning"),
        bool(response.get("coaching_text")),
    )
    # Debounce identical held poses so we don't write 5 rows/second.
    if (
        last_log_key
        and last_log_key[0] == key
        and (now - last_log_mono[0]) < ATTEMPT_LOG_COOLDOWN_SECONDS
    ):
        return

    try:
        with session_scope() as db:
            log_attempt(
                db,
                session_id=session_id,
                mode=mode,
                gesture=str(gesture),
                score=float(response.get("score") or 0.0),
                correct=response.get("correct"),
                coaching_text=response.get("coaching_text"),
                meaning=response.get("meaning"),
            )
        last_log_key[:] = [key]
        last_log_mono[:] = [now]
    except Exception as err:  # noqa: BLE001 — never break the WS loop for logging
        # Live scoring/coaching continues; history logging is best-effort.
        print(f"[db] failed to log attempt (practice continues): {err}")


@app.websocket("/ws/landmarks")
async def landmarks_ws(websocket: WebSocket):
    await websocket.accept()
    print("[ws/landmarks] client connected")

    landmark_buffer: deque[dict[str, Any]] = deque(maxlen=BUFFER_MAX_FRAMES)
    aviation_hold: dict[str, Any] = {"gesture": None, "frames": 0}
    last_coaching_mono = 0.0
    last_coaching_text: str | None = None
    coaching_in_flight = False
    last_mode: str | None = None
    connection_session_id = uuid.uuid4().hex
    last_log_key: list[Any] = []
    last_log_mono: list[float] = [0.0]

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[ws/landmarks] invalid JSON: {raw[:200]}")
                await websocket.send_json(
                    {"status": "error", "detail": "invalid_json"}
                )
                continue

            timestamp = message.get("timestamp")
            mode = message.get("mode")
            landmarks = message.get("landmarks") or {}
            client_session = message.get("session_id")
            session_id = (
                str(client_session)
                if isinstance(client_session, str) and client_session.strip()
                else connection_session_id
            )

            if mode != last_mode:
                landmark_buffer.clear()
                aviation_hold["gesture"] = None
                aviation_hold["frames"] = 0
                last_coaching_text = None
                last_mode = mode if isinstance(mode, str) else None

            response = _none_result(timestamp)
            sign_model_fields: dict[str, Any] | None = None
            sign_recognizer: str | None = None

            if isinstance(landmarks, dict) and landmarks:
                # Only buffer frames that match the active landmark schema.
                aviation_frame = "leftShoulder" in landmarks or "leftWrist" in landmarks
                hand_frame = "wrist" in landmarks and "index_tip" in landmarks

                if mode == "aviation" and aviation_frame and not hand_frame:
                    landmark_buffer.append(landmarks)
                elif mode == "sign_language" and hand_frame:
                    landmark_buffer.append(landmarks)

                if mode == "aviation":
                    if not aviation_frame or hand_frame:
                        await websocket.send_json(response)
                        continue

                    scored = _evaluate_aviation(
                        landmarks, landmark_buffer, hold_state=aviation_hold
                    )
                    response.update(scored)
                    print(
                        f"[ws/landmarks] aviation gesture={scored.get('gesture')} "
                        f"correct={scored.get('correct')} score={scored.get('score')} "
                        f"L={scored.get('left_arm_angle')} R={scored.get('right_arm_angle')} "
                        f"reason={scored.get('reason')}"
                    )

                    needs_coaching = (
                        scored.get("gesture") in ("exit_pointing", "seatbelt_demo")
                        and scored.get("correct") is False
                    )

                    if needs_coaching:
                        loop = asyncio.get_running_loop()
                        now = loop.time()
                        cooldown_ok = (
                            now - last_coaching_mono
                        ) >= COACHING_COOLDOWN_SECONDS

                        if last_coaching_text and not cooldown_ok:
                            response["coaching_text"] = last_coaching_text
                            _persist_attempt(
                                session_id=session_id,
                                mode=mode,
                                response=response,
                                last_log_key=last_log_key,
                                last_log_mono=last_log_mono,
                            )
                            await websocket.send_json(response)
                            continue

                        if coaching_in_flight:
                            response["coaching_pending"] = True
                            if last_coaching_text:
                                response["coaching_text"] = last_coaching_text
                            await websocket.send_json(response)
                            continue

                        response["coaching_pending"] = True
                        await websocket.send_json(response)

                        coaching_in_flight = True
                        last_coaching_mono = now
                        deviation_data = _coaching_payload(scored)
                        score_snapshot = dict(response)

                        async def _send_coaching(
                            snapshot: dict[str, Any] = score_snapshot,
                            payload: dict[str, Any] = deviation_data,
                            sid: str = session_id,
                            active_mode: str | None = mode,
                        ) -> None:
                            nonlocal last_coaching_text, coaching_in_flight
                            try:
                                try:
                                    coaching_text = await asyncio.wait_for(
                                        asyncio.to_thread(
                                            get_coaching_text, payload
                                        ),
                                        timeout=COACHING_TIMEOUT_SECONDS,
                                    )
                                except asyncio.TimeoutError:
                                    print(
                                        f"[coaching] timed out after "
                                        f"{COACHING_TIMEOUT_SECONDS}s — using fallback"
                                    )
                                    coaching_text = COACHING_FALLBACK_TEXT
                                last_coaching_text = coaching_text
                                follow_up = {
                                    **snapshot,
                                    "coaching_pending": False,
                                    "coaching_text": coaching_text,
                                }
                                _persist_attempt(
                                    session_id=sid,
                                    mode=active_mode,
                                    response=follow_up,
                                    last_log_key=last_log_key,
                                    last_log_mono=last_log_mono,
                                )
                                await websocket.send_json(follow_up)
                                print(f"[coaching] {coaching_text!r}")
                            except Exception as err:  # noqa: BLE001
                                print(f"[coaching] follow-up failed: {err}")
                                try:
                                    follow_up = {
                                        **snapshot,
                                        "coaching_pending": False,
                                        "coaching_text": COACHING_FALLBACK_TEXT,
                                    }
                                    last_coaching_text = COACHING_FALLBACK_TEXT
                                    await websocket.send_json(follow_up)
                                except Exception:  # noqa: BLE001
                                    pass
                            finally:
                                coaching_in_flight = False

                        asyncio.create_task(_send_coaching())
                        continue

                    last_coaching_text = None
                    _persist_attempt(
                        session_id=session_id,
                        mode=mode,
                        response=response,
                        last_log_key=last_log_key,
                        last_log_mono=last_log_mono,
                    )

                elif mode == "sign_language":
                    if not hand_frame:
                        response = _none_result(
                            timestamp, reason="waiting_for_hand_landmarks"
                        )
                        response.update(ws_recognizer_fields())
                        await websocket.send_json(response)
                        continue

                    scored = _evaluate_sign_language(landmarks, landmark_buffer)
                    scored, sign_model_fields, sign_recognizer = (
                        apply_model_to_sign_result(scored, landmarks)
                    )
                    response = {
                        "status": "received",
                        "timestamp": timestamp,
                        "gesture": scored.get("gesture", "none"),
                        "correct": bool(scored.get("correct")),
                        "score": scored.get("score", 0.0),
                        "deviations": scored.get("deviations", []),
                    }
                    if "meaning" in scored:
                        response["meaning"] = scored.get("meaning")
                    if scored.get("reason"):
                        response["reason"] = scored["reason"]
                    if scored.get("attempted_gesture"):
                        response["attempted_gesture"] = scored["attempted_gesture"]
                    if scored.get("hint"):
                        response["hint"] = scored["hint"]
                    if "buffer_progress" in scored:
                        response["buffer_progress"] = scored["buffer_progress"]
                    if "buffer_frames" in scored:
                        response["buffer_frames"] = scored["buffer_frames"]
                    if "buffer_needed" in scored:
                        response["buffer_needed"] = scored["buffer_needed"]
                    if "dtw_distance" in scored:
                        response["dtw_distance"] = scored["dtw_distance"]
                    if "scores" in scored:
                        response["scores"] = scored["scores"]
                    if "framing" in scored:
                        response["framing"] = scored["framing"]
                    if "classified_as" in scored:
                        response["classified_as"] = scored["classified_as"]
                    if "motion_active" in scored:
                        response["motion_active"] = scored["motion_active"]
                    if "motion_energy" in scored:
                        response["motion_energy"] = scored["motion_energy"]
                    if "hand_style" in scored:
                        response["hand_style"] = scored["hand_style"]

                    print(
                        f"[ws/landmarks] sign gesture={response.get('gesture')} "
                        f"correct={response.get('correct')} "
                        f"meaning={response.get('meaning')} "
                        f"score={response.get('score')} reason={response.get('reason')} "
                        f"motion={response.get('motion_active')} "
                        f"style={response.get('hand_style')}"
                    )
                    _persist_attempt(
                        session_id=session_id,
                        mode=mode,
                        response=response,
                        last_log_key=last_log_key,
                        last_log_mono=last_log_mono,
                    )

            if mode == "sign_language":
                response.update(
                    ws_recognizer_fields(sign_model_fields, sign_recognizer)
                )

            await websocket.send_json(response)
    except WebSocketDisconnect:
        print("[ws/landmarks] client disconnected")
