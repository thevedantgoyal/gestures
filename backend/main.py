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
    SEATBELT_MIN_CLOSING_DELTA,
    SIGN_ATTEMPT_FEEDBACK_THRESHOLD,
    SIGN_RECOGNITION_THRESHOLD,
    assess_hand_framing,
    best_sign_pose,
    classify_aviation_attempt,
    classify_gesture_attempt,
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
)
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
    load_recorded_references()
    try:
        init_db()
    except Exception:
        # init_db already printed a clear console error — keep the API up for
        # live scoring even when history logging is unavailable.
        pass
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
BUFFER_MAX_FRAMES = 10
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


@app.get("/reference/details")
def get_reference_details():
    """Full recorded payloads plus the references currently in use."""
    return {
        "status": reference_status(),
        "recorded": recorded_reference_details(),
        "active": active_references(),
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
    attempt = classify_gesture_attempt(
        landmarks, mode="sign_language", buffer=frames
    )
    print(f"[classify] mode=sign_language attempt={attempt}")

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
        }

    if attempt == "wave":
        motion_result = score_motion(frames, get_reference("wave"))
        scores = {**pose_scores, "wave": float(motion_result.get("score") or 0.0)}
        if not motion_result.get("ready"):
            return {
                "gesture": "wave",
                "correct": False,
                "score": 0.0,
                "meaning": None,
                "reason": "motion_buffering",
                "attempted_gesture": "wave",
                "buffer_progress": motion_result.get("buffer_progress", 0.0),
                "buffer_frames": motion_result.get("buffer_frames"),
                "buffer_needed": motion_result.get("buffer_needed"),
                "scores": scores,
                "deviations": motion_result.get("deviations", []),
                "framing": framing,
            }
        wave_score = float(motion_result.get("score") or 0.0)
        matched = wave_score >= SIGN_RECOGNITION_THRESHOLD and bool(
            motion_result.get("matched")
        )
        return {
            "gesture": "wave",
            "correct": matched,
            "meaning": SIGN_MEANINGS["wave"] if matched else None,
            "score": wave_score,
            "reason": None if matched else "not_recognized",
            "attempted_gesture": "wave",
            "hint": None if matched else "Try a slightly wider, slower wave",
            "dtw_distance": motion_result.get("dtw_distance"),
            "scores": scores,
            "deviations": motion_result.get("deviations", []),
            "framing": framing,
        }

    display_gesture = best_gesture
    if (
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
                        await websocket.send_json(response)
                        continue

                    scored = _evaluate_sign_language(landmarks, landmark_buffer)
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

                    print(
                        f"[ws/landmarks] sign gesture={response.get('gesture')} "
                        f"correct={response.get('correct')} "
                        f"meaning={response.get('meaning')} "
                        f"score={response.get('score')} reason={response.get('reason')}"
                    )
                    _persist_attempt(
                        session_id=session_id,
                        mode=mode,
                        response=response,
                        last_log_key=last_log_key,
                        last_log_mono=last_log_mono,
                    )

            await websocket.send_json(response)
    except WebSocketDisconnect:
        print("[ws/landmarks] client disconnected")
