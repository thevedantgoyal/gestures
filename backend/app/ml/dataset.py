"""Gold training holds for sign language. One Postgres row = one confirmed hold."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import func, select

from app.db import session_scope
from app.ml.registry import latest_meta
from app.ml.runtime import get_runtime
from app.models import TrainingSample
from app.scoring.calibration import normalize_frames, trim_transition_frames
from app.scoring.engine import hand_pose_features, hand_wave_features
from app.scoring.reference_gestures import gesture_type
from app.scoring.sign_meanings import SIGN_GESTURES, SIGN_MEANINGS

FEATURE_VERSION = 1
FEATURE_DIM = 15
MAX_FRAMES_PER_HOLD = 10
MIN_FRAMES_PER_HOLD = 3
TRAINABLE_HOLDS = 50
PROMOTE_HOLDS = 200
MIN_TRAIN_HOLDS_PER_CLASS = 3
MIN_TRAIN_CLASSES = 2

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLES_PATH = DATA_DIR / "samples.jsonl"

STATIC_GESTURES = tuple(name for name in SIGN_GESTURES if name != "wave")

HoldStatus = Literal["collecting", "trainable", "ready"]


class DatasetError(ValueError):
    """Raised when a submitted hold cannot be stored."""


class DatasetStoreError(RuntimeError):
    """Raised when PostgreSQL cannot persist or load gold holds."""


def _even_sample(rows: list[list[float]], max_len: int) -> list[list[float]]:
    if len(rows) <= max_len:
        return rows
    if max_len <= 1:
        return [rows[-1]]
    step = (len(rows) - 1) / (max_len - 1)
    return [rows[int(round(index * step))] for index in range(max_len)]


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _handedness(landmarks: dict[str, Any]) -> str | None:
    value = landmarks.get("handedness")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _status_for_holds(holds: int) -> HoldStatus:
    if holds >= PROMOTE_HOLDS:
        return "ready"
    if holds >= TRAINABLE_HOLDS:
        return "trainable"
    return "collecting"


def _live_path() -> str:
    runtime = get_runtime()
    source = runtime.get("source")
    if source == "model":
        return "model"
    if source == "hybrid":
        return "hybrid"
    return "rules"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, set)):
        return [_jsonable(item) for item in value]
    return None


def hold_from_record(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a dict (JSONL or API) into a gold-hold record."""
    name = payload.get("gesture")
    if name not in SIGN_GESTURES:
        return None
    hold_id = str(payload.get("id") or uuid.uuid4().hex)
    kind = payload.get("gesture_type") or gesture_type(name) or "pose"
    created = payload.get("created_at") or _iso_now()
    if isinstance(created, datetime):
        created = created.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    session_id = payload.get("session_id")
    return {
        "id": hold_id,
        "gesture": name,
        "name": str(payload.get("name") or SIGN_MEANINGS[name]),
        "source": str(payload.get("source") or "user"),
        "quality": str(payload.get("quality") or "ok"),
        "split": str(payload.get("split") or "train"),
        "session_id": session_id.strip() if isinstance(session_id, str) else None,
        "handedness": payload.get("handedness")
        if isinstance(payload.get("handedness"), str)
        else None,
        "created_at": created,
        "feature_version": int(payload.get("feature_version") or FEATURE_VERSION),
        "gesture_type": kind,
        "frame_count": int(payload.get("frame_count") or 0),
        "features": _jsonable(payload.get("features")),
        "sequence": _jsonable(payload.get("sequence")),
        "landmarks": _jsonable(payload.get("landmarks")),
    }


def _row_to_hold(row: TrainingSample) -> dict[str, Any]:
    created = row.created_at
    if created is not None:
        created_at = created.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    else:
        created_at = _iso_now()
    return {
        "id": row.id,
        "gesture": row.gesture,
        "name": row.name,
        "source": row.source,
        "quality": row.quality,
        "split": row.split,
        "session_id": row.session_id,
        "handedness": row.handedness,
        "created_at": created_at,
        "feature_version": row.feature_version,
        "gesture_type": row.gesture_type,
        "frame_count": row.frame_count,
        "features": row.features,
        "sequence": row.sequence,
        "landmarks": row.landmarks,
    }


def build_hold_from_sequence(
    *,
    gesture: str,
    landmark_sequence: list[Any],
    session_id: str | None = None,
    source: str = "user",
) -> dict[str, Any]:
    """Turn a recorded landmark clip into one gold hold (downsampled features)."""
    name = gesture.strip()
    if name not in SIGN_GESTURES:
        raise DatasetError(
            f"Unknown sign '{gesture}'. Supported: {', '.join(SIGN_GESTURES)}"
        )

    if source != "user":
        raise DatasetError("Only confirmed user holds can be stored.")

    frames = normalize_frames(landmark_sequence)
    if not frames:
        raise DatasetError("landmark_sequence contained no usable frames.")

    trimmed = trim_transition_frames(frames)
    payloads = [landmarks for _ts, landmarks in trimmed]
    kind = gesture_type(name)
    handedness = _handedness(payloads[0]) if payloads else None

    if kind == "motion":
        sequence: list[list[float]] = []
        for landmarks in payloads:
            row = hand_wave_features(landmarks)
            if row:
                sequence.append([float(value) for value in row])
        if len(sequence) < MIN_FRAMES_PER_HOLD:
            raise DatasetError(
                "Not enough wave motion captured — wag left and right in frame."
            )
        sequence = _even_sample(sequence, MAX_FRAMES_PER_HOLD)
        features: list[list[float]] | None = None
        stored_sequence: list[list[float]] | None = sequence
        frame_count = len(sequence)
    else:
        feature_rows: list[list[float]] = []
        for landmarks in payloads:
            row = hand_pose_features(landmarks)
            if row:
                feature_rows.append([float(value) for value in row])
        if len(feature_rows) < MIN_FRAMES_PER_HOLD:
            raise DatasetError(
                "Not enough hand data captured — keep one hand clearly in frame."
            )
        feature_rows = _even_sample(feature_rows, MAX_FRAMES_PER_HOLD)
        features = feature_rows
        stored_sequence = None
        frame_count = len(feature_rows)

    return {
        "id": uuid.uuid4().hex,
        "gesture": name,
        "name": SIGN_MEANINGS[name],
        "source": source,
        "quality": "ok",
        "split": "train",
        "session_id": session_id.strip() if isinstance(session_id, str) else None,
        "handedness": handedness,
        "created_at": _iso_now(),
        "feature_version": FEATURE_VERSION,
        "gesture_type": kind,
        "frame_count": frame_count,
        "features": features,
        "sequence": stored_sequence,
        "landmarks": _jsonable(payloads),
    }


def append_gold_hold(record: dict[str, Any]) -> dict[str, Any]:
    hold = hold_from_record(record)
    if hold is None:
        raise DatasetError("Cannot store an invalid gold hold.")
    created_raw = hold["created_at"]
    try:
        created_at = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
    except ValueError:
        created_at = datetime.now(timezone.utc)
    try:
        with session_scope() as db:
            db.add(
                TrainingSample(
                    id=hold["id"],
                    gesture=hold["gesture"],
                    name=hold["name"],
                    gesture_type=str(hold["gesture_type"]),
                    source=hold["source"],
                    quality=hold["quality"],
                    split=hold["split"],
                    session_id=hold["session_id"],
                    handedness=hold["handedness"],
                    feature_version=int(hold["feature_version"]),
                    frame_count=int(hold["frame_count"]),
                    features=hold["features"],
                    sequence=hold["sequence"],
                    landmarks=hold["landmarks"],
                    created_at=created_at,
                )
            )
    except Exception as err:
        raise DatasetStoreError(
            "PostgreSQL did not save this sample. Check the database connection."
        ) from err
    return hold


def _iter_holds() -> list[dict[str, Any]]:
    try:
        with session_scope() as db:
            rows = db.scalars(
                select(TrainingSample)
                .where(TrainingSample.quality == "ok")
                .order_by(TrainingSample.created_at.asc())
            ).all()
            return [_row_to_hold(row) for row in rows]
    except Exception as err:
        raise DatasetStoreError(
            "PostgreSQL could not load training samples."
        ) from err


def list_gold_holds(
    *,
    gesture: str | None = None,
    name: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit), 1000))
    stmt = (
        select(TrainingSample)
        .where(TrainingSample.quality == "ok")
        .order_by(TrainingSample.created_at.desc())
        .limit(capped)
    )
    if gesture:
        stmt = stmt.where(TrainingSample.gesture == gesture.strip())
    if name:
        stmt = stmt.where(func.lower(TrainingSample.name) == name.strip().lower())
    try:
        with session_scope() as db:
            return [_row_to_hold(row) for row in db.scalars(stmt).all()]
    except Exception as err:
        raise DatasetStoreError(
            "PostgreSQL could not list training samples."
        ) from err


def collect_static_examples() -> list[dict[str, Any]]:
    """Flatten gold pose holds into labeled feature rows grouped by hold id."""
    examples: list[dict[str, Any]] = []
    for hold in _iter_holds():
        name = hold.get("gesture")
        if name not in STATIC_GESTURES:
            continue
        if int(hold.get("feature_version") or 0) != FEATURE_VERSION:
            continue
        rows = hold.get("features")
        if not isinstance(rows, list):
            continue
        hold_id = str(hold.get("id") or "")
        session_id = hold.get("session_id")
        for row in rows:
            if not isinstance(row, list) or len(row) != FEATURE_DIM:
                continue
            try:
                values = [float(item) for item in row]
            except (TypeError, ValueError):
                continue
            examples.append(
                {
                    "hold_id": hold_id,
                    "session_id": session_id,
                    "gesture": name,
                    "features": values,
                }
            )
    return examples


def train_readiness() -> dict[str, Any]:
    counts: dict[str, int] = {name: 0 for name in STATIC_GESTURES}
    for hold in _iter_holds():
        name = hold.get("gesture")
        if name in counts:
            counts[name] += 1
    included = {
        name: count
        for name, count in counts.items()
        if count >= MIN_TRAIN_HOLDS_PER_CLASS
    }
    blockers: list[str] = []
    if len(included) < MIN_TRAIN_CLASSES:
        blockers.append(
            "Need at least "
            f"{MIN_TRAIN_CLASSES} different signs with "
            f"{MIN_TRAIN_HOLDS_PER_CLASS}+ holds each. Currently ready: "
            f"{', '.join(included) or 'none'}."
        )
    short = [
        f"{name} ({counts[name]})"
        for name in STATIC_GESTURES
        if 0 < counts[name] < MIN_TRAIN_HOLDS_PER_CLASS
    ]
    if short:
        blockers.append(
            "Need more holds before training these signs: " + ", ".join(short)
        )
    warnings: list[str] = []
    thin = [name for name, count in included.items() if count < TRAINABLE_HOLDS]
    if thin:
        warnings.append(
            "Expect weak accuracy until ~50 holds per sign. Thin classes: "
            + ", ".join(thin)
        )
    can_train = len(included) >= MIN_TRAIN_CLASSES
    return {
        "can_train": can_train,
        "included_classes": included,
        "class_counts": counts,
        "blockers": blockers if not can_train else [],
        "warnings": warnings,
        "notes": blockers if can_train else [],
    }


def get_dataset_stats() -> dict[str, Any]:
    holds = _iter_holds()
    counts: dict[str, int] = {name: 0 for name in SIGN_GESTURES}
    for hold in holds:
        name = hold.get("gesture")
        if name in counts:
            counts[name] += 1

    last = latest_meta()
    per_class = {}
    if isinstance(last, dict):
        raw_per_class = last.get("per_class")
        if isinstance(raw_per_class, dict):
            per_class = raw_per_class

    live_path = _live_path()
    runtime = get_runtime()
    readiness = train_readiness()
    gesture_rows: list[dict[str, Any]] = []
    trainable_signs = 0
    promote_ready_signs = 0
    static_progress_sum = 0.0

    for name in SIGN_GESTURES:
        gold = counts[name]
        status = _status_for_holds(gold)
        if name != "wave":
            static_progress_sum += min(gold, PROMOTE_HOLDS)
            if status in ("trainable", "ready"):
                trainable_signs += 1
            if status == "ready":
                promote_ready_signs += 1
        kind = gesture_type(name)
        class_metrics = per_class.get(name)
        gesture_rows.append(
            {
                "gesture": name,
                "meaning": SIGN_MEANINGS[name],
                "type": kind,
                "gold_holds": gold,
                "status": status,
                "progress": round(min(gold / PROMOTE_HOLDS, 1.0), 4),
                "live_path": live_path,
                "metrics": class_metrics if isinstance(class_metrics, dict) else None,
            }
        )

    static_denom = max(len(STATIC_GESTURES) * PROMOTE_HOLDS, 1)
    gold_total = sum(counts.values())
    live_id = runtime.get("live_model_id")
    last_version = last.get("version") if isinstance(last, dict) else None
    can_promote = bool(last_version) and (
        runtime.get("source") != "model" or live_id != last_version
    )

    return {
        "source": runtime["source"],
        "model": {
            "available": bool(runtime.get("model_available")),
            "version": live_id,
        },
        "feature_version": FEATURE_VERSION,
        "trainable_threshold": TRAINABLE_HOLDS,
        "promote_threshold": PROMOTE_HOLDS,
        "min_train_holds": MIN_TRAIN_HOLDS_PER_CLASS,
        "static_sign_count": len(STATIC_GESTURES),
        "overall": {
            "gold_holds": gold_total,
            "trainable_signs": trainable_signs,
            "promote_ready_signs": promote_ready_signs,
            "progress": round(static_progress_sum / static_denom, 4),
        },
        "last_train": last,
        "can_train": bool(readiness["can_train"]),
        "can_promote": can_promote,
        "train_blockers": readiness["blockers"],
        "train_warnings": readiness["warnings"],
        "gestures": gesture_rows,
    }
