"""Gesture references for aviation and sign-language modes.

Recorded calibrations (from POST /reference/record) take priority; the hardcoded
defaults below are used only for gestures that have not been recorded yet.

Each gesture has a gesture_type of "pose" (held position) or "motion" (DTW sequence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from sqlalchemy import select

from app.db import session_scope
from app.models import RecordedReference
from app.scoring.sign_meanings import SIGN_GESTURES, SIGN_MEANINGS

GestureType = Literal["pose", "motion"]
GestureDomain = Literal["aviation", "sign_language"]

RECORDED_REFERENCES_PATH = Path(__file__).with_name("recorded_references.json")


class GestureSpec(TypedDict):
    name: str
    type: GestureType
    domain: GestureDomain


# Registry of every supported gesture key → type + domain.
GESTURE_REGISTRY: dict[str, GestureSpec] = {
    "exit_pointing": {
        "name": "exit_pointing",
        "type": "pose",
        "domain": "aviation",
    },
    "seatbelt_demo": {
        "name": "seatbelt_demo",
        "type": "motion",
        "domain": "aviation",
    },
    "thumbs_up": {"name": "thumbs_up", "type": "pose", "domain": "sign_language"},
    "thumbs_down": {"name": "thumbs_down", "type": "pose", "domain": "sign_language"},
    "open_palm": {"name": "open_palm", "type": "pose", "domain": "sign_language"},
    "fist": {"name": "fist", "type": "pose", "domain": "sign_language"},
    "four": {"name": "four", "type": "pose", "domain": "sign_language"},
    "pointing": {"name": "pointing", "type": "pose", "domain": "sign_language"},
    "peace_sign": {"name": "peace_sign", "type": "pose", "domain": "sign_language"},
    "please": {"name": "please", "type": "pose", "domain": "sign_language"},
    "you": {"name": "you", "type": "pose", "domain": "sign_language"},
    "want": {"name": "want", "type": "pose", "domain": "sign_language"},
    "okay": {"name": "okay", "type": "pose", "domain": "sign_language"},
    "i_love_you": {"name": "i_love_you", "type": "pose", "domain": "sign_language"},
    "wave": {"name": "wave", "type": "motion", "domain": "sign_language"},
}

AVIATION_GESTURES = tuple(
    name for name, spec in GESTURE_REGISTRY.items() if spec["domain"] == "aviation"
)
ALL_GESTURE_KEYS = tuple(GESTURE_REGISTRY.keys())


class PoseReference(TypedDict, total=False):
    name: str
    type: str
    domain: str
    # Aviation (arm angles)
    target_elbow_angle_deg: float
    elbow_angle_tolerance_deg: float
    horizontal_y_tolerance: float
    # Sign language (hand tip feature vector)
    reference_features: list[float] | None
    feature_tolerance: float
    match_threshold: float
    attempt_threshold: float


class MotionReference(TypedDict, total=False):
    name: str
    type: str
    domain: str
    # Feature rows — aviation: wrist-rel-waist; sign wave: wrist/tip trajectory
    reference_sequence: list[list[float]]
    max_dtw_distance: float
    match_threshold: float
    attempt_threshold: float
    feature_kind: str  # "wrist_waist" | "hand_wave"


# --- Aviation defaults -------------------------------------------------------

DEFAULT_EXIT_POINTING: PoseReference = {
    "name": "exit_pointing",
    "type": "pose",
    "domain": "aviation",
    # MediaPipe 3D elbow angles for outstretched arms typically sit ~155–175°,
    # not a perfect geometric 180°. Wide zone = natural human variation.
    "target_elbow_angle_deg": 170.0,
    "elbow_angle_tolerance_deg": 30.0,
    "horizontal_y_tolerance": 0.10,
    "match_threshold": 0.55,
    "attempt_threshold": 0.35,
}

DEFAULT_SEATBELT_DEMO: MotionReference = {
    "name": "seatbelt_demo",
    "type": "motion",
    "domain": "aviation",
    "feature_kind": "wrist_waist",
    "reference_sequence": [
        [-0.28, 0.02, 0.28, 0.02],
        [-0.24, 0.015, 0.24, 0.015],
        [-0.20, 0.01, 0.20, 0.01],
        [-0.16, 0.005, 0.16, 0.005],
        [-0.12, 0.0, 0.12, 0.0],
        [-0.08, 0.0, 0.08, 0.0],
        [-0.05, 0.0, 0.05, 0.0],
        [-0.03, 0.0, 0.03, 0.0],
        [-0.01, 0.0, 0.01, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    "max_dtw_distance": 2.0,
    "match_threshold": 0.55,
    "attempt_threshold": 0.35,
}


# --- Sign-language defaults (heuristic scoring until a recording exists) -----

_SIGN_POSE_BASE: PoseReference = {
    "type": "pose",
    "domain": "sign_language",
    "reference_features": None,
    "feature_tolerance": 0.85,
    "match_threshold": 0.55,
    "attempt_threshold": 0.35,
}

DEFAULT_SIGN_POSES: dict[str, PoseReference] = {
    name: {**_SIGN_POSE_BASE, "name": name}  # type: ignore[misc]
    for name in SIGN_GESTURES
    if GESTURE_REGISTRY[name]["type"] == "pose"
}

# Side-to-side wave: [wrist_x, wrist_y, index_rel_x, index_rel_y]
DEFAULT_WAVE: MotionReference = {
    "name": "wave",
    "type": "motion",
    "domain": "sign_language",
    "feature_kind": "hand_wave",
    "reference_sequence": [
        [0.35, 0.45, 0.02, -0.12],
        [0.42, 0.45, 0.02, -0.12],
        [0.50, 0.45, 0.02, -0.12],
        [0.58, 0.45, 0.02, -0.12],
        [0.65, 0.45, 0.02, -0.12],
        [0.58, 0.45, 0.02, -0.12],
        [0.50, 0.45, 0.02, -0.12],
        [0.42, 0.45, 0.02, -0.12],
        [0.35, 0.45, 0.02, -0.12],
        [0.42, 0.45, 0.02, -0.12],
    ],
    "max_dtw_distance": 1.8,
    "match_threshold": 0.70,
    "attempt_threshold": 0.45,
}


_DEFAULTS: dict[str, PoseReference | MotionReference] = {
    "exit_pointing": DEFAULT_EXIT_POINTING,
    "seatbelt_demo": DEFAULT_SEATBELT_DEMO,
    **DEFAULT_SIGN_POSES,
    "wave": DEFAULT_WAVE,
}

# Cache of recorded overlays loaded from Postgres for live scoring.
_recorded: dict[str, dict[str, Any]] = {}


def gesture_type(gesture: str) -> GestureType | None:
    spec = GESTURE_REGISTRY.get(gesture)
    return spec["type"] if spec else None


def gesture_domain(gesture: str) -> GestureDomain | None:
    spec = GESTURE_REGISTRY.get(gesture)
    return spec["domain"] if spec else None


def is_known_gesture(gesture: str) -> bool:
    return gesture in GESTURE_REGISTRY


def load_recorded_references() -> dict[str, dict[str, Any]]:
    """Read recorded references from Postgres into the in-memory cache."""
    global _recorded
    loaded: dict[str, dict[str, Any]] = {}
    try:
        with session_scope() as db:
            rows = db.scalars(select(RecordedReference)).all()
            for row in rows:
                if row.gesture not in GESTURE_REGISTRY or not isinstance(row.payload, dict):
                    continue
                loaded[row.gesture] = dict(row.payload)
    except Exception as err:
        print(f"[references] could not load from Postgres: {err}")
        loaded = {}
    _recorded = loaded
    print(f"[references] loaded recorded references: {sorted(_recorded)}")
    return _recorded


def save_recorded_reference(gesture: str, payload: dict[str, Any]) -> None:
    """Persist one recorded gesture reference in Postgres and refresh the cache."""
    if gesture not in GESTURE_REGISTRY:
        raise ValueError(f"Unknown gesture key: {gesture}")

    spec = GESTURE_REGISTRY[gesture]
    stored = {
        **payload,
        "gesture": gesture,
        "gesture_type": spec["type"],
        "domain": spec["domain"],
    }
    with session_scope() as db:
        row = db.get(RecordedReference, gesture)
        if row is None:
            db.add(
                RecordedReference(
                    gesture=gesture,
                    name=SIGN_MEANINGS.get(gesture, gesture),
                    gesture_type=spec["type"],
                    domain=spec["domain"],
                    payload=stored,
                )
            )
        else:
            row.name = SIGN_MEANINGS.get(gesture, gesture)
            row.gesture_type = spec["type"]
            row.domain = spec["domain"]
            row.payload = stored
    _recorded[gesture] = stored
    print(f"[references] saved recorded reference for {gesture}")


def clear_recorded_reference(gesture: str) -> bool:
    """Drop one recorded reference so the built-in default is used."""
    removed = False
    with session_scope() as db:
        row = db.get(RecordedReference, gesture)
        if row is not None:
            db.delete(row)
            removed = True
    if gesture in _recorded:
        del _recorded[gesture]
        removed = True
    if removed:
        print(f"[references] cleared recorded reference for {gesture}")
    return removed


def clear_all_recorded_references() -> int:
    """Drop every recorded reference."""
    count = len(_recorded)
    with session_scope() as db:
        rows = db.scalars(select(RecordedReference)).all()
        count = max(count, len(rows))
        for row in rows:
            db.delete(row)
    _recorded.clear()
    print(f"[references] cleared all recorded references ({count})")
    return count


def _merge_pose(default: PoseReference, recorded: dict[str, Any] | None) -> PoseReference:
    reference: PoseReference = dict(default)  # type: ignore[assignment]
    if not recorded:
        return reference

    for key in (
        "target_elbow_angle_deg",
        "elbow_angle_tolerance_deg",
        "horizontal_y_tolerance",
        "feature_tolerance",
        "match_threshold",
        "attempt_threshold",
    ):
        value = recorded.get(key)
        if isinstance(value, (int, float)):
            reference[key] = float(value)  # type: ignore[literal-required]

    features = recorded.get("reference_features")
    if isinstance(features, list) and len(features) >= 5:
        reference["reference_features"] = [float(v) for v in features]

    return reference


def _merge_motion(
    default: MotionReference, recorded: dict[str, Any] | None
) -> MotionReference:
    reference: MotionReference = dict(default)  # type: ignore[assignment]
    if not recorded:
        return reference

    sequence = recorded.get("reference_sequence")
    if isinstance(sequence, list) and len(sequence) >= 3:
        reference["reference_sequence"] = [
            [float(value) for value in row] for row in sequence
        ]

    max_distance = recorded.get("max_dtw_distance")
    if isinstance(max_distance, (int, float)):
        reference["max_dtw_distance"] = float(max_distance)

    for key in ("match_threshold", "attempt_threshold"):
        value = recorded.get(key)
        if isinstance(value, (int, float)):
            reference[key] = float(value)  # type: ignore[literal-required]

    kind = recorded.get("feature_kind")
    if isinstance(kind, str):
        reference["feature_kind"] = kind

    return reference


def get_exit_pointing() -> PoseReference:
    return _merge_pose(DEFAULT_EXIT_POINTING, _recorded.get("exit_pointing"))


def get_seatbelt_demo() -> MotionReference:
    return _merge_motion(DEFAULT_SEATBELT_DEMO, _recorded.get("seatbelt_demo"))


def get_reference(gesture: str) -> PoseReference | MotionReference:
    if gesture not in GESTURE_REGISTRY:
        raise KeyError(f"Unknown gesture: {gesture}")

    default = _DEFAULTS[gesture]
    recorded = _recorded.get(gesture)
    if GESTURE_REGISTRY[gesture]["type"] == "pose":
        return _merge_pose(default, recorded)  # type: ignore[arg-type]
    return _merge_motion(default, recorded)  # type: ignore[arg-type]


def reference_status() -> dict[str, str]:
    """Which gestures use a recorded reference vs the hardcoded default."""
    return {
        name: ("recorded" if _recorded.get(name) else "default")
        for name in GESTURE_REGISTRY
    }


def recorded_reference_details() -> dict[str, dict[str, Any]]:
    """Raw recorded payloads, for status/debugging responses."""
    return {name: dict(value) for name, value in _recorded.items()}


def active_references() -> dict[str, PoseReference | MotionReference]:
    return {name: get_reference(name) for name in GESTURE_REGISTRY}


# Backward-compatible alias used by older imports.
GestureName = Literal["exit_pointing", "seatbelt_demo"]
