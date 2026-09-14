"""Turn a recorded landmark sequence into reference data for the scoring engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.scoring.engine import (
    hand_pose_features,
    hand_wave_features,
    is_hand_landmarks,
    joint_angle_deg,
    wrist_features_relative_to_waist,
)
from app.scoring.reference_gestures import gesture_type

# Frames captured while moving into/out of the pose are dropped before averaging.
TRANSITION_TRIM_SECONDS = 0.5
# Used when frames arrive without timestamps: drop this fraction from each end.
FALLBACK_TRIM_FRACTION = 0.15
MIN_FRAMES_AFTER_TRIM = 3
RECORDED_ANGLE_TOLERANCE_DEG = 30.0
RECORDED_HAND_FEATURE_TOLERANCE = 0.85
# Keep recorded motion references close to the live buffer length so DTW distances
# stay on the same scale the thresholds were tuned for.
MAX_MOTION_REFERENCE_FRAMES = 12
DTW_DISTANCE_PER_FRAME = 0.15


class CalibrationError(ValueError):
    """Raised when a submitted recording cannot produce a usable reference."""


def normalize_frames(
    raw_sequence: list[Any],
) -> list[tuple[float | None, dict[str, Any]]]:
    """Accept either flat landmark dicts or {timestamp, landmarks} wrappers."""
    frames: list[tuple[float | None, dict[str, Any]]] = []

    for entry in raw_sequence:
        if not isinstance(entry, dict):
            continue

        landmarks = entry.get("landmarks")
        if isinstance(landmarks, dict):
            payload = landmarks
        else:
            payload = {
                key: value for key, value in entry.items() if isinstance(value, dict)
            }

        if not payload:
            continue

        timestamp = entry.get("timestamp")
        frames.append(
            (float(timestamp) if isinstance(timestamp, (int, float)) else None, payload)
        )

    return frames


def trim_transition_frames(
    frames: list[tuple[float | None, dict[str, Any]]],
) -> list[tuple[float | None, dict[str, Any]]]:
    """Drop the first/last ~0.5s so the transition into the pose isn't averaged in."""
    if len(frames) <= MIN_FRAMES_AFTER_TRIM:
        return frames

    timestamps = [ts for ts, _ in frames if ts is not None]
    trimmed: list[tuple[float | None, dict[str, Any]]] = []

    if len(timestamps) == len(frames):
        start, end = timestamps[0], timestamps[-1]
        trim_ms = TRANSITION_TRIM_SECONDS * 1000.0

        if (end - start) > (2 * trim_ms):
            trimmed = [
                frame
                for frame in frames
                if frame[0] is not None
                and (start + trim_ms) <= frame[0] <= (end - trim_ms)
            ]
    else:
        cut = int(len(frames) * FALLBACK_TRIM_FRACTION)
        trimmed = frames[cut : len(frames) - cut] if cut else list(frames)

    if len(trimmed) >= MIN_FRAMES_AFTER_TRIM:
        return trimmed

    # Recording was too short to trim — keep everything rather than failing.
    return frames


def _downsample(sequence: list[list[float]], max_length: int) -> list[list[float]]:
    if len(sequence) <= max_length:
        return sequence

    indices = np.linspace(0, len(sequence) - 1, max_length)
    return [sequence[int(round(index))] for index in indices]


def build_pose_reference(
    frames: list[tuple[float | None, dict[str, Any]]],
    gesture: str = "exit_pointing",
) -> dict[str, Any]:
    """Build a held-pose reference — arm angles (aviation) or hand features (signs)."""
    sample = frames[0][1] if frames else {}
    if is_hand_landmarks(sample) or (
        gesture != "exit_pointing" and gesture_type(gesture) == "pose"
    ):
        return _build_hand_pose_reference(frames, gesture)
    return _build_arm_pose_reference(frames)


def _build_arm_pose_reference(
    frames: list[tuple[float | None, dict[str, Any]]],
) -> dict[str, Any]:
    """Average shoulder–elbow–wrist angles across the held pose."""
    left_angles: list[float] = []
    right_angles: list[float] = []

    for _timestamp, landmarks in frames:
        for side, bucket in (("left", left_angles), ("right", right_angles)):
            shoulder = landmarks.get(f"{side}Shoulder")
            elbow = landmarks.get(f"{side}Elbow")
            wrist = landmarks.get(f"{side}Wrist")

            if (
                isinstance(shoulder, dict)
                and isinstance(elbow, dict)
                and isinstance(wrist, dict)
            ):
                bucket.append(joint_angle_deg(shoulder, elbow, wrist))

    all_angles = left_angles + right_angles
    if len(all_angles) < MIN_FRAMES_AFTER_TRIM:
        raise CalibrationError(
            "Not enough frames with both arms visible to calibrate exit_pointing."
        )

    target_angle = float(np.mean(all_angles))

    return {
        "gesture_type": "pose",
        "target_elbow_angle_deg": round(target_angle, 2),
        "elbow_angle_tolerance_deg": RECORDED_ANGLE_TOLERANCE_DEG,
        "horizontal_y_tolerance": 0.10,
        "match_threshold": 0.55,
        "left_arm_angle_avg": (
            round(float(np.mean(left_angles)), 2) if left_angles else None
        ),
        "right_arm_angle_avg": (
            round(float(np.mean(right_angles)), 2) if right_angles else None
        ),
        "frames_used": len(frames),
    }


def _build_hand_pose_reference(
    frames: list[tuple[float | None, dict[str, Any]]],
    gesture: str,
) -> dict[str, Any]:
    """Average hand tip feature vectors for a held sign."""
    feature_rows: list[list[float]] = []

    for _timestamp, landmarks in frames:
        features = hand_pose_features(landmarks)
        if features is not None:
            feature_rows.append(features)

    if len(feature_rows) < MIN_FRAMES_AFTER_TRIM:
        raise CalibrationError(
            f"Not enough frames with a clear hand visible to calibrate {gesture}."
        )

    mean_features = np.mean(np.asarray(feature_rows, dtype=float), axis=0)

    return {
        "gesture_type": "pose",
        "reference_features": [round(float(v), 5) for v in mean_features.tolist()],
        "feature_tolerance": RECORDED_HAND_FEATURE_TOLERANCE,
        "frames_used": len(feature_rows),
    }


def build_motion_reference(
    frames: list[tuple[float | None, dict[str, Any]]],
    gesture: str = "seatbelt_demo",
) -> dict[str, Any]:
    """Store a DTW reference sequence — waist wrists or hand-wave trajectory."""
    sample = frames[0][1] if frames else {}
    if is_hand_landmarks(sample) or gesture == "wave":
        return _build_hand_wave_reference(frames, gesture)
    return _build_seatbelt_motion_reference(frames)


def _build_seatbelt_motion_reference(
    frames: list[tuple[float | None, dict[str, Any]]],
) -> dict[str, Any]:
    """Store wrist positions relative to the waist as the new DTW reference."""
    sequence: list[list[float]] = []

    for _timestamp, landmarks in frames:
        features = wrist_features_relative_to_waist(landmarks)
        if features is not None:
            sequence.append([round(float(value), 5) for value in features])

    if len(sequence) < MIN_FRAMES_AFTER_TRIM:
        raise CalibrationError(
            "Not enough frames with both wrists and hips visible to calibrate "
            "seatbelt_demo."
        )

    captured_frames = len(sequence)
    sequence = _downsample(sequence, MAX_MOTION_REFERENCE_FRAMES)

    return {
        "gesture_type": "motion",
        "feature_kind": "wrist_waist",
        "reference_sequence": sequence,
        "max_dtw_distance": round(
            max(1.0, DTW_DISTANCE_PER_FRAME * len(sequence)), 3
        ),
        "frames_used": captured_frames,
    }


def _build_hand_wave_reference(
    frames: list[tuple[float | None, dict[str, Any]]],
    gesture: str,
) -> dict[str, Any]:
    """Store wrist/index trajectory for wave DTW."""
    sequence: list[list[float]] = []

    for _timestamp, landmarks in frames:
        features = hand_wave_features(landmarks)
        if features is not None:
            sequence.append([round(float(value), 5) for value in features])

    if len(sequence) < MIN_FRAMES_AFTER_TRIM:
        raise CalibrationError(
            f"Not enough frames with a visible hand to calibrate {gesture}."
        )

    captured_frames = len(sequence)
    sequence = _downsample(sequence, MAX_MOTION_REFERENCE_FRAMES)

    return {
        "gesture_type": "motion",
        "feature_kind": "hand_wave",
        "reference_sequence": sequence,
        "max_dtw_distance": round(
            max(1.0, DTW_DISTANCE_PER_FRAME * len(sequence) * 1.2), 3
        ),
        "frames_used": captured_frames,
    }
