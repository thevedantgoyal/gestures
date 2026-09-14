"""Gesture scoring engine for held poses and short motion sequences.

Supports aviation (body pose) and sign_language (hand landmarks) modes.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastdtw import fastdtw

from app.scoring.reference_gestures import MotionReference, PoseReference
from app.scoring.sign_meanings import SIGN_MEANINGS

WRIST_VISIBILITY_THRESHOLD = 0.3  # raised arms often drop MediaPipe visibility
# HandLandmarker typically has no useful per-landmark visibility — use framing instead.
HAND_MIN_BBOX_AREA = 0.008  # normalized image area; too small ≈ hand cut off / far
HAND_EDGE_MARGIN = 0.02  # landmarks must stay inside [margin, 1-margin]
SIGN_RECOGNITION_THRESHOLD = 0.55  # speak / mark correct at or above this
SIGN_ATTEMPT_FEEDBACK_THRESHOLD = 0.35  # show live Incorrect / trying below match

SIGN_POSE_GESTURES = (
    "thumbs_up",
    "open_palm",
    "fist",
    "pointing",
    "peace_sign",
)

HAND_LANDMARK_KEYS = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

AviationClass = Literal["exit_pointing", "seatbelt_demo", "none"]
SignClass = Literal[
    "thumbs_up",
    "open_palm",
    "fist",
    "pointing",
    "peace_sign",
    "wave",
    "none",
]
GestureClass = AviationClass | SignClass

# Exit pointing — multi-signal attempt (normalized to shoulder width).
EXIT_HEIGHT_SLACK = 0.18  # wrist may sit slightly below shoulder line
EXIT_MIN_SPREAD_RATIO = 0.75  # wrist lateral / shoulder_width
EXIT_MIN_EXTENSION_RATIO = 0.85  # horizontal wrist–shoulder / upper-arm
EXIT_ATTEMPT_THRESHOLD = 0.50
EXIT_HOLD_THRESHOLD = 0.40  # keep classifying exit while holding (hysteresis)

# Seatbelt — waist zone relative to torso.
SEATBELT_MAX_WRIST_ABOVE_HIP = 0.16
SEATBELT_MAX_WRIST_BELOW_HIP = 0.22
SEATBELT_ATTEMPT_THRESHOLD = 0.45
SEATBELT_HOLD_THRESHOLD = 0.35
# Aviation idle gate: mean wrist displacement across recent frames.
AVIATION_IDLE_MOTION_THRESHOLD = 0.018
AVIATION_IDLE_MIN_FRAMES = 4
# Seatbelt must show wrists closing toward each other.
SEATBELT_MIN_CLOSING_DELTA = 0.04
# Hold last aviation gesture briefly to fight MediaPipe flicker.
AVIATION_HOLD_FRAMES = 4


# Finger extension: tip must be substantially farther from wrist than the MCP.
FINGER_EXTENDED_MCP_RATIO = 1.28
# Wave: min std-dev of wrist x across the live buffer (raised to avoid stealing holds).
WAVE_MIN_WRIST_X_STD = 0.055

HAND_TIP_KEYS = (
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)
HAND_MCP_KEYS = (
    "thumb_mcp",
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
)
HAND_PIP_KEYS = (
    "thumb_ip",  # thumb has IP instead of PIP
    "index_pip",
    "middle_pip",
    "ring_pip",
    "pinky_pip",
)


def _euclidean(a: Any, b: Any) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _point(landmarks: dict[str, Any], key: str) -> dict[str, float] | None:
    value = landmarks.get(key)
    if not value or not isinstance(value, dict):
        return None
    if "x" not in value or "y" not in value:
        return None
    return value


def _vec3(point: dict[str, float]) -> np.ndarray:
    return np.array(
        [float(point["x"]), float(point["y"]), float(point.get("z", 0.0))],
        dtype=float,
    )


def _dist2(a: dict[str, float], b: dict[str, float]) -> float:
    return float(
        np.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))
    )


def joint_angle_deg(
    a: dict[str, float],
    b: dict[str, float],
    c: dict[str, float],
) -> float:
    """Angle at point b formed by points a–b–c, in degrees."""
    ba = _vec3(a) - _vec3(b)
    bc = _vec3(c) - _vec3(b)
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos_angle = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def wrist_visibility(landmarks: dict[str, Any]) -> tuple[float, float]:
    left = _point(landmarks, "leftWrist")
    right = _point(landmarks, "rightWrist")
    left_vis = float(left.get("visibility", 0.0)) if left else 0.0
    right_vis = float(right.get("visibility", 0.0)) if right else 0.0
    return left_vis, right_vis


def has_low_wrist_visibility(landmarks: dict[str, Any]) -> bool:
    """True only when wrists are missing or both have very low visibility.

    Raised arms often report visibility < 0.6 even when x/y are valid — do not
    hard-block scoring solely on Pose visibility.
    """
    left = _point(landmarks, "leftWrist")
    right = _point(landmarks, "rightWrist")
    if not left or not right:
        return True
    if "x" not in left or "y" not in left or "x" not in right or "y" not in right:
        return True
    left_vis, right_vis = wrist_visibility(landmarks)
    # Block only if BOTH wrists are nearly invisible.
    return (
        left_vis < WRIST_VISIBILITY_THRESHOLD
        and right_vis < WRIST_VISIBILITY_THRESHOLD
    )


def is_hand_landmarks(landmarks: dict[str, Any]) -> bool:
    """True when the payload looks like MediaPipe hand landmarks (not body pose)."""
    return _point(landmarks, "wrist") is not None and _point(landmarks, "index_tip") is not None


def assess_hand_framing(landmarks: dict[str, Any]) -> dict[str, Any]:
    """Decide if a hand is fully in-frame using x/y bounds — not Pose-style visibility.

    MediaPipe HandLandmarker landmarks usually lack a meaningful `visibility`
    field (often missing or always 0). Instead we require all 21 points to have
    coordinates inside the normalized frame and a reasonable bounding-box size.
    """
    if not is_hand_landmarks(landmarks):
        return {
            "ok": False,
            "reason": "missing_hand_landmarks",
            "points_in_frame": 0,
            "points_total": len(HAND_LANDMARK_KEYS),
            "bbox_area": 0.0,
            "sample_visibility": None,
        }

    xs: list[float] = []
    ys: list[float] = []
    in_frame = 0
    sample_visibility = None

    for key in HAND_LANDMARK_KEYS:
        point = _point(landmarks, key)
        if not point:
            continue
        x = float(point["x"])
        y = float(point["y"])
        xs.append(x)
        ys.append(y)
        if sample_visibility is None and "visibility" in point:
            sample_visibility = point.get("visibility")
        if (
            HAND_EDGE_MARGIN <= x <= (1.0 - HAND_EDGE_MARGIN)
            and HAND_EDGE_MARGIN <= y <= (1.0 - HAND_EDGE_MARGIN)
        ):
            in_frame += 1

    if len(xs) < len(HAND_LANDMARK_KEYS):
        return {
            "ok": False,
            "reason": "incomplete_landmarks",
            "points_in_frame": in_frame,
            "points_total": len(HAND_LANDMARK_KEYS),
            "points_present": len(xs),
            "bbox_area": 0.0,
            "sample_visibility": sample_visibility,
        }

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
    all_in = in_frame >= len(HAND_LANDMARK_KEYS)
    size_ok = bbox_area >= HAND_MIN_BBOX_AREA
    ok = all_in and size_ok

    reason = "ok"
    if not all_in:
        reason = "landmarks_near_edge"
    elif not size_ok:
        reason = "bbox_too_small"

    return {
        "ok": ok,
        "reason": reason,
        "points_in_frame": in_frame,
        "points_total": len(HAND_LANDMARK_KEYS),
        "points_present": len(xs),
        "bbox_area": round(bbox_area, 5),
        "bbox": {
            "min_x": round(min_x, 4),
            "max_x": round(max_x, 4),
            "min_y": round(min_y, 4),
            "max_y": round(max_y, 4),
        },
        "sample_visibility": sample_visibility,
    }


def has_low_hand_visibility(landmarks: dict[str, Any]) -> bool:
    """Backward-compatible name — now uses framing, not Pose visibility."""
    return not bool(assess_hand_framing(landmarks).get("ok"))


def _finger_extended(landmarks: dict[str, Any], tip_key: str, pip_key: str, mcp_key: str) -> bool:
    """True when a finger is clearly outstretched from the palm."""
    wrist = _point(landmarks, "wrist")
    tip = _point(landmarks, tip_key)
    mcp = _point(landmarks, mcp_key)
    if not wrist or not tip or not mcp:
        return False
    # Curled fingertips sit near the MCP; extended tips sit well beyond it.
    return _dist2(wrist, tip) > _dist2(wrist, mcp) * FINGER_EXTENDED_MCP_RATIO


def _thumb_extended_up(landmarks: dict[str, Any]) -> bool:
    """Thumb tip clearly above the MCP (image y smaller) and away from the palm."""
    wrist = _point(landmarks, "wrist")
    tip = _point(landmarks, "thumb_tip")
    mcp = _point(landmarks, "thumb_mcp")
    if not wrist or not tip or not mcp:
        return False
    extended = _dist2(wrist, tip) > _dist2(wrist, mcp) * 1.15
    pointing_up = tip["y"] < mcp["y"] - 0.02
    return extended and pointing_up


def finger_extension_flags(landmarks: dict[str, Any]) -> dict[str, bool] | None:
    if not is_hand_landmarks(landmarks):
        return None
    return {
        "thumb_up": _thumb_extended_up(landmarks),
        "thumb": _finger_extended(landmarks, "thumb_tip", "thumb_ip", "thumb_mcp"),
        "index": _finger_extended(landmarks, "index_tip", "index_pip", "index_mcp"),
        "middle": _finger_extended(landmarks, "middle_tip", "middle_pip", "middle_mcp"),
        "ring": _finger_extended(landmarks, "ring_tip", "ring_pip", "ring_mcp"),
        "pinky": _finger_extended(landmarks, "pinky_tip", "pinky_pip", "pinky_mcp"),
    }


def hand_pose_features(landmarks: dict[str, Any]) -> list[float] | None:
    """Normalized tip positions + extension ratios relative to the wrist/palm."""
    wrist = _point(landmarks, "wrist")
    middle_mcp = _point(landmarks, "middle_mcp")
    if not wrist or not middle_mcp:
        return None

    scale = _dist2(wrist, middle_mcp) + 1e-6
    features: list[float] = []

    for tip_key in HAND_TIP_KEYS:
        tip = _point(landmarks, tip_key)
        if not tip:
            return None
        features.append((float(tip["x"]) - float(wrist["x"])) / scale)
        features.append((float(tip["y"]) - float(wrist["y"])) / scale)

    for tip_key, mcp_key in zip(HAND_TIP_KEYS, HAND_MCP_KEYS, strict=True):
        tip = _point(landmarks, tip_key)
        mcp = _point(landmarks, mcp_key)
        if not tip or not mcp:
            return None
        features.append(_dist2(wrist, tip) / (_dist2(wrist, mcp) + 1e-6))

    return features


def hand_wave_features(landmarks: dict[str, Any]) -> list[float] | None:
    """Per-frame features for wave DTW: wrist position + index tip relative to wrist."""
    wrist = _point(landmarks, "wrist")
    index_tip = _point(landmarks, "index_tip")
    if not wrist or not index_tip:
        return None
    return [
        float(wrist["x"]),
        float(wrist["y"]),
        float(index_tip["x"]) - float(wrist["x"]),
        float(index_tip["y"]) - float(wrist["y"]),
    ]


def _heuristic_sign_score(landmarks: dict[str, Any], gesture: str) -> float:
    """How cleanly the finger pattern matches a known sign (0–1)."""
    flags = finger_extension_flags(landmarks)
    if not flags:
        return 0.0

    index = flags["index"]
    middle = flags["middle"]
    ring = flags["ring"]
    pinky = flags["pinky"]
    thumb_up = flags["thumb_up"]
    others = (index, middle, ring, pinky)

    if gesture == "thumbs_up":
        curled = sum(1 for f in others if not f)
        return min(1.0, (0.55 if thumb_up else 0.0) + 0.1125 * curled)
    if gesture == "open_palm":
        extended = sum(1 for f in others if f)
        return min(1.0, 0.25 * extended + (0.1 if flags["thumb"] else 0.0))
    if gesture == "fist":
        curled = sum(1 for f in others if not f)
        thumb_ok = not thumb_up
        return min(1.0, 0.2 * curled + (0.2 if thumb_ok else 0.0))
    if gesture == "pointing":
        score = 0.0
        if index:
            score += 0.45
        score += 0.183 * sum(1 for f in (middle, ring, pinky) if not f)
        return min(1.0, score)
    if gesture == "peace_sign":
        score = 0.0
        if index:
            score += 0.3
        if middle:
            score += 0.3
        score += 0.2 * sum(1 for f in (ring, pinky) if not f)
        return min(1.0, score)
    return 0.0


def _aviation_torso_metrics(
    landmarks: dict[str, Any],
) -> dict[str, Any] | None:
    left_shoulder = _point(landmarks, "leftShoulder")
    right_shoulder = _point(landmarks, "rightShoulder")
    left_wrist = _point(landmarks, "leftWrist")
    right_wrist = _point(landmarks, "rightWrist")
    left_elbow = _point(landmarks, "leftElbow")
    right_elbow = _point(landmarks, "rightElbow")
    left_hip = _point(landmarks, "leftHip")
    right_hip = _point(landmarks, "rightHip")

    if not left_shoulder or not right_shoulder or not left_wrist or not right_wrist:
        return None

    shoulder_width = abs(float(right_shoulder["x"]) - float(left_shoulder["x"])) + 1e-6
    mid_shoulder_x = (float(left_shoulder["x"]) + float(right_shoulder["x"])) / 2.0
    mid_shoulder_y = (float(left_shoulder["y"]) + float(right_shoulder["y"])) / 2.0

    return {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "shoulder_width": shoulder_width,
        "mid_shoulder_x": mid_shoulder_x,
        "mid_shoulder_y": mid_shoulder_y,
    }


def _side_exit_signals(
    *,
    side: str,
    shoulder: dict[str, float],
    elbow: dict[str, float] | None,
    wrist: dict[str, float],
    mid_shoulder_x: float,
    shoulder_width: float,
) -> dict[str, float]:
    """Per-arm height / spread / extension scores in [0, 1]."""
    # Height: 1 when wrist at/above shoulder, soft falloff below.
    delta_y = float(wrist["y"]) - float(shoulder["y"])
    if delta_y <= 0:
        height = 1.0
    elif delta_y >= EXIT_HEIGHT_SLACK:
        height = 0.0
    else:
        height = 1.0 - (delta_y / EXIT_HEIGHT_SLACK)

    # Spread: lateral distance from mid-shoulder, normalized by shoulder width.
    if side == "left":
        lateral = mid_shoulder_x - float(wrist["x"])
    else:
        lateral = float(wrist["x"]) - mid_shoulder_x
    spread_ratio = max(0.0, lateral) / shoulder_width
    spread = min(1.0, spread_ratio / EXIT_MIN_SPREAD_RATIO)

    # Extension: horizontal reach (not vertical hang) vs upper-arm length.
    horizontal_reach = abs(float(wrist["x"]) - float(shoulder["x"]))
    if elbow:
        upper = _dist2(shoulder, elbow) + 1e-6
        extension_ratio = horizontal_reach / upper
    else:
        extension_ratio = horizontal_reach / shoulder_width
    extension = min(1.0, extension_ratio / EXIT_MIN_EXTENSION_RATIO)

    # Height is required for exit pointing — hanging arms must not score high.
    raw = (0.40 * height) + (0.35 * spread) + (0.25 * extension)
    score = raw * (0.25 + 0.75 * height)

    return {
        "height": height,
        "spread": spread,
        "extension": extension,
        "delta_y": delta_y,
        "spread_ratio": spread_ratio,
        "extension_ratio": extension_ratio,
        "score": score,
    }


def exit_pointing_attempt_score(
    landmarks: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Multi-signal confidence that the trainee is attempting exit pointing."""
    torso = _aviation_torso_metrics(landmarks)
    if not torso:
        return 0.0, {"reason": "missing_shoulders_or_wrists"}

    left = _side_exit_signals(
        side="left",
        shoulder=torso["left_shoulder"],
        elbow=torso["left_elbow"],
        wrist=torso["left_wrist"],
        mid_shoulder_x=torso["mid_shoulder_x"],
        shoulder_width=torso["shoulder_width"],
    )
    right = _side_exit_signals(
        side="right",
        shoulder=torso["right_shoulder"],
        elbow=torso["right_elbow"],
        wrist=torso["right_wrist"],
        mid_shoulder_x=torso["mid_shoulder_x"],
        shoulder_width=torso["shoulder_width"],
    )
    score = (left["score"] + right["score"]) / 2.0
    debug = {
        "shoulder_width": round(torso["shoulder_width"], 4),
        "left": {k: round(v, 4) for k, v in left.items()},
        "right": {k: round(v, 4) for k, v in right.items()},
        "exit_score": round(score, 4),
    }
    return score, debug


def seatbelt_attempt_score(
    landmarks: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Confidence that hands are in the waist seatbelt zone."""
    torso = _aviation_torso_metrics(landmarks)
    if not torso or not torso["left_hip"] or not torso["right_hip"]:
        return 0.0, {"reason": "missing_hips_or_wrists"}

    mid_hip_y = (
        float(torso["left_hip"]["y"]) + float(torso["right_hip"]["y"])
    ) / 2.0
    mid_shoulder_y = torso["mid_shoulder_y"]
    shoulder_width = torso["shoulder_width"]
    left_shoulder = torso["left_shoulder"]
    right_shoulder = torso["right_shoulder"]

    left_w = torso["left_wrist"]
    right_w = torso["right_wrist"]

    below_shoulders = (
        float(left_w["y"]) > mid_shoulder_y + 0.03
        and float(right_w["y"]) > mid_shoulder_y + 0.03
    )
    if not below_shoulders:
        return 0.0, {
            "reason": "wrists_not_below_shoulders",
            "left_y": round(float(left_w["y"]), 4),
            "right_y": round(float(right_w["y"]), 4),
            "mid_shoulder_y": round(mid_shoulder_y, 4),
        }

    # Arms hanging at sides (wrists under shoulders) are not a seatbelt attempt.
    left_hang = abs(float(left_w["x"]) - float(left_shoulder["x"])) / shoulder_width
    right_hang = abs(float(right_w["x"]) - float(right_shoulder["x"])) / shoulder_width
    if left_hang < 0.35 and right_hang < 0.35:
        return 0.0, {
            "reason": "arms_hanging_at_sides",
            "left_hang": round(left_hang, 4),
            "right_hang": round(right_hang, 4),
        }

    def _waist_score(wrist: dict[str, float]) -> float:
        dy = float(wrist["y"]) - mid_hip_y
        if -SEATBELT_MAX_WRIST_ABOVE_HIP <= dy <= SEATBELT_MAX_WRIST_BELOW_HIP:
            return 1.0
        # Soft falloff outside the band.
        if dy < -SEATBELT_MAX_WRIST_ABOVE_HIP:
            over = -SEATBELT_MAX_WRIST_ABOVE_HIP - dy
        else:
            over = dy - SEATBELT_MAX_WRIST_BELOW_HIP
        return max(0.0, 1.0 - over / 0.12)

    left_waist = _waist_score(left_w)
    right_waist = _waist_score(right_w)
    separation = abs(float(right_w["x"]) - float(left_w["x"]))
    sep_ratio = separation / shoulder_width
    # Prefer a clear closing path: either fairly wide or already close.
    path_bias = 1.0
    if 0.9 < sep_ratio < 1.15:
        path_bias = 0.55

    score = 0.45 * left_waist + 0.45 * right_waist + 0.10 * path_bias
    debug = {
        "left_waist": round(left_waist, 4),
        "right_waist": round(right_waist, 4),
        "wrist_separation": round(separation, 4),
        "left_hang": round(left_hang, 4),
        "right_hang": round(right_hang, 4),
        "seatbelt_score": round(score, 4),
    }
    return score, debug


def classify_aviation_attempt(
    landmarks: dict[str, Any],
    *,
    previous: AviationClass | None = None,
) -> tuple[AviationClass, dict[str, Any]]:
    """Classify aviation attempt with multi-signal scores + hold hysteresis."""
    exit_score, exit_debug = exit_pointing_attempt_score(landmarks)
    seat_score, seat_debug = seatbelt_attempt_score(landmarks)

    exit_thresh = (
        EXIT_HOLD_THRESHOLD if previous == "exit_pointing" else EXIT_ATTEMPT_THRESHOLD
    )
    seat_thresh = (
        SEATBELT_HOLD_THRESHOLD
        if previous == "seatbelt_demo"
        else SEATBELT_ATTEMPT_THRESHOLD
    )

    debug = {
        "exit": exit_debug,
        "seatbelt": seat_debug,
        "exit_thresh": exit_thresh,
        "seatbelt_thresh": seat_thresh,
        "previous": previous,
    }

    # Prefer the stronger signal when both fire.
    if exit_score >= exit_thresh and exit_score >= seat_score:
        debug["chosen"] = "exit_pointing"
        return "exit_pointing", debug
    if seat_score >= seat_thresh:
        debug["chosen"] = "seatbelt_demo"
        return "seatbelt_demo", debug

    debug["chosen"] = "none"
    return "none", debug


def classify_aviation_attempt_legacy(landmarks: dict[str, Any]) -> AviationClass:
    """Backward-compatible wrapper returning only the class label."""
    label, _debug = classify_aviation_attempt(landmarks)
    return label


def _buffer_looks_like_wave(buffer: list[dict[str, Any]]) -> bool:
    xs: list[float] = []
    for frame in buffer:
        wrist = _point(frame, "wrist")
        if wrist:
            xs.append(float(wrist["x"]))
    if len(xs) < 5:
        return False
    return float(np.std(xs)) >= WAVE_MIN_WRIST_X_STD


def classify_sign_attempt(
    landmarks: dict[str, Any],
    buffer: list[dict[str, Any]] | None = None,
) -> SignClass:
    """Classify a hand landmark frame into one of the 6 signs (or none)."""
    flags = finger_extension_flags(landmarks)
    if not flags:
        return "none"

    index = flags["index"]
    middle = flags["middle"]
    ring = flags["ring"]
    pinky = flags["pinky"]
    thumb_up = flags["thumb_up"]

    # Held shapes first — wrist jitter must not steal open-palm / thumbs-up.
    if thumb_up and not index and not middle and not ring and not pinky:
        return "thumbs_up"

    if index and middle and not ring and not pinky:
        return "peace_sign"

    if index and not middle and not ring and not pinky:
        return "pointing"

    if index and middle and ring and pinky:
        return "open_palm"

    if not index and not middle and not ring and not pinky:
        return "fist"

    if buffer and _buffer_looks_like_wave(buffer):
        return "wave"

    return "none"


def classify_gesture_attempt(
    landmarks: dict[str, Any],
    mode: str = "aviation",
    buffer: list[dict[str, Any]] | None = None,
) -> GestureClass:
    """Cheap heuristic: what gesture is being attempted, if any.

    Branches by mode — aviation uses arm/shoulder geometry; sign_language uses
    finger tip / joint positions on the primary hand.
    """
    if mode == "sign_language":
        return classify_sign_attempt(landmarks, buffer)
    label, _debug = classify_aviation_attempt(landmarks)
    return label


def aviation_motion_energy(buffer: list[dict[str, Any]]) -> float:
    """Mean wrist displacement between consecutive buffered frames."""
    if len(buffer) < 2:
        return 0.0
    deltas: list[float] = []
    prev_l = _point(buffer[0], "leftWrist")
    prev_r = _point(buffer[0], "rightWrist")
    for frame in buffer[1:]:
        left = _point(frame, "leftWrist")
        right = _point(frame, "rightWrist")
        if prev_l and left:
            deltas.append(_dist2(prev_l, left))
        if prev_r and right:
            deltas.append(_dist2(prev_r, right))
        prev_l, prev_r = left, right
    if not deltas:
        return 0.0
    return float(np.mean(deltas))


def is_aviation_idle(buffer: list[dict[str, Any]]) -> bool:
    """True when the trainee is standing still (no meaningful arm motion)."""
    if len(buffer) < AVIATION_IDLE_MIN_FRAMES:
        return False
    return aviation_motion_energy(buffer) < AVIATION_IDLE_MOTION_THRESHOLD


def seatbelt_closing_delta(landmark_sequence: list[dict[str, Any]]) -> float | None:
    """How much wrist separation decreased over the buffer (positive = closing)."""
    separations: list[float] = []
    for frame in landmark_sequence:
        left = _point(frame, "leftWrist")
        right = _point(frame, "rightWrist")
        if left and right:
            separations.append(abs(float(right["x"]) - float(left["x"])))
    if len(separations) < 3:
        return None
    early = float(np.mean(separations[: max(1, len(separations) // 3)]))
    late = float(np.mean(separations[-max(1, len(separations) // 3) :]))
    return early - late


def _horizontal_alignment_score(
    landmarks: dict[str, Any],
    y_tol: float,
) -> tuple[float, list[dict[str, Any]], list[str]]:
    """Score how level wrists are with the shoulder line; return hints."""
    deviations: list[dict[str, Any]] = []
    hints: list[str] = []
    scores: list[float] = []

    for side in ("left", "right"):
        shoulder = _point(landmarks, f"{side}Shoulder")
        wrist = _point(landmarks, f"{side}Wrist")
        if not shoulder or not wrist:
            continue
        # y grows downward — positive delta means wrist below shoulder.
        delta_y = float(wrist["y"]) - float(shoulder["y"])
        score = max(0.0, 1.0 - (abs(delta_y) / max(y_tol, 1e-6)))
        scores.append(score)
        deviations.append(
            {
                "joint": f"{side}_wrist_height",
                "expected": "wrist near shoulder height",
                "actual": round(delta_y, 4),
                "error": round(abs(delta_y), 4),
                "tolerance": y_tol,
                "arm_score": round(score, 4),
            }
        )
        if delta_y > y_tol:
            hints.append(f"Raise your {side} arm toward shoulder height")
        elif delta_y < -y_tol:
            hints.append(f"Lower your {side} arm slightly toward the shoulder line")

    if not scores:
        return 0.0, deviations, hints
    return float(np.mean(scores)), deviations, hints


def score_pose(
    landmarks: dict[str, Any],
    reference: PoseReference,
) -> dict[str, Any]:
    """Score exit_pointing from elbow angles + horizontal wrist alignment."""
    deviations: list[dict[str, Any]] = []
    hints: list[str] = []

    target = float(reference["target_elbow_angle_deg"])
    angle_tol = float(reference["elbow_angle_tolerance_deg"])
    match_threshold = float(reference["match_threshold"])
    y_tol = float(reference.get("horizontal_y_tolerance", 0.10))

    left_arm_angle: float | None = None
    right_arm_angle: float | None = None
    left_arm_score = 0.0
    right_arm_score = 0.0
    arms_scored = 0

    for side in ("left", "right"):
        shoulder = _point(landmarks, f"{side}Shoulder")
        elbow = _point(landmarks, f"{side}Elbow")
        wrist = _point(landmarks, f"{side}Wrist")

        if not shoulder or not elbow or not wrist:
            deviations.append(
                {
                    "joint": f"{side}_arm",
                    "issue": "missing_landmarks",
                    "expected": "shoulder, elbow, wrist",
                    "actual": None,
                }
            )
            hints.append(f"Keep your {side} arm fully visible")
            continue

        angle = joint_angle_deg(shoulder, elbow, wrist)
        deviation = abs(angle - target)
        arm_score = max(0.0, 1.0 - (deviation / angle_tol))
        arms_scored += 1

        if side == "left":
            left_arm_angle = round(angle, 2)
            left_arm_score = arm_score
        else:
            right_arm_angle = round(angle, 2)
            right_arm_score = arm_score

        deviations.append(
            {
                "joint": f"{side}_elbow_angle",
                "expected": target,
                "actual": round(angle, 2),
                "error": round(deviation, 2),
                "tolerance": angle_tol,
                "arm_score": round(arm_score, 4),
            }
        )
        if angle < target - angle_tol * 0.45:
            hints.append(f"Straighten your {side} arm more")
        elif angle > target + angle_tol * 0.55:
            hints.append(f"Relax your {side} elbow slightly")

    angle_score = (
        (left_arm_score + right_arm_score) / 2.0 if arms_scored else 0.0
    )
    horiz_score, horiz_devs, horiz_hints = _horizontal_alignment_score(
        landmarks, y_tol
    )
    deviations.extend(horiz_devs)
    hints.extend(horiz_hints)

    # Elbow shape carries more weight; horizontal line still matters.
    final_score = 0.7 * angle_score + 0.3 * horiz_score
    matched = final_score >= match_threshold
    # Pass band shown in UI = how far from target still scores >= match_threshold
    # for the elbow term alone: (1 - match) * tol
    pass_band_deg = round((1.0 - match_threshold) * angle_tol / 0.7, 1)

    print(
        f"[score_pose] L_angle={left_arm_angle} R_angle={right_arm_angle} "
        f"target={target} tol={angle_tol} "
        f"angle_score={angle_score:.4f} horiz={horiz_score:.4f} "
        f"final={final_score:.4f} matched={matched}"
    )

    return {
        "matched": matched,
        "correct": matched,
        "score": round(final_score, 4),
        "deviations": deviations,
        "hints": hints[:3],
        "left_arm_angle": left_arm_angle,
        "right_arm_angle": right_arm_angle,
        "left_arm_score": round(left_arm_score, 4),
        "right_arm_score": round(right_arm_score, 4),
        "horizontal_score": round(horiz_score, 4),
        "target_angle": target,
        "tolerance": angle_tol,
        "pass_band_deg": pass_band_deg,
    }



def _feature_match_score(
    live: list[float],
    ref_features: list[float],
    tolerance: float,
) -> tuple[float, dict[str, Any]]:
    """Combine L1 and cosine so modest scale/angle drift still scores well."""
    live_a = np.asarray(live, dtype=float)
    ref_a = np.asarray(ref_features, dtype=float)
    mean_err = float(np.mean(np.abs(live_a - ref_a)))
    l1_score = max(0.0, 1.0 - (mean_err / max(tolerance, 1e-6)))

    denom = float(np.linalg.norm(live_a) * np.linalg.norm(ref_a)) + 1e-8
    cosine = float(np.dot(live_a, ref_a) / denom)
    # Map cosine [-1, 1] → [0, 1]
    cosine_score = max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    score = max(l1_score, cosine_score)
    detail = {
        "joint": "hand_features",
        "expected": "feature match within tolerance / cosine",
        "actual": round(mean_err, 4),
        "error": round(mean_err, 4),
        "tolerance": tolerance,
        "l1_score": round(l1_score, 4),
        "cosine": round(cosine, 4),
        "cosine_score": round(cosine_score, 4),
        "feature_dims": len(live),
    }
    return score, detail


def score_hand_pose(
    landmarks: dict[str, Any],
    reference: PoseReference,
    gesture: str,
    *,
    recognition_threshold: float | None = None,
) -> dict[str, Any]:
    """Score a held hand shape — hybrid of recorded features + finger heuristics.

    Recorded templates alone are brittle (distance/angle). Heuristics alone are
    noisy. Taking the max keeps a clear open-palm from scoring 0% against a
    slightly different recording.
    """
    threshold = (
        float(recognition_threshold)
        if recognition_threshold is not None
        else float(reference.get("match_threshold", SIGN_RECOGNITION_THRESHOLD))
    )
    # Looser than calibration default so re-tests after record still match.
    tolerance = float(reference.get("feature_tolerance", 0.85))
    if tolerance < 0.85:
        tolerance = 0.85
    ref_features = reference.get("reference_features")
    deviations: list[dict[str, Any]] = []

    live = hand_pose_features(landmarks)
    if live is None:
        return {
            "matched": False,
            "correct": False,
            "score": 0.0,
            "deviations": [
                {
                    "joint": "hand",
                    "issue": "missing_landmarks",
                    "expected": "full hand landmark set",
                    "actual": None,
                }
            ],
        }

    heuristic = _heuristic_sign_score(landmarks, gesture)
    feature_score: float | None = None

    if isinstance(ref_features, list) and len(ref_features) == len(live):
        feature_score, feature_detail = _feature_match_score(
            live, ref_features, tolerance
        )
        deviations.append(feature_detail)
        score = max(float(feature_score), heuristic)
        deviations.append(
            {
                "joint": "hand_hybrid",
                "feature_score": round(float(feature_score), 4),
                "heuristic_score": round(heuristic, 4),
                "combined": round(score, 4),
            }
        )
    else:
        score = heuristic
        deviations.append(
            {
                "joint": "hand_heuristic",
                "expected": f"clear {gesture} finger pattern",
                "actual": round(score, 4),
                "note": "no reference_features of matching length",
                "ref_len": len(ref_features) if isinstance(ref_features, list) else None,
                "live_len": len(live),
            }
        )

    matched = score >= threshold
    return {
        "matched": matched,
        "correct": matched,
        "score": round(score, 4),
        "deviations": deviations,
    }


def score_all_sign_poses(landmarks: dict[str, Any]) -> dict[str, float]:
    """Compute match scores against every held-pose sign reference.

    Returns scores for all signs. Gestures with saved ``reference_features`` use
    feature matching; others fall back to finger-pattern heuristics.
    """
    from app.scoring.reference_gestures import get_reference

    scores: dict[str, float] = {}
    for gesture in SIGN_POSE_GESTURES:
        result = score_hand_pose(
            landmarks,
            get_reference(gesture),
            gesture,
            recognition_threshold=SIGN_RECOGNITION_THRESHOLD,
        )
        scores[gesture] = float(result.get("score") or 0.0)
    return scores


def best_sign_pose(
    landmarks: dict[str, Any],
    *,
    classified_attempt: str | None = None,
) -> tuple[str | None, float, dict[str, float], dict[str, Any]]:
    """Pick the best pose sign from hybrid scores (features ∪ heuristics).

    When the finger classifier names an attempt, that sign gets a small boost so
    a clear open-palm isn't beaten by a partial peace/pointing heuristic.
    """
    from app.scoring.reference_gestures import get_reference

    scores = score_all_sign_poses(landmarks)
    if (
        classified_attempt in SIGN_POSE_GESTURES
        and classified_attempt in scores
    ):
        # Prefer the classified shape when fingers already match it.
        scores[classified_attempt] = min(
            1.0, float(scores[classified_attempt]) + 0.08
        )

    chosen = max(scores, key=scores.get) if scores else None
    chosen_score = float(scores.get(chosen, 0.0)) if chosen else 0.0

    if chosen is None:
        return None, 0.0, scores, {}

    detail = score_hand_pose(
        landmarks,
        get_reference(chosen),
        chosen,
        recognition_threshold=SIGN_RECOGNITION_THRESHOLD,
    )
    # Keep the (possibly boosted) chosen score for thresholding / UI.
    detail = {**detail, "score": round(chosen_score, 4)}
    return chosen, chosen_score, scores, detail


def wrist_features_relative_to_waist(landmarks: dict[str, Any]) -> list[float] | None:
    """Return [lx, ly, rx, ry] of wrists relative to mid-hip, or None if incomplete."""
    left_hip = _point(landmarks, "leftHip")
    right_hip = _point(landmarks, "rightHip")
    left_wrist = _point(landmarks, "leftWrist")
    right_wrist = _point(landmarks, "rightWrist")

    if not left_hip or not right_hip or not left_wrist or not right_wrist:
        return None

    mid_x = (left_hip["x"] + right_hip["x"]) / 2.0
    mid_y = (left_hip["y"] + right_hip["y"]) / 2.0

    return [
        left_wrist["x"] - mid_x,
        left_wrist["y"] - mid_y,
        right_wrist["x"] - mid_x,
        right_wrist["y"] - mid_y,
    ]


def score_motion(
    landmark_sequence: list[dict[str, Any]],
    reference: MotionReference,
) -> dict[str, Any]:
    """Compare a buffered feature sequence to a motion reference via FastDTW."""
    deviations: list[dict[str, Any]] = []
    hints: list[str] = []
    min_frames = max(3, len(reference["reference_sequence"]) // 2)
    match_threshold = float(reference["match_threshold"])
    feature_kind = reference.get("feature_kind", "wrist_waist")

    feature_fn = (
        hand_wave_features if feature_kind == "hand_wave" else wrist_features_relative_to_waist
    )

    live_series: list[list[float]] = []
    for frame in landmark_sequence:
        features = feature_fn(frame)
        if features is not None:
            live_series.append(features)

    ref_series = [list(map(float, row)) for row in reference["reference_sequence"]]

    if len(live_series) < min_frames:
        progress = (
            float(len(live_series)) / float(min_frames) if min_frames > 0 else 0.0
        )
        return {
            "matched": False,
            "correct": False,
            "score": 0.0,
            "ready": False,
            "buffer_frames": len(live_series),
            "buffer_needed": min_frames,
            "buffer_progress": round(min(1.0, progress), 3),
            "deviations": [
                {
                    "joint": "motion_buffer",
                    "issue": "insufficient_frames",
                    "expected": f">= {min_frames} valid frames",
                    "actual": len(live_series),
                }
            ],
            "hints": ["Bring both hands toward your waist and slide them together"],
        }

    closing = None
    if feature_kind == "wrist_waist":
        closing = seatbelt_closing_delta(landmark_sequence)
        if closing is not None and closing < SEATBELT_MIN_CLOSING_DELTA:
            deviations.append(
                {
                    "joint": "seatbelt_closing",
                    "issue": "no_closing_motion",
                    "expected": f"wrist separation decrease >= {SEATBELT_MIN_CLOSING_DELTA}",
                    "actual": round(float(closing), 4),
                }
            )
            return {
                "matched": False,
                "correct": False,
                "score": round(
                    max(
                        0.0,
                        min(0.45, float(closing) / SEATBELT_MIN_CLOSING_DELTA * 0.45),
                    ),
                    4,
                ),
                "ready": True,
                "closing_delta": round(float(closing), 4),
                "deviations": deviations,
                "hints": [
                    "Slide both hands together across your waist — keep moving until they meet"
                ],
            }

    live = np.asarray(live_series, dtype=float)
    ref = np.asarray(ref_series, dtype=float)

    distance, _path = fastdtw(live, ref, dist=_euclidean)
    max_distance = reference["max_dtw_distance"]
    score = max(0.0, min(1.0, 1.0 - (float(distance) / max_distance)))

    deviations.append(
        {
            "joint": "motion_dtw",
            "expected": f"DTW distance < {max_distance}",
            "actual": round(float(distance), 4),
            "error": round(float(distance), 4),
            "tolerance": max_distance,
            "feature_kind": feature_kind,
        }
    )
    if closing is not None:
        deviations.append(
            {
                "joint": "seatbelt_closing",
                "expected": f">= {SEATBELT_MIN_CLOSING_DELTA}",
                "actual": round(float(closing), 4),
            }
        )

    matched = score >= match_threshold
    if not matched:
        hints.append("Keep both hands near your waist and close them more smoothly")

    return {
        "matched": matched,
        "correct": matched,
        "score": round(score, 4),
        "ready": True,
        "dtw_distance": round(float(distance), 4),
        "closing_delta": round(float(closing), 4) if closing is not None else None,
        "deviations": deviations,
        "hints": hints,
    }



def meaning_for(gesture: str) -> str | None:
    return SIGN_MEANINGS.get(gesture)
