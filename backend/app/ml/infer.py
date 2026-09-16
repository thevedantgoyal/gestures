"""Load the promoted sklearn model and score a live hand frame."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.ml.registry import RegistryError, latest_meta, load_bundle
from app.ml.runtime import (
    MODEL_MATCH_THRESHOLD,
    get_runtime,
    save_runtime,
)
from app.scoring.engine import hand_pose_features
from app.scoring.sign_meanings import SIGN_MEANINGS

_lock = threading.Lock()
_classifier: Any = None
_meta: dict[str, Any] | None = None
_loaded_version: str | None = None


def _empty_model_fields() -> dict[str, Any]:
    runtime = get_runtime()
    return {
        "available": False,
        "version": runtime.get("live_model_id"),
        "label": None,
        "prob": None,
        "agree": None,
    }


def _load_version(version: str) -> dict[str, Any] | None:
    try:
        classifier, meta = load_bundle(version)
    except RegistryError as err:
        print(f"[ml] could not load model {version}: {err}")
        with _lock:
            global _classifier, _meta, _loaded_version
            _classifier = None
            _meta = None
            _loaded_version = None
        return None
    with _lock:
        _classifier = classifier
        _meta = meta
        _loaded_version = version
    print(f"[ml] model loaded {version} classes={meta.get('classes')}")
    return meta


def load_live_model() -> dict[str, Any] | None:
    """Load the promoted bundle, or the latest trained model for live testing."""
    runtime = get_runtime()
    version = runtime.get("live_model_id")
    if runtime.get("source") not in ("hybrid", "model") or not version:
        latest = latest_meta()
        version = str(latest["version"]) if latest else None
        if not version:
            with _lock:
                global _classifier, _meta, _loaded_version
                _classifier = None
                _meta = None
                _loaded_version = None
            return None
    return _load_version(str(version))


def set_shadow_model(version: str) -> dict[str, Any] | None:
    """Keep the just-trained model in memory for live accuracy, without promoting."""
    return _load_version(version)


def promote_model(version: str) -> dict[str, Any]:
    classifier, meta = load_bundle(version)
    save_runtime(source="model", live_model_id=version)
    with _lock:
        global _classifier, _meta, _loaded_version
        _classifier = classifier
        _meta = meta
        _loaded_version = version
    print(f"[ml] promoted {version} — Talk now uses the trained model")
    return get_runtime()


def demote_to_rules() -> dict[str, Any]:
    runtime = get_runtime()
    save_runtime(
        source="heuristic",
        live_model_id=runtime.get("live_model_id"),
    )
    print("[ml] demoted — Talk uses hardcoded rules again")
    return get_runtime()


def predict_sign(landmarks: dict[str, Any]) -> dict[str, Any]:
    fields = _empty_model_fields()
    with _lock:
        classifier = _classifier
        meta = _meta
        version = _loaded_version
    if classifier is None or not meta:
        return fields

    vector = hand_pose_features(landmarks)
    if vector is None:
        fields["available"] = True
        fields["version"] = version
        return fields

    expected = int(meta.get("feature_dim") or 0)
    if expected and len(vector) != expected:
        fields["available"] = True
        fields["version"] = version
        return fields

    try:
        x = np.asarray([vector], dtype=float)
        label = str(classifier.predict(x)[0])
        prob = None
        if hasattr(classifier, "predict_proba"):
            classes = list(classifier.classes_)
            probs = classifier.predict_proba(x)[0]
            if label in classes:
                prob = float(probs[classes.index(label)])
            else:
                prob = float(max(probs))
    except Exception as err:  # noqa: BLE001
        print(f"[ml] predict failed: {err}")
        fields["available"] = True
        fields["version"] = version
        return fields

    fields["available"] = True
    fields["version"] = version
    fields["label"] = label
    fields["prob"] = round(prob, 4) if prob is not None else None
    return fields


def apply_model_to_sign_result(
    heuristic: dict[str, Any],
    landmarks: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Optionally replace the heuristic label with the promoted model.

    Wave stays on rules. If the model is live but unsure, rules fill in.
    """
    heuristic_label = str(heuristic.get("gesture") or "none")
    prediction = predict_sign(landmarks)
    prediction["agree"] = (
        prediction.get("label") == heuristic_label
        if prediction.get("label")
        else None
    )
    runtime = get_runtime()
    recognizer = "heuristic"

    if heuristic_label == "wave":
        return heuristic, prediction, recognizer

    if runtime.get("source") != "model" or not prediction.get("available"):
        return heuristic, prediction, recognizer

    label = prediction.get("label")
    prob = prediction.get("prob")
    if (
        isinstance(label, str)
        and label in SIGN_MEANINGS
        and label != "wave"
        and isinstance(prob, (int, float))
        and float(prob) >= MODEL_MATCH_THRESHOLD
    ):
        result = dict(heuristic)
        result["gesture"] = label
        result["score"] = float(prob)
        result["correct"] = True
        result["meaning"] = SIGN_MEANINGS[label]
        result["reason"] = None
        result["classified_as"] = label
        result["attempted_gesture"] = label
        return result, prediction, "model"

    return heuristic, prediction, "model_fallback"
