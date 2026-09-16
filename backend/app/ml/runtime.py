"""Live recognizer source: heuristic rules or a promoted model (Postgres)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

from app.db import session_scope
from app.models import MlRuntime

SignSource = Literal["heuristic", "hybrid", "model"]

MODELS_DIR = Path(__file__).resolve().parent / "models"
RUNTIME_PATH = MODELS_DIR / "runtime.json"

LIVE_SOURCE_DEFAULT: SignSource = "heuristic"
MODEL_MATCH_THRESHOLD = 0.55
_RUNTIME_ROW_ID = 1

_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def _default_runtime() -> dict[str, Any]:
    return {
        "source": LIVE_SOURCE_DEFAULT,
        "override_gestures": [],
        "live_model_id": None,
        "model_available": False,
    }


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    if source not in ("heuristic", "hybrid", "model"):
        source = LIVE_SOURCE_DEFAULT
    live_model_id = payload.get("live_model_id")
    return {
        "source": source,
        "override_gestures": list(payload.get("override_gestures") or []),
        "live_model_id": live_model_id,
        "model_available": bool(live_model_id),
    }


def load_runtime() -> dict[str, Any]:
    """Load persisted Talk-path source from Postgres."""
    global _cache
    payload = _default_runtime()
    try:
        with session_scope() as db:
            row = db.get(MlRuntime, _RUNTIME_ROW_ID)
            if row is not None:
                payload = _normalize(
                    {
                        "source": row.source,
                        "override_gestures": row.override_gestures,
                        "live_model_id": row.live_model_id,
                    }
                )
    except Exception as err:
        print(f"[ml] runtime load failed, using defaults: {err}")
        payload = _default_runtime()
    with _lock:
        _cache = payload
        return dict(_cache)


def get_runtime() -> dict[str, Any]:
    if _cache is None:
        return load_runtime()
    with _lock:
        return dict(_cache)


def save_runtime(
    *,
    source: SignSource,
    live_model_id: str | None,
    override_gestures: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "source": source,
        "override_gestures": list(override_gestures or []),
        "live_model_id": live_model_id,
        "model_available": bool(live_model_id) and source in ("hybrid", "model"),
    }
    try:
        with session_scope() as db:
            row = db.get(MlRuntime, _RUNTIME_ROW_ID)
            if row is None:
                db.add(
                    MlRuntime(
                        id=_RUNTIME_ROW_ID,
                        source=source,
                        live_model_id=live_model_id,
                        override_gestures=payload["override_gestures"],
                    )
                )
            else:
                row.source = source
                row.live_model_id = live_model_id
                row.override_gestures = payload["override_gestures"]
    except Exception as err:
        print(f"[ml] runtime save failed: {err}")
        raise
    with _lock:
        global _cache
        _cache = dict(payload)
        return dict(_cache)


def ws_recognizer_fields(
    model: dict[str, Any] | None = None,
    recognizer: str | None = None,
) -> dict[str, Any]:
    runtime = get_runtime()
    stub = {
        "available": False,
        "version": None,
        "label": None,
        "prob": None,
        "agree": None,
    }
    if isinstance(model, dict):
        stub.update(model)
    return {
        "recognizer": recognizer or runtime["source"],
        "model": stub,
    }
