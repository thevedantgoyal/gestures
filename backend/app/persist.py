"""One-time import of local JSON files into PostgreSQL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db import session_scope
from app.ml.dataset import SAMPLES_PATH, hold_from_record
from app.ml.numpy_clf import SoftmaxClassifier
from app.ml.registry import _VERSION_RE
from app.ml.runtime import MODELS_DIR, RUNTIME_PATH
from app.models import MlRuntime, RecordedReference, TrainedModel, TrainingSample
from app.scoring.reference_gestures import (
    GESTURE_REGISTRY,
    RECORDED_REFERENCES_PATH,
)
from app.scoring.sign_meanings import SIGN_MEANINGS


def _parse_created(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    return datetime.now(timezone.utc)


def _archive(path: Path) -> None:
    if not path.exists():
        return
    dest = path.with_name(path.name + ".imported")
    if dest.exists():
        dest = path.with_name(f"{path.name}.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.imported")
    path.replace(dest)
    print(f"[db] archived {path.name} -> {dest.name}")


def migrate_training_samples() -> int:
    if not SAMPLES_PATH.exists():
        return 0
    raw = SAMPLES_PATH.read_text(encoding="utf-8")
    added = 0
    with session_scope() as db:
        existing = set(db.scalars(select(TrainingSample.id)).all())
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            hold = hold_from_record(payload)
            if hold is None or hold["id"] in existing:
                continue
            db.add(
                TrainingSample(
                    id=hold["id"],
                    gesture=hold["gesture"],
                    name=hold["name"],
                    gesture_type=hold["gesture_type"],
                    source=hold["source"],
                    quality=hold["quality"],
                    split=hold["split"],
                    session_id=hold["session_id"],
                    handedness=hold["handedness"],
                    feature_version=hold["feature_version"],
                    frame_count=hold["frame_count"],
                    features=hold["features"],
                    sequence=hold["sequence"],
                    landmarks=hold.get("landmarks"),
                    created_at=_parse_created(hold.get("created_at")),
                )
            )
            existing.add(hold["id"])
            added += 1
    if added:
        print(f"[db] imported {added} training samples from samples.jsonl")
        _archive(SAMPLES_PATH)
    elif SAMPLES_PATH.exists():
        _archive(SAMPLES_PATH)
    return added


def migrate_recorded_references() -> int:
    if not RECORDED_REFERENCES_PATH.exists():
        return 0
    try:
        data = json.loads(RECORDED_REFERENCES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print(f"[db] skip recorded_references.json: {err}")
        return 0
    if not isinstance(data, dict):
        return 0
    added = 0
    with session_scope() as db:
        existing = set(db.scalars(select(RecordedReference.gesture)).all())
        for gesture, payload in data.items():
            if gesture not in GESTURE_REGISTRY or not isinstance(payload, dict):
                continue
            if gesture in existing:
                continue
            spec = GESTURE_REGISTRY[gesture]
            stored = {
                **payload,
                "gesture": gesture,
                "gesture_type": spec["type"],
                "domain": spec["domain"],
            }
            db.add(
                RecordedReference(
                    gesture=gesture,
                    name=SIGN_MEANINGS.get(gesture, gesture),
                    gesture_type=spec["type"],
                    domain=spec["domain"],
                    payload=stored,
                )
            )
            existing.add(gesture)
            added += 1
    if added or RECORDED_REFERENCES_PATH.exists():
        if added:
            print(f"[db] imported {added} recorded references")
        _archive(RECORDED_REFERENCES_PATH)
    return added


def migrate_runtime() -> None:
    if not RUNTIME_PATH.exists():
        return
    try:
        payload = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _archive(RUNTIME_PATH)
        return
    if not isinstance(payload, dict):
        _archive(RUNTIME_PATH)
        return
    source = payload.get("source")
    if source not in ("heuristic", "hybrid", "model"):
        source = "heuristic"
    with session_scope() as db:
        row = db.get(MlRuntime, 1)
        if row is None:
            db.add(
                MlRuntime(
                    id=1,
                    source=source,
                    live_model_id=payload.get("live_model_id"),
                    override_gestures=list(payload.get("override_gestures") or []),
                )
            )
            print("[db] imported ml_runtime from runtime.json")
    _archive(RUNTIME_PATH)


def migrate_trained_models() -> int:
    if not MODELS_DIR.exists():
        return 0
    added = 0
    with session_scope() as db:
        existing = set(db.scalars(select(TrainedModel.version)).all())
        for meta_path in sorted(MODELS_DIR.glob("sign_clf_v*.meta.json")):
            version = meta_path.name.removeprefix("sign_clf_").removesuffix(".meta.json")
            if not _VERSION_RE.match(version) or version in existing:
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(meta, dict):
                continue
            weights_path = MODELS_DIR / f"sign_clf_{version}.json"
            if not weights_path.exists():
                continue
            try:
                weights = json.loads(weights_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(weights, dict):
                continue
            try:
                SoftmaxClassifier.from_dict(weights)
            except Exception:
                continue
            db.add(
                TrainedModel(
                    version=version,
                    algorithm=str(meta.get("algorithm") or "SoftmaxClassifier"),
                    status=str(meta.get("status") or "shadow"),
                    classes=list(meta.get("classes") or []),
                    hold_counts=dict(meta.get("hold_counts") or {}),
                    meta={**meta, "version": version},
                    weights=weights,
                )
            )
            existing.add(version)
            added += 1
            _archive(meta_path)
            _archive(weights_path)
    if added:
        print(f"[db] imported {added} trained models")
    return added


def migrate_local_files_into_postgres() -> None:
    """Move JSONL/JSON caches into Postgres so the database is the only store."""
    try:
        samples = migrate_training_samples()
        refs = migrate_recorded_references()
        migrate_runtime()
        models = migrate_trained_models()
        print(
            f"[db] persist ready samples_imported={samples} "
            f"refs_imported={refs} models_imported={models}"
        )
    except Exception as err:
        print(f"[db] persist import failed: {err}")
        raise
