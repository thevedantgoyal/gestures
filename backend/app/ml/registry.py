"""On-disk and Postgres sklearn bundles: sign_clf_vN + matching meta."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from app.db import session_scope
from app.ml.numpy_clf import SoftmaxClassifier
from app.models import TrainedModel

_VERSION_RE = re.compile(r"^v(\d+)$")


class RegistryError(ValueError):
    """Raised when a model bundle is missing or unreadable."""


def _version_sort_key(version: str) -> int:
    match = _VERSION_RE.match(version)
    return int(match.group(1)) if match else 0


def list_versions() -> list[str]:
    try:
        with session_scope() as db:
            versions = list(db.scalars(select(TrainedModel.version)).all())
    except Exception as err:
        print(f"[ml] list_versions failed: {err}")
        return []
    versions.sort(key=_version_sort_key)
    return versions


def next_version() -> str:
    versions = list_versions()
    if not versions:
        return "v1"
    last = versions[-1]
    match = _VERSION_RE.match(last)
    number = int(match.group(1)) + 1 if match else len(versions) + 1
    return f"v{number}"


def read_meta(version: str) -> dict[str, Any]:
    try:
        with session_scope() as db:
            row = db.get(TrainedModel, version)
            if row is None:
                raise RegistryError(f"No metadata for model {version}.")
            meta = dict(row.meta or {})
            meta["version"] = row.version
            return meta
    except RegistryError:
        raise
    except Exception as err:
        raise RegistryError(f"Could not read metadata for {version}.") from err


def latest_meta() -> dict[str, Any] | None:
    versions = list_versions()
    if not versions:
        return None
    try:
        return read_meta(versions[-1])
    except RegistryError:
        return None


def save_bundle(version: str, classifier: Any, meta: dict[str, Any]) -> dict[str, Any]:
    if not hasattr(classifier, "to_dict"):
        raise RegistryError("Cannot save this classifier (no to_dict).")
    payload = {**meta, "version": version}
    weights = classifier.to_dict()
    try:
        with session_scope() as db:
            row = db.get(TrainedModel, version)
            if row is None:
                db.add(
                    TrainedModel(
                        version=version,
                        algorithm=str(payload.get("algorithm") or "SoftmaxClassifier"),
                        status=str(payload.get("status") or "shadow"),
                        classes=list(payload.get("classes") or []),
                        hold_counts=dict(payload.get("hold_counts") or {}),
                        meta=payload,
                        weights=weights,
                    )
                )
            else:
                row.algorithm = str(payload.get("algorithm") or row.algorithm)
                row.status = str(payload.get("status") or row.status)
                row.classes = list(payload.get("classes") or [])
                row.hold_counts = dict(payload.get("hold_counts") or {})
                row.meta = payload
                row.weights = weights
    except Exception as err:
        raise RegistryError(f"PostgreSQL could not save model {version}.") from err
    return payload


def load_bundle(version: str) -> tuple[Any, dict[str, Any]]:
    try:
        with session_scope() as db:
            row = db.get(TrainedModel, version)
            if row is None:
                raise RegistryError(f"No saved weights for model {version}.")
            meta = dict(row.meta or {})
            meta["version"] = row.version
            weights = dict(row.weights or {})
    except RegistryError:
        raise
    except Exception as err:
        raise RegistryError(f"Could not read weights for {version}.") from err
    try:
        return SoftmaxClassifier.from_dict(weights), meta
    except Exception as err:
        raise RegistryError(f"Invalid weights for {version}.") from err
