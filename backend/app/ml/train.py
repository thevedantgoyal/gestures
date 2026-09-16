"""Train a softmax sign classifier from gold pose holds."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.ml.dataset import (
    FEATURE_DIM,
    FEATURE_VERSION,
    MIN_TRAIN_CLASSES,
    MIN_TRAIN_HOLDS_PER_CLASS,
    DatasetStoreError,
    collect_static_examples,
    train_readiness,
)
from app.ml.numpy_clf import EPOCHS, L2, LEARNING_RATE, SoftmaxClassifier
from app.ml.registry import next_version, save_bundle

RANDOM_STATE = 42
VAL_FRACTION = 0.25


class TrainError(ValueError):
    """Raised when a training run cannot start or finish."""


def _hold_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    seen: dict[str, set[str]] = {}
    for row in examples:
        gesture = str(row["gesture"])
        seen.setdefault(gesture, set()).add(str(row["hold_id"]))
    return {name: len(ids) for name, ids in seen.items()}


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> list[list[int]]:
    index = {name: i for i, name in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for truth, pred in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        row = index.get(str(truth))
        col = index.get(str(pred))
        if row is None or col is None:
            continue
        matrix[row][col] += 1
    return matrix


def _per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    matrix = _confusion(y_true, y_pred, classes)
    metrics: dict[str, Any] = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(classes)) if r != i)
        fn = sum(matrix[i][c] for c in range(len(classes)) if c != i)
        support = sum(matrix[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        metrics[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }
    return metrics


def _group_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    unique_groups = sorted({str(item) for item in groups.tolist()})
    if len(unique_groups) < 4:
        return x, y, None, None, "train_only"
    n_val = max(1, int(round(len(unique_groups) * VAL_FRACTION)))
    n_val = min(n_val, len(unique_groups) - 1)
    val_groups = set(unique_groups[-n_val:])
    val_mask = [str(item) in val_groups for item in groups.tolist()]
    train_idx = [i for i, flag in enumerate(val_mask) if not flag]
    val_idx = [i for i, flag in enumerate(val_mask) if flag]
    if not train_idx or not val_idx:
        return x, y, None, None, "train_only"
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], "group_holdout"


def train_sign_classifier() -> dict[str, Any]:
    """Fit a softmax classifier on gold holds. Talk stays on rules until promote."""
    readiness = train_readiness()
    if not readiness["can_train"]:
        detail = " ".join(readiness["blockers"]) or "Not enough labeled holds."
        raise TrainError(detail)

    included = set(readiness["included_classes"])
    examples = [
        row
        for row in collect_static_examples()
        if row["gesture"] in included and len(row["features"]) == FEATURE_DIM
    ]
    if not examples:
        raise TrainError("No usable feature rows in the gold dataset.")

    x = np.asarray([row["features"] for row in examples], dtype=float)
    y = np.asarray([row["gesture"] for row in examples], dtype=object)
    groups = np.asarray([row["hold_id"] for row in examples], dtype=object)
    classes = sorted({str(item) for item in y.tolist()})
    if len(classes) < MIN_TRAIN_CLASSES:
        raise TrainError(
            f"Need at least {MIN_TRAIN_CLASSES} sign classes; found {classes}."
        )

    x_train, y_train, x_val, y_val, split_used = _group_split(x, y, groups)
    classifier = SoftmaxClassifier(random_state=RANDOM_STATE)
    classifier.fit(x_train, y_train)

    train_pred = classifier.predict(x_train)
    train_accuracy = _accuracy(y_train, train_pred)
    test_x = x_val if x_val is not None else x_train
    test_y = y_val if y_val is not None else y_train
    test_pred = classifier.predict(test_x)
    val_accuracy = _accuracy(test_y, test_pred)
    per_class = _per_class(test_y, test_pred, classes)
    matrix = _confusion(test_y, test_pred, classes)

    version = next_version()
    hold_counts = _hold_counts(examples)
    meta = {
        "version": version,
        "status": "shadow",
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "algorithm": "SoftmaxClassifier",
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "l2": L2,
            "random_state": RANDOM_STATE,
        },
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "classes": classes,
        "hold_counts": hold_counts,
        "n_rows": int(len(examples)),
        "n_train_rows": int(len(y_train)),
        "n_val_rows": int(len(test_y) if x_val is not None else 0),
        "split": split_used,
        "train_accuracy": round(train_accuracy, 4),
        "val_accuracy": round(val_accuracy, 4),
        "accuracy_split": "validation" if x_val is not None else "train",
        "per_class": per_class,
        "confusion_matrix": {
            "labels": classes,
            "matrix": matrix,
        },
        "warnings": list(readiness["warnings"]),
        "min_holds_per_class": MIN_TRAIN_HOLDS_PER_CLASS,
        "row_counts": dict(Counter(str(item) for item in y.tolist())),
    }
    saved = save_bundle(version, classifier, meta)
    print(
        f"[ml] trained {version} classes={classes} "
        f"train_acc={train_accuracy:.3f} val_acc={val_accuracy:.3f} "
        f"split={split_used}"
    )
    return saved
