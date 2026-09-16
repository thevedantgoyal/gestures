"""Multinomial logistic regression in NumPy — real gradient-descent training."""

from __future__ import annotations

from typing import Any

import numpy as np

LEARNING_RATE = 0.25
EPOCHS = 400
L2 = 1e-3
RANDOM_STATE = 42


class SoftmaxClassifier:
    """Softmax / multinomial logistic regression with L2 regularization."""

    def __init__(
        self,
        *,
        learning_rate: float = LEARNING_RATE,
        epochs: int = EPOCHS,
        l2: float = L2,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.weights: np.ndarray | None = None  # (n_features + 1, n_classes)

    def fit(self, x: np.ndarray, y: np.ndarray) -> SoftmaxClassifier:
        classes = np.asarray(sorted(set(y.tolist())), dtype=object)
        self.classes_ = classes
        class_index = {name: index for index, name in enumerate(classes.tolist())}
        n_samples, n_features = x.shape
        n_classes = len(classes)
        weights = np.zeros((n_features + 1, n_classes), dtype=float)
        scale = 0.01 / float(max(self.random_state, 1))
        for row in range(n_features + 1):
            for col in range(n_classes):
                weights[row, col] = (
                    ((row + 1) * (col + 3) * (self.random_state % 97)) % 997
                ) * scale / 997.0
        bias_ones = np.ones((n_samples, 1), dtype=float)
        features = np.concatenate([x, bias_ones], axis=1)
        one_hot = np.zeros((n_samples, n_classes), dtype=float)
        for row, label in enumerate(y.tolist()):
            one_hot[row, class_index[label]] = 1.0

        for _epoch in range(self.epochs):
            logits = features @ weights
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
            error = probs - one_hot
            grad = (features.T @ error) / n_samples
            grad[:-1] += self.l2 * weights[:-1]
            weights -= self.learning_rate * grad

        self.weights = weights
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("SoftmaxClassifier is not fitted.")
        bias_ones = np.ones((x.shape[0], 1), dtype=float)
        features = np.concatenate([x, bias_ones], axis=1)
        logits = features @ self.weights
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("SoftmaxClassifier is not fitted.")
        probs = self.predict_proba(x)
        indices = probs.argmax(axis=1)
        return np.asarray([self.classes_[index] for index in indices], dtype=object)

    def to_dict(self) -> dict[str, Any]:
        if self.weights is None or self.classes_ is None:
            raise RuntimeError("SoftmaxClassifier is not fitted.")
        return {
            "type": "softmax",
            "classes": [str(item) for item in self.classes_.tolist()],
            "weights": self.weights.astype(float).tolist(),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SoftmaxClassifier:
        model = cls(
            learning_rate=float(payload.get("learning_rate", LEARNING_RATE)),
            epochs=int(payload.get("epochs", EPOCHS)),
            l2=float(payload.get("l2", L2)),
            random_state=int(payload.get("random_state", RANDOM_STATE)),
        )
        model.classes_ = np.asarray(payload["classes"], dtype=object)
        model.weights = np.asarray(payload["weights"], dtype=float)
        return model
