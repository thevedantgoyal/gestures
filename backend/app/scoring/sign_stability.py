"""Temporal stability for live sign recognition.

Industry AAC pattern: never commit a word from a single noisy frame.
Vote across a short window, require a confidence margin vs the runner-up,
and hold the locked label with hysteresis until the hand clearly changes.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any


class SignStabilizer:
    """Per-websocket smoother for sign_language Talk."""

    def __init__(
        self,
        *,
        window: int = 7,
        lock_votes: int = 3,
        unlock_votes: int = 3,
        min_score: float = 0.55,
        margin: float = 0.07,
    ) -> None:
        self.window = max(3, int(window))
        self.lock_votes = max(2, int(lock_votes))
        self.unlock_votes = max(2, int(unlock_votes))
        self.min_score = float(min_score)
        self.margin = float(margin)
        self._votes: deque[tuple[str, float]] = deque(maxlen=self.window)
        self._locked: str | None = None
        self._locked_score: float = 0.0
        self._dissent: int = 0

    def reset(self) -> None:
        self._votes.clear()
        self._locked = None
        self._locked_score = 0.0
        self._dissent = 0

    def update(
        self,
        label: str | None,
        score: float,
        scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Return a stabilized decision for this frame.

        Keys: label, score, locked, stable, reason
        """
        raw = (label or "none").strip() or "none"
        conf = float(score or 0.0)
        if raw == "none" or conf < self.min_score * 0.72:
            raw = "none"
            conf = 0.0

        if scores and raw not in ("none", "wave"):
            ordered = sorted(
                ((name, float(value)) for name, value in scores.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            if len(ordered) >= 2:
                top_name, top_score = ordered[0]
                second_score = ordered[1][1]
                # Ambiguous still poses → do not promote a weak winner.
                if top_score - second_score < self.margin and raw != self._locked:
                    if self._locked and self._locked in scores:
                        raw = self._locked
                        conf = float(scores[self._locked])
                    else:
                        raw = "none"
                        conf = 0.0
                elif top_name == raw:
                    conf = max(conf, top_score)

        self._votes.append((raw, conf))

        majority, majority_count, majority_score = self._majority()
        meta = {
            "label": "none",
            "score": 0.0,
            "locked": self._locked,
            "stable": False,
            "reason": "warming",
            "majority": majority,
            "majority_count": majority_count,
        }

        if self._locked:
            if majority == self._locked and majority_count >= self.lock_votes:
                self._dissent = 0
                self._locked_score = max(self._locked_score, majority_score)
                meta.update(
                    {
                        "label": self._locked,
                        "score": self._locked_score,
                        "stable": True,
                        "reason": "hold",
                    }
                )
                return meta

            disagree = majority != self._locked and majority != "none"
            lost = majority == "none" and majority_count >= self.unlock_votes
            if disagree or lost:
                self._dissent += 1
            else:
                self._dissent = max(0, self._dissent - 1)

            if self._dissent >= self.unlock_votes:
                previous = self._locked
                self._locked = None
                self._locked_score = 0.0
                self._dissent = 0
                if (
                    majority not in ("none", previous)
                    and majority_count >= self.lock_votes
                    and majority_score >= self.min_score
                ):
                    self._locked = majority
                    self._locked_score = majority_score
                    meta.update(
                        {
                            "label": majority,
                            "score": majority_score,
                            "locked": majority,
                            "stable": True,
                            "reason": "relock",
                        }
                    )
                    return meta
                meta.update({"label": "none", "score": 0.0, "reason": "unlock"})
                return meta

            # Keep showing the locked label while dissent builds (hysteresis).
            meta.update(
                {
                    "label": self._locked,
                    "score": self._locked_score,
                    "stable": True,
                    "reason": "hysteresis",
                }
            )
            return meta

        if (
            majority != "none"
            and majority_count >= self.lock_votes
            and majority_score >= self.min_score
        ):
            self._locked = majority
            self._locked_score = majority_score
            self._dissent = 0
            meta.update(
                {
                    "label": majority,
                    "score": majority_score,
                    "locked": majority,
                    "stable": True,
                    "reason": "lock",
                }
            )
            return meta

        # Not locked yet — expose majority only as a soft attempt (UI), not a commit.
        if majority != "none" and majority_count >= max(2, self.lock_votes - 1):
            meta.update(
                {
                    "label": majority,
                    "score": majority_score,
                    "stable": False,
                    "reason": "candidate",
                }
            )
            return meta

        meta["reason"] = "unstable"
        return meta

    def _majority(self) -> tuple[str, int, float]:
        if not self._votes:
            return "none", 0, 0.0
        labels = [label for label, _score in self._votes]
        counts = Counter(labels)
        # Prefer non-none when tied with none.
        ranked = sorted(
            counts.items(),
            key=lambda item: (item[1], 0 if item[0] == "none" else 1),
            reverse=True,
        )
        majority = ranked[0][0]
        count = ranked[0][1]
        scores = [score for label, score in self._votes if label == majority]
        avg = float(sum(scores) / len(scores)) if scores else 0.0
        return majority, count, avg
