"""Helpers for logging attempts and aggregating practice sessions."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.models import Attempt


def log_attempt(
    db: Session,
    *,
    session_id: str,
    mode: str,
    gesture: str,
    score: float,
    correct: bool | None = None,
    coaching_text: str | None = None,
    meaning: str | None = None,
) -> Attempt | None:
    """Insert one Attempt row. Returns None when gesture is \"none\"."""
    if not gesture or gesture == "none":
        return None

    row = Attempt(
        session_id=session_id,
        mode=mode,
        gesture=gesture,
        correct=correct,
        score=float(score or 0.0),
        coaching_text=coaching_text,
        meaning=meaning,
    )
    db.add(row)
    db.flush()
    return row


def list_sessions(db: Session) -> list[dict[str, Any]]:
    """Aggregate attempts into session summaries, newest first."""
    start_times = (
        select(
            Attempt.session_id.label("session_id"),
            func.min(Attempt.timestamp).label("started_at"),
            func.max(Attempt.timestamp).label("ended_at"),
            func.count(Attempt.id).label("total_attempts"),
            # Prefer the mode from the earliest attempt in the session.
            func.min(Attempt.mode).label("mode_proxy"),
        )
        .group_by(Attempt.session_id)
        .subquery()
    )

    # Resolve mode from the first attempt row per session (more reliable than min()).
    sessions_raw = db.execute(
        select(
            start_times.c.session_id,
            start_times.c.started_at,
            start_times.c.ended_at,
            start_times.c.total_attempts,
        ).order_by(start_times.c.started_at.desc())
    ).all()

    results: list[dict[str, Any]] = []
    for session_id, started_at, ended_at, total in sessions_raw:
        first = db.execute(
            select(Attempt)
            .where(Attempt.session_id == session_id)
            .order_by(Attempt.timestamp.asc(), Attempt.id.asc())
            .limit(1)
        ).scalar_one()
        mode = first.mode
        summary = _session_summary(db, session_id, mode)
        results.append(
            {
                "session_id": session_id,
                "mode": mode,
                "started_at": _iso(started_at),
                "ended_at": _iso(ended_at),
                "total_attempts": int(total),
                "summary": summary,
            }
        )
    return results


def get_session_attempts(db: Session, session_id: str) -> dict[str, Any] | None:
    rows = (
        db.execute(
            select(Attempt)
            .where(Attempt.session_id == session_id)
            .order_by(Attempt.timestamp.asc(), Attempt.id.asc())
        )
        .scalars()
        .all()
    )
    if not rows:
        return None

    mode = rows[0].mode
    recording = get_session_recording_meta(db, session_id)
    return {
        "session_id": session_id,
        "mode": mode,
        "started_at": _iso(rows[0].timestamp),
        "total_attempts": len(rows),
        "summary": _session_summary(db, session_id, mode),
        "attempts": [_attempt_dict(row) for row in rows],
        "recording": recording,
    }


def get_session_recording_meta(db: Session, session_id: str) -> dict[str, Any] | None:
    from app.models import SessionRecording

    row = db.execute(
        select(SessionRecording).where(SessionRecording.session_id == session_id)
    ).scalar_one_or_none()
    if row is None:
        return None
    timeline: Any = []
    try:
        timeline = json.loads(row.timeline_json or "[]")
    except json.JSONDecodeError:
        timeline = []
    return {
        "session_id": row.session_id,
        "mode": row.mode,
        "content_type": row.content_type,
        "duration_ms": row.duration_ms,
        "timeline": timeline,
        "video_url": f"/sessions/{row.session_id}/recording/video",
        "created_at": _iso(row.created_at),
    }


def save_session_recording(
    db: Session,
    *,
    session_id: str,
    mode: str,
    video_path: str,
    content_type: str,
    timeline: list[dict[str, Any]],
    duration_ms: int | None,
) -> dict[str, Any]:
    from app.models import SessionRecording

    existing = db.execute(
        select(SessionRecording).where(SessionRecording.session_id == session_id)
    ).scalar_one_or_none()
    payload = json.dumps(timeline)
    if existing:
        existing.mode = mode
        existing.video_path = video_path
        existing.content_type = content_type
        existing.timeline_json = payload
        existing.duration_ms = duration_ms
    else:
        db.add(
            SessionRecording(
                session_id=session_id,
                mode=mode,
                video_path=video_path,
                content_type=content_type,
                timeline_json=payload,
                duration_ms=duration_ms,
            )
        )
    db.flush()
    return get_session_recording_meta(db, session_id) or {}


def get_session_recording_path(db: Session, session_id: str) -> tuple[str, str] | None:
    from app.models import SessionRecording

    row = db.execute(
        select(SessionRecording).where(SessionRecording.session_id == session_id)
    ).scalar_one_or_none()
    if row is None:
        return None
    return row.video_path, row.content_type



def _session_summary(db: Session, session_id: str, mode: str) -> dict[str, Any]:
    if mode == "aviation":
        correct_count = db.scalar(
            select(func.count(Attempt.id)).where(
                Attempt.session_id == session_id,
                Attempt.correct.is_(True),
            )
        ) or 0
        incorrect_count = db.scalar(
            select(func.count(Attempt.id)).where(
                Attempt.session_id == session_id,
                Attempt.correct.is_(False),
            )
        ) or 0
        return {
            "text": f"{correct_count} correct / {incorrect_count} incorrect",
            "correct": int(correct_count),
            "incorrect": int(incorrect_count),
        }

    # sign_language — distinct recognized meanings / gestures
    meanings = db.execute(
        select(distinct(Attempt.meaning))
        .where(
            Attempt.session_id == session_id,
            Attempt.meaning.is_not(None),
        )
        .order_by(Attempt.meaning.asc())
    ).scalars().all()
    gestures = db.execute(
        select(distinct(Attempt.gesture))
        .where(Attempt.session_id == session_id)
        .order_by(Attempt.gesture.asc())
    ).scalars().all()

    labels = [m for m in meanings if m] or list(gestures)
    if labels:
        text = "Signs: " + ", ".join(labels)
    else:
        text = "No signs recognized"
    return {
        "text": text,
        "signs": list(labels),
        "gestures": list(gestures),
    }


def _attempt_dict(row: Attempt) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "mode": row.mode,
        "gesture": row.gesture,
        "correct": row.correct,
        "score": row.score,
        "coaching_text": row.coaching_text,
        "meaning": row.meaning,
        "timestamp": _iso(row.timestamp),
    }


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
