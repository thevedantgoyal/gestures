"""SQLAlchemy ORM models for session history and training data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    gesture: Mapped[str] = mapped_column(String(64), nullable=False)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    coaching_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    meaning: Mapped[str | None] = mapped_column(String(128), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class SessionRecording(Base):
    """Webcam capture + event timeline for one practice session."""

    __tablename__ = "session_recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), index=True, unique=True, nullable=False
    )
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="aviation")
    video_path: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="video/webm"
    )
    timeline_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class TrainingSample(Base):
    """One confirmed gold hold, keyed by sign name (Hello, Yes, …)."""

    __tablename__ = "training_samples"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    gesture: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    gesture_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    quality: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
    split: Mapped[str] = mapped_column(String(16), nullable=False, default="train")
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    handedness: Mapped[str | None] = mapped_column(String(16), nullable=True)
    feature_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    features: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    sequence: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    landmarks: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class RecordedReference(Base):
    """User-recorded pose/motion template that overlays the hardcoded default."""

    __tablename__ = "recorded_references"

    gesture: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    gesture_type: Mapped[str] = mapped_column(String(16), nullable=False)
    domain: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class MlRuntime(Base):
    """Singleton Talk-path setting: rules vs promoted model."""

    __tablename__ = "ml_runtime"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="heuristic")
    live_model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    override_gestures: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TrainedModel(Base):
    """Saved classifier bundle after Train now."""

    __tablename__ = "trained_models"

    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="shadow")
    classes: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    hold_counts: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    weights: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
