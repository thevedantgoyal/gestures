"""SQLAlchemy ORM models for session / attempt logging."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
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
