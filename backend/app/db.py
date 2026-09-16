"""PostgreSQL connection via SQLAlchemy.

Reads DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD from the environment
(see .env / .env.example).
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Ensure backend/.env is loaded when this module is imported early.
from app import config as _config  # noqa: F401


def _sqlalchemy_driver() -> str:
    """Prefer psycopg2; fall back to pg8000 if the native DLL is blocked."""
    try:
        import psycopg2  # noqa: F401

        return "psycopg2"
    except Exception:
        return "pg8000"


def _database_url() -> str:
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "gestures")
    user = os.environ.get("DB_USER", "postgres")
    password = quote_plus(os.environ.get("DB_PASSWORD", "postgres"))
    return (
        f"postgresql+{_sqlalchemy_driver()}://"
        f"{quote_plus(user)}:{password}@{host}:{port}/{quote_plus(name)}"
    )


def _db_target_label() -> str:
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "gestures")
    user = os.environ.get("DB_USER", "postgres")
    return f"{user}@{host}:{port}/{name}"


class Base(DeclarativeBase):
    pass


engine = create_engine(
    _database_url(),
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create tables if they do not already exist. Raises on connection failure."""
    from app import models  # noqa: F401

    target = _db_target_label()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(bind=engine)
        # Imported here to avoid a persist <-> db cycle at module load.
        from app.persist import migrate_local_files_into_postgres

        migrate_local_files_into_postgres()
        print(f"[db] tables ready ({target})")
    except Exception as err:
        print("[db] ERROR: PostgreSQL is not reachable — session history will not work.")
        print(f"[db]   target: {target}")
        print(f"[db]   detail: {err}")
        print(
            "[db]   Fix DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD in "
            "backend/.env, then restart uvicorn."
        )
        raise


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency-style session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
