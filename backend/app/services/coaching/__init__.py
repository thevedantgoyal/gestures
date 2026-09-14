"""Coaching router — picks Gemini or Bedrock from app.config.COACHING_PROVIDER."""

from __future__ import annotations

from typing import Any

from app.config import COACHING_FALLBACK_TEXT, COACHING_PROVIDER
from app.services.coaching import bedrock_provider, gemini_provider


def get_coaching_text(deviation_data: dict[str, Any]) -> str:
    """Route to the configured provider; never raise to the WebSocket handler."""
    provider = (COACHING_PROVIDER or "gemini").strip().lower()

    try:
        if provider == "bedrock":
            return bedrock_provider.get_coaching_text(deviation_data)
        if provider == "gemini":
            return gemini_provider.get_coaching_text(deviation_data)
        raise ValueError(f"Unknown COACHING_PROVIDER: {provider!r}")
    except Exception as err:  # noqa: BLE001 — coaching must never crash scoring
        print(f"[coaching] provider={provider} failed: {err}")
        return COACHING_FALLBACK_TEXT
