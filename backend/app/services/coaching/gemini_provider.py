"""Gemini coaching provider — active when COACHING_PROVIDER = \"gemini\"."""

from __future__ import annotations

import os
from typing import Any

from app.config import GEMINI_MODEL
from app.services.coaching.base import build_coaching_prompt


def get_coaching_text(deviation_data: dict[str, Any]) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = build_coaching_prompt(deviation_data)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )

    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise RuntimeError("Gemini returned an empty coaching response")

    return str(text).strip()
