"""App-level configuration. Switch COACHING_PROVIDER to "bedrock" later."""

from __future__ import annotations

import os
from pathlib import Path

# Load backend/.env if present (does nothing when the file or package is missing).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass

# Active coaching backend. Change to "bedrock" when AWS credentials are ready.
COACHING_PROVIDER = os.environ.get("COACHING_PROVIDER", "gemini")  # change to "bedrock" later

# Fallback shown if the live provider errors (bad key, rate limit, network).
COACHING_FALLBACK_TEXT = "Adjust your arm position and try again."

# Gemini model id used by gemini_provider.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Bedrock defaults for the stub provider (override via env when wiring AWS).
BEDROCK_REGION = os.environ.get("AWS_REGION") or os.environ.get(
    "AWS_DEFAULT_REGION", "us-east-1"
)
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)
