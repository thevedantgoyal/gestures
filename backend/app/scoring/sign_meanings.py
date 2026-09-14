"""Spoken meanings for recognized sign-language gestures."""

from __future__ import annotations

SIGN_MEANINGS: dict[str, str] = {
    "thumbs_up": "Yes",
    "open_palm": "Hello",
    "fist": "Stop",
    "pointing": "Help",
    "peace_sign": "Thank you",
    "wave": "Goodbye",
}

SIGN_GESTURES = tuple(SIGN_MEANINGS.keys())
