"""Spoken meanings for recognized sign-language gestures."""

from __future__ import annotations

SIGN_MEANINGS: dict[str, str] = {
    "thumbs_up": "Yes",
    "thumbs_down": "No",
    "open_palm": "Hello",
    "fist": "Clear",
    "pointing": "Help",
    "peace_sign": "Thank you",
    "please": "Please",
    "you": "You",
    "want": "Want",
    "okay": "Okay",
    "i_love_you": "I love you",
    "wave": "Goodbye",
}

SIGN_GESTURES = tuple(SIGN_MEANINGS.keys())
