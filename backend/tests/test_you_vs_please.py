"""Pinky-only You must not be stolen by soft-thumb Please / shaka."""

from __future__ import annotations

from app.scoring.engine import (
    classify_sign_attempt,
    looks_like_please,
    looks_like_you,
)


def _pt(x: float, y: float, z: float = 0.0) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _base_hand() -> dict[str, dict[str, float]]:
    """Wrist bottom-center; knuckles in a horizontal row."""
    return {
        "wrist": _pt(0.5, 0.7),
        "thumb_cmc": _pt(0.42, 0.62),
        "thumb_mcp": _pt(0.38, 0.55),
        "thumb_ip": _pt(0.36, 0.50),
        "thumb_tip": _pt(0.35, 0.48),
        "index_mcp": _pt(0.46, 0.52),
        "index_pip": _pt(0.46, 0.48),
        "index_dip": _pt(0.46, 0.46),
        "index_tip": _pt(0.46, 0.45),
        "middle_mcp": _pt(0.50, 0.51),
        "middle_pip": _pt(0.50, 0.47),
        "middle_dip": _pt(0.50, 0.45),
        "middle_tip": _pt(0.50, 0.44),
        "ring_mcp": _pt(0.54, 0.52),
        "ring_pip": _pt(0.54, 0.48),
        "ring_dip": _pt(0.54, 0.46),
        "ring_tip": _pt(0.54, 0.45),
        "pinky_mcp": _pt(0.58, 0.54),
        "pinky_pip": _pt(0.58, 0.48),
        "pinky_dip": _pt(0.58, 0.42),
        "pinky_tip": _pt(0.58, 0.34),
    }


def _you_hand() -> dict[str, dict[str, float]]:
    """Pinky up; other fingers curled; thumb tucked near index knuckle.

    Thumb tip sits slightly above MCP so soft thumb_up can still fire —
    that used to misclassify as Please.
    """
    hand = _base_hand()
    # Curled index/middle/ring tips near knuckles.
    hand["index_tip"] = _pt(0.46, 0.56)
    hand["index_pip"] = _pt(0.46, 0.54)
    hand["middle_tip"] = _pt(0.50, 0.55)
    hand["middle_pip"] = _pt(0.50, 0.53)
    hand["ring_tip"] = _pt(0.54, 0.56)
    hand["ring_pip"] = _pt(0.54, 0.54)
    # Soft resting thumb — weakly "up" but close to index MCP.
    hand["thumb_mcp"] = _pt(0.40, 0.55)
    hand["thumb_ip"] = _pt(0.41, 0.52)
    hand["thumb_tip"] = _pt(0.42, 0.50)
    # Strong pinky up.
    hand["pinky_mcp"] = _pt(0.58, 0.54)
    hand["pinky_pip"] = _pt(0.59, 0.42)
    hand["pinky_dip"] = _pt(0.59, 0.34)
    hand["pinky_tip"] = _pt(0.59, 0.26)
    return hand


def _please_hand() -> dict[str, dict[str, float]]:
    """Shaka: thumb stuck out sideways + pinky up."""
    hand = _you_hand()
    hand["thumb_mcp"] = _pt(0.40, 0.55)
    hand["thumb_ip"] = _pt(0.30, 0.48)
    hand["thumb_tip"] = _pt(0.22, 0.42)
    return hand


def test_pinky_only_is_you_not_please() -> None:
    hand = _you_hand()
    assert looks_like_you(hand)
    assert not looks_like_please(hand)
    assert classify_sign_attempt(hand) == "you"


def test_shaka_is_please_not_you() -> None:
    hand = _please_hand()
    assert looks_like_please(hand)
    assert not looks_like_you(hand)
    assert classify_sign_attempt(hand) == "please"
