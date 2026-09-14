"""Shared coaching contract and prompt helpers."""

from __future__ import annotations

from typing import Any, Protocol


class CoachingProvider(Protocol):
    def get_coaching_text(self, deviation_data: dict[str, Any]) -> str:
        """Return a short coaching tip for an incorrect gesture attempt."""


def build_coaching_prompt(deviation_data: dict[str, Any]) -> str:
    """Build the coaching prompt from scored deviation metrics."""
    gesture = deviation_data.get("gesture", "unknown")
    score = deviation_data.get("score")
    hints = deviation_data.get("hints") or []
    hint_line = (
        f"- Local hints: {'; '.join(hints)}\n" if hints else ""
    )

    if gesture == "exit_pointing":
        tol = deviation_data.get("tolerance")
        pass_band = deviation_data.get("pass_band_deg")
        details = (
            f"- Gesture: Exit Pointing (arms out in a near-horizontal line)\n"
            f"- Score: {score} (pass at about 0.55+)\n"
            f"- Left arm angle: {deviation_data.get('left_arm_angle')}°\n"
            f"- Right arm angle: {deviation_data.get('right_arm_angle')}°\n"
            f"- Target angle: {deviation_data.get('target_angle')}° "
            f"(comfortable zone about ±{pass_band or tol}°)\n"
            f"- Scoring tolerance: ±{tol}°\n"
            f"{hint_line}"
        )
        goal = (
            "They should hold both arms out near shoulder height, elbows nearly "
            "straight, wrists roughly level with the shoulders."
        )
    elif gesture == "seatbelt_demo":
        details = (
            f"- Gesture: Seatbelt Demo (hands slide together at the waist)\n"
            f"- Score: {score} (pass at about 0.55+)\n"
            f"- DTW distance: {deviation_data.get('dtw_distance')}\n"
            f"{hint_line}"
            f"- Deviations: {deviation_data.get('deviations')}\n"
        )
        goal = (
            "They should bring both hands to waist height and smoothly slide them "
            "together until they meet in front of the body."
        )
    else:
        details = f"- Gesture: {gesture}\n- Score: {score}\n- Data: {deviation_data}\n"
        goal = "Help them match the required cabin-crew gesture."

    return (
        "You are a concise aviation cabin-crew gesture coach.\n"
        "The trainee is actively moving but the gesture is outside the acceptable "
        "range. Using ONLY the metrics below, write ONE short sentence that says "
        "they are doing it wrong and gives one concrete physical correction "
        "(which arm/hand, raise/lower/straighten/slide). Prefer the local hints "
        "when present. Do not mention scores, percentages, JSON, or that you are "
        "an AI. No bullet points.\n\n"
        f"Goal: {goal}\n\n"
        f"{details}"
    )


def get_coaching_text(deviation_data: dict[str, Any]) -> str:
    """Protocol-level signature expected by every provider module."""
    raise NotImplementedError(
        "Call app.services.coaching.get_coaching_text, not the base module."
    )
