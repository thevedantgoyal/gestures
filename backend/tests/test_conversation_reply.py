"""Unit tests for AAC conversation reply (Phase 1).

Run from the backend directory:
  python -m unittest tests.test_conversation_reply
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from app.services import conversation_reply as cr


class ConversationReplyTests(unittest.TestCase):
    def test_empty_sentence_returns_fallback(self) -> None:
        result = cr.generate_conversation_reply("")
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["reply"], cr.FALLBACK_REPLY)
        self.assertEqual(result["detail"], "empty_sentence")

    def test_missing_ollama_url_returns_fallback(self) -> None:
        with patch.object(cr, "OLLAMA_BASE_URL", ""):
            result = cr.generate_conversation_reply("I want help.")
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["reply"], cr.FALLBACK_REPLY)
        self.assertIn("OLLAMA_BASE_URL", str(result.get("detail")))

    def test_ollama_success_returns_reply(self) -> None:
        fake_body = {
            "choices": [
                {"message": {"content": "Sure. What do you need help with?"}}
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(fake_body).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch.object(cr, "OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1/"):
            with patch.object(cr.urllib.request, "urlopen", return_value=mock_response):
                result = cr.generate_conversation_reply(
                    "I want help.",
                    tokens=["Want", "Help"],
                    history=[
                        {"role": "user", "content": "Hello, how are you?"},
                        {
                            "role": "assistant",
                            "content": "Hello. I am here to help.",
                        },
                    ],
                )
        self.assertEqual(result["source"], "ollama")
        self.assertEqual(result["reply"], "Sure. What do you need help with?")

    def test_ollama_http_error_returns_fallback(self) -> None:
        with patch.object(cr, "OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1/"):
            with patch.object(
                cr.urllib.request,
                "urlopen",
                side_effect=cr.urllib.error.HTTPError(
                    url="http://x",
                    code=503,
                    msg="unavailable",
                    hdrs=None,
                    fp=None,
                ),
            ):
                result = cr.generate_conversation_reply("Hello.")
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["reply"], cr.FALLBACK_REPLY)
        self.assertEqual(result["detail"], "http_503")

    def test_history_is_capped(self) -> None:
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]
        cleaned = cr._normalize_history(history)
        self.assertEqual(len(cleaned), cr.MAX_HISTORY_TURNS)
        self.assertEqual(cleaned[0]["content"], "msg 12")


class ConversationReplyRouteSmoke(unittest.TestCase):
    """Import-level smoke: request models and route exist on the app."""

    def test_route_registered(self) -> None:
        from main import app

        paths = {route.path for route in app.routes}
        self.assertIn("/ml/reply", paths)
        self.assertIn("/ml/polish-sentence", paths)


if __name__ == "__main__":
    unittest.main()
