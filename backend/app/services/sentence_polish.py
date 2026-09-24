"""Polish AAC sign tokens into a natural spoken sentence via Ollama.

Prefer curated templates. Call Ollama only for light grammar polish.
Fall back to the template when the LLM is down, slow, or invents meaning.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

# Ordered token keys that already have a good composeSignSentence clause.
# Keep in sync with frontend/src/lib/signSentence.ts PAIR_CLAUSES / TRIPLE_CLAUSES.
_KNOWN_GOOD_KEYS: frozenset[str] = frozenset(
    {
        "Hello|Help",
        "Hello|You",
        "Hello|Thank you",
        "Hello|Goodbye",
        "Hello|I love you",
        "Yes|Help",
        "Yes|Please",
        "Please|Help",
        "Please|You",
        "Please|Thank you",
        "You|Help",
        "You|Thank you",
        "Want|Help",
        "Want|You",
        "Want|Please",
        "No|Help",
        "No|Thank you",
        "Okay|Thank you",
        "Okay|Goodbye",
        "Thank you|Goodbye",
        "I love you|You",
        "Please|Want|Help",
        "Yes|Please|Help",
        "Please|You|Help",
        "Hello|I love you|You",
    }
)

# Meaning → stems that must still appear in a polished sentence.
_TOKEN_STEMS: dict[str, tuple[str, ...]] = {
    "Hello": ("hello",),
    "Yes": ("yes",),
    "No": ("no",),
    "Help": ("help",),
    "Thank you": ("thank",),
    "Goodbye": ("goodbye", "bye"),
    "Please": ("please",),
    "You": ("you",),
    "Want": ("want",),
    "Okay": ("okay", "ok"),
    "I love you": ("love",),
}

_SYSTEM = (
    "You turn AAC signed words into ONE natural spoken English sentence. "
    "Keep the same meaning. Include every signed idea. "
    "Do not invent people, places, or new actions. "
    "Do not reply with a bare list of words. "
    "No quotes, no markdown, no explanation.\n"
    "Examples:\n"
    "Signed: Please, Want, Help → Please, I want help.\n"
    "Signed: Hello, You → Hello, how are you.\n"
    "Signed: Yes, Please, Help → Yes, please help me.\n"
    "Signed: Want, Help → I want help."
)


def _chat_url() -> str:
    base = (OLLAMA_BASE_URL or "").rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _normalize_sentence(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _sanitize(text: str) -> str | None:
    cleaned = _normalize_sentence(text)
    if not cleaned:
        return None
    if len(cleaned) > 180 or "\n" in cleaned:
        first = cleaned.split("\n", 1)[0].strip()
        if not first or len(first) > 180:
            return None
        cleaned = first
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _token_key(words: list[str]) -> str:
    return "|".join(words)


def _required_stems(words: list[str]) -> list[str]:
    stems: list[str] = []
    for word in words:
        for stem in _TOKEN_STEMS.get(word, ()):
            if stem not in stems:
                stems.append(stem)
        if word not in _TOKEN_STEMS:
            # Unknown token: require its lowercase form if alphabetic.
            compact = re.sub(r"[^a-z]+", "", word.lower())
            if compact and compact not in stems:
                stems.append(compact)
    return stems


def _has_stem(haystack: str, stem: str) -> bool:
    return bool(re.search(rf"\b{re.escape(stem)}\b", haystack, flags=re.IGNORECASE))


def _template_already_good(words: list[str], template: str) -> bool:
    """Skip the LLM only for a single signed word (nothing to compose)."""
    if not template:
        return False
    # One word: speak the template as-is. Multi-word: always try Ollama.
    return len(words) <= 1


def _reply_keeps_meaning(words: list[str], template: str, reply: str) -> bool:
    """Reject inventing/dropping meaning relative to signed words + draft."""
    stems = _required_stems(words)
    if stems and not all(_has_stem(reply, stem) for stem in stems):
        return False

    # Do not allow a much longer rewrite (usually means invented filler).
    if template and len(reply) > max(len(template) + 40, int(len(template) * 1.8) + 12):
        return False

    # Block common invented roles/places that AAC signs did not include.
    banned = (
        "doctor",
        "hospital",
        "police",
        "teacher",
        "school",
        "friend",
        "mom",
        "dad",
        "mother",
        "father",
        "assist me",
        "can you",
        "could you",
    )
    reply_l = reply.lower()
    template_l = template.lower()
    for phrase in banned:
        if phrase in reply_l and phrase not in template_l:
            # Only ban if those words were not part of the signed stems.
            if not any(phrase.startswith(stem) or stem in phrase for stem in stems):
                return False
    return True


def polish_sentence_with_ollama(
    tokens: list[str],
    *,
    fallback: str,
) -> dict[str, Any]:
    """Return {sentence, source: 'ollama'|'fallback'|'template', detail?}."""
    template = _sanitize(fallback or "") or (fallback or "").strip()
    words = [str(token).strip() for token in tokens if str(token).strip()]
    if not words:
        return {"sentence": "", "source": "fallback", "detail": "empty"}

    if not template:
        # Last-resort draft if client forgot fallback.
        template = _sanitize(", ".join(words)) or ", ".join(words)

    if _template_already_good(words, template):
        return {
            "sentence": template if template.endswith((".", "!", "?")) else f"{template}.",
            "source": "template",
            "detail": "curated_or_complete",
        }

    if not OLLAMA_BASE_URL:
        return {
            "sentence": template,
            "source": "fallback",
            "detail": "OLLAMA_BASE_URL not set",
        }

    user_prompt = (
        "Signed words in order: "
        + ", ".join(words)
        + ".\n"
        "Draft sentence (use this if unsure): "
        + template
        + "\n"
        "Reply with ONE natural spoken sentence only. "
        "Prefer the draft when it already uses the signed words."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 60,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _chat_url(),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        sentence = _sanitize(str(content or ""))
        if not sentence:
            return {
                "sentence": template,
                "source": "fallback",
                "detail": "empty_or_invalid_llm_reply",
            }
        if not _reply_keeps_meaning(words, template, sentence):
            print(f"[ollama] rejected inventing/dropping reply: {sentence!r}")
            return {
                "sentence": template,
                "source": "fallback",
                "detail": "rejected_meaning_drift",
            }
        return {"sentence": sentence, "source": "ollama"}
    except urllib.error.HTTPError as err:
        detail = f"http_{err.code}"
        print(f"[ollama] polish failed: {detail}")
        return {"sentence": template, "source": "fallback", "detail": detail}
    except Exception as err:  # noqa: BLE001
        detail = type(err).__name__
        print(f"[ollama] polish failed: {err}")
        return {"sentence": template, "source": "fallback", "detail": detail}
