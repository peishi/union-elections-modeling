"""Minimal OpenCode Go chat-completions client (OpenAI-compatible).

Used by the interactive UI and the batch labeling pipeline.  The Go lane
requires an ``x-opencode-session`` header and a browser-like ``User-Agent``;
it serves both the primary and fallback models over ``/chat/completions``.

When the primary model is exhausted (HTTP 429 / quota-style errors) the client
switches to ``fallback_model`` for the rest of the run and records which model
produced each result.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

LOG = logging.getLogger("naics-llm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "gemini-4-argon"
DEFAULT_FALLBACK_MODEL = "claude-haiku-5-5"

# Friendly aliases accepted anywhere a model name is passed.  Both models are
# served by the same OpenAI-compatible endpoint, so they differ only by name.
MODEL_ALIASES = {
    "gemini": DEFAULT_MODEL,
    "gemini-4": DEFAULT_MODEL,
    DEFAULT_MODEL: DEFAULT_MODEL,
    "claude": DEFAULT_FALLBACK_MODEL,
    "haiku": DEFAULT_FALLBACK_MODEL,
    "claude-haiku": DEFAULT_FALLBACK_MODEL,
    DEFAULT_FALLBACK_MODEL: DEFAULT_FALLBACK_MODEL,
}


def resolve_model(name: str) -> str:
    """Map a model alias (``gemini``, ``claude``, ...) to its full model ID."""
    key = str(name).strip().lower()
    try:
        return MODEL_ALIASES[key]
    except KeyError:
        known = ", ".join(sorted({*MODEL_ALIASES.values()}))
        aliases = ", ".join(
            sorted(alias for alias in MODEL_ALIASES if alias not in MODEL_ALIASES.values())
        )
        raise ValueError(
            f"unknown model {name!r}; choose one of: {known} (aliases: {aliases})"
        ) from None
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

RETRYABLE_STATUS = {408, 425, 500, 502, 503, 504}
EXHAUSTED_STATUS = {402, 403, 429}
EXHAUSTED_WORDS = (
    "exhaust",
    "quota",
    "usage limit",
    "rate limit",
    "insufficient",
    "out of credit",
)


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE pairs from ``.env`` without extra dependencies."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def api_key() -> str | None:
    return os.environ.get("OPENCODE_GO_API") or os.environ.get("OPENCODE_API_KEY")


@dataclass
class ChatResult:
    content: str
    model: str
    usage: dict[str, int]
    raw: dict[str, Any]


def _extract_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    opencode_tokens = (usage.get("opencode") or {}).get("tokens") or {}
    if prompt is None:
        prompt = opencode_tokens.get("input", 0)
    if completion is None:
        completion = opencode_tokens.get("output", 0)
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    total = int(total if total is not None else prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


class LLMClient:
    """Thread-safe chat-completions client with one-step model fallback."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        key: str | None = None,
        timeout: float = 90.0,
        retries: int = 4,
        session: str = "naics-pipeline",
    ) -> None:
        load_dotenv()
        self.key = key or api_key()
        if not self.key:
            raise RuntimeError(
                "OPENCODE_GO_API is not set (add it to .env or the environment)"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        if fallback_model and fallback_model == model:
            LOG.warning(
                "fallback model equals the starting model (%s); fallback disabled",
                model,
            )
            fallback_model = ""
        self.fallback_model = fallback_model
        self.active_model = model
        self.fallback_active = False
        self.timeout = timeout
        self.retries = max(1, retries)
        self.session = session
        self._lock = threading.Lock()
        self.usage_totals: dict[str, dict[str, int]] = {}
        self.calls = 0

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _is_exhausted(status: int, detail: str) -> bool:
        if status in EXHAUSTED_STATUS:
            return True
        lowered = detail.lower()
        return any(word in lowered for word in EXHAUSTED_WORDS)

    def _switch_to_fallback(self) -> bool:
        with self._lock:
            if self.fallback_active or not self.fallback_model:
                return False
            LOG.warning(
                "model %s exhausted; switching to fallback %s",
                self.active_model,
                self.fallback_model,
            )
            self.active_model = self.fallback_model
            self.fallback_active = True
            return True

    def _post(self, model: str, messages: Sequence[dict], max_tokens: int) -> dict:
        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "User-Agent": BROWSER_UA,
                "x-opencode-session": self.session,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _record(self, model: str, usage: dict[str, int]) -> None:
        with self._lock:
            self.calls += 1
            bucket = self.usage_totals.setdefault(
                model,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                bucket[name] += int(usage.get(name, 0))

    # -- public ------------------------------------------------------------

    def chat(
        self, messages: Sequence[dict], max_tokens: int = 1024
    ) -> ChatResult:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            model = self.active_model
            try:
                body = self._post(model, messages, max_tokens)
                usage = _extract_usage(body.get("usage"))
                self._record(model, usage)
                message = body["choices"][0]["message"]
                return ChatResult(
                    content=message.get("content") or "",
                    model=model,
                    usage=usage,
                    raw=body,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if self._is_exhausted(exc.code, detail) and self._switch_to_fallback():
                    continue
                if exc.code in RETRYABLE_STATUS and attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 20))
                    continue
                raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 20))
                    continue
                raise RuntimeError(f"request failed: {exc}") from exc
        raise RuntimeError(f"request failed: {last_error}")

    def total_usage(self) -> dict[str, int]:
        with self._lock:
            totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for bucket in self.usage_totals.values():
                for name in totals:
                    totals[name] += bucket[name]
            return totals
