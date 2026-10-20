# Databricks notebook source
"""Local LLM client via Ollama (zero-cost generation).

Talks to a locally running Ollama server over its HTTP API. No paid API is used.
If the server is not running, a clear error is raised rather than silently
switching to a paid service.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_HOST = "http://localhost:11434"


# COMMAND ----------
class LLMUnavailableError(RuntimeError):
    """Raised when the local LLM server cannot be reached."""


# COMMAND ----------
class OllamaClient:
    """Minimal client for Ollama's /api/generate endpoint."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start it and `ollama pull {self.model}`. ({exc})"
            ) from exc
        return (data.get("response") or "").strip()
