# Databricks notebook source
"""Local LLM client via Ollama (zero-cost generation).

Talks to a locally running Ollama server over its HTTP API. No paid API is used.
If the server is not running, a clear error is raised rather than silently
switching to a paid service.

Plain English: this sends your prompt to a model running on your own machine
(via Ollama) and returns the generated answer text.
"""

from __future__ import annotations

# json to build/parse the request body; urllib to make the HTTP call (standard library).
import json
import urllib.error
import urllib.request

# Defaults: a small local model, and Ollama's default address on your machine.
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_HOST = "http://localhost:11434"


# COMMAND ----------
# A custom error type we raise when Ollama cannot be reached (clear, specific).
class LLMUnavailableError(RuntimeError):
    """Raised when the local LLM server cannot be reached."""


# COMMAND ----------
# The client object that talks to Ollama.
class OllamaClient:
    """Minimal client for Ollama's /api/generate endpoint."""

    # Store which model to call, where the server is, and how long to wait.
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")     # remove any trailing slash from the URL
        self.timeout = timeout           # seconds to wait before giving up

    # Send a prompt to the model and return its text answer.
    def generate(self, prompt: str, system: str | None = None) -> str:
        # Build the request body Ollama expects. stream=False -> get the whole answer at once.
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if system:                       # include the system rules if provided
            payload["system"] = system
        # Prepare the HTTP POST request with the JSON body.
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),        # turn the dict into JSON bytes
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            # Send the request and read the response (with a timeout).
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))  # parse the JSON reply
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # If the server is down/unreachable, raise a clear, actionable error.
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start it and `ollama pull {self.model}`. ({exc})"
            ) from exc
        # The answer text is under the "response" key; strip surrounding whitespace.
        return (data.get("response") or "").strip()
