"""LLM adapters: Databricks chat endpoint (default) + local Ollama fallback.

The default path generates answers with a Databricks Foundation Model chat
endpoint (OpenAI-compatible). A local Ollama client is kept only as an optional
offline fallback.

Plain English: this sends your prompt to a model and returns the generated
answer text. By default that model runs on Databricks, not your machine.
"""

from __future__ import annotations

# json/urllib are only used by the optional local Ollama fallback below.
import json
import urllib.error
import urllib.request

# The Databricks chat endpoint used by this project.
DEFAULT_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"

# Defaults for the optional local Ollama fallback (NOT the default path).
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_HOST = "http://localhost:11434"


# COMMAND ----------
# A custom error type raised when the model cannot be reached (clear, specific).
class LLMUnavailableError(RuntimeError):
    """Raised when the LLM (Databricks endpoint or local server) cannot be reached."""


# COMMAND ----------
# DEFAULT adapter: generate answers via a Databricks chat serving endpoint.
class DatabricksChatClient:
    """Minimal client for a Databricks Foundation Model chat endpoint.

    Keeps the same ``generate(prompt, system)`` interface as the Ollama client
    so the RAG chain does not change.
    """

    # Remember the endpoint, auth profile, and generation settings.
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        profile: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.endpoint = endpoint          # the serving endpoint name
        self.profile = profile            # optional CLI profile (None in app/notebook)
        self.max_tokens = max_tokens      # cap the answer length (and token usage)
        self.temperature = temperature    # 0.0 = deterministic, grounded answers
        self._client = None               # the WorkspaceClient, created lazily

    # Create (once) the Databricks client used to call the endpoint.
    def _ws(self):
        if self._client is None:
            from src.serving.databricks_client import get_workspace_client

            self._client = get_workspace_client(self.profile)
        return self._client

    # Send a prompt to the endpoint and return its text answer.
    def generate(self, prompt: str, system: str | None = None) -> str:
        # ChatMessage/ChatMessageRole describe the OpenAI-style messages array.
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        messages = []
        if system:  # the system message carries the grounding rules
            messages.append(ChatMessage(role=ChatMessageRole.SYSTEM, content=system))
        messages.append(ChatMessage(role=ChatMessageRole.USER, content=prompt))

        try:
            resp = self._ws().serving_endpoints.query(
                name=self.endpoint,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  # network/auth/quota problems -> clear error
            raise LLMUnavailableError(
                f"Could not query Databricks endpoint '{self.endpoint}'. "
                f"Check the endpoint name, your profile, and your Free Edition quota. ({exc})"
            ) from exc

        # The answer text is in the first choice's message content.
        if not resp.choices:
            return ""
        return (resp.choices[0].message.content or "").strip()


# COMMAND ----------
# OPTIONAL FALLBACK: local Ollama client (offline only, NOT the default).
class OllamaClient:
    """Minimal client for a local Ollama server's /api/generate endpoint."""

    # Store which model to call, where the server is, and how long to wait.
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")     # remove any trailing slash from the URL
        self.timeout = timeout           # seconds to wait before giving up

    # Send a prompt to the local model and return its text answer.
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
