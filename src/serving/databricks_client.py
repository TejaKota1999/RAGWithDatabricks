"""Shared Databricks workspace client helper (authentication).

Centralizes how the RAG code authenticates to Databricks so every adapter
(embedder, retriever, llm) connects the same way.

Authentication order (handled by the Databricks SDK automatically):
  - In a Databricks App or notebook: the injected credentials are used, so
    no profile is needed (pass profile=None).
  - On a local machine: set the CLI profile via the DATABRICKS_CONFIG_PROFILE
    environment variable, or pass profile="sairam" explicitly.

No secrets are stored here; the SDK reads them from the environment / CLI config.
"""

from __future__ import annotations

# os: read the optional profile from the environment; functools: cache the client.
import os
from functools import lru_cache


# COMMAND ----------
# Return a cached Databricks WorkspaceClient. Cached per profile so we reuse one
# connection instead of opening a new one on every call.
@lru_cache(maxsize=4)
def get_workspace_client(profile: str | None = None):
    """Return a databricks.sdk.WorkspaceClient.

    If ``profile`` is None we fall back to the DATABRICKS_CONFIG_PROFILE env var
    (if set); if that is also unset, the SDK uses its default resolution
    (injected app/notebook credentials, or the DEFAULT CLI profile).
    """
    from databricks.sdk import WorkspaceClient

    # Use the explicit profile, else the env var, else let the SDK decide.
    profile = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if profile:
        return WorkspaceClient(profile=profile)
    return WorkspaceClient()
