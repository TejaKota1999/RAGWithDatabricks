"""Live pipeline status for the app's "How it works" flow view.

Reads ONLY cheap metadata from the workspace - no SQL warehouse, no model calls:
  - the source-docs volume listing        (Files API)
  - the Vector Search index status        (indexed row count, ready flag)
  - the latest run of the ingest job      (Jobs API)

Every function returns a plain dict and never raises: on failure (missing
permission, resource not found, quota shut-off) it returns {"error": "..."} so
the UI can show that one card as unavailable while the rest still render.
"""

from __future__ import annotations

from datetime import datetime, timezone


# Convert a Databricks epoch-milliseconds timestamp to an aware UTC datetime.
def _from_ms(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc) if ms else None


# Short, safe error text for the UI (exception type + first line, no stack trace).
def _err(exc: Exception) -> dict:
    msg = str(exc).splitlines()[0] if str(exc) else ""
    return {"error": f"{type(exc).__name__}: {msg[:160]}"}


# COMMAND ----------
# Files currently in the source-docs volume (newest first).
def volume_files(ws, volume_path: str) -> dict:
    try:
        files = [
            {"name": f.name, "size": f.file_size, "modified": _from_ms(f.last_modified)}
            for f in ws.files.list_directory_contents(volume_path)
            if not f.is_directory
        ]
    except Exception as exc:
        return _err(exc)
    files.sort(key=lambda f: f["modified"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return {"path": volume_path, "files": files}


# Status of the managed-embeddings Vector Search index.
def index_status(ws, index_name: str) -> dict:
    try:
        idx = ws.vector_search_indexes.get_index(index_name)
    except Exception as exc:
        return _err(exc)
    status = idx.status
    spec = idx.delta_sync_index_spec
    cols = (spec.embedding_source_columns or []) if spec else []
    return {
        "name": index_name,
        "endpoint": idx.endpoint_name,
        "ready": bool(status and status.ready),
        "row_count": status.indexed_row_count if status else None,
        "message": status.message if status else None,
        "source_table": spec.source_table if spec else None,
        "embedding_endpoint": cols[0].embedding_model_endpoint_name if cols else None,
    }


# Latest run of the ingest job. Bundle deployments prefix the job name (e.g.
# "[dev user] [dev] RAG Ingest (AI Functions)"), so we match by substring.
def last_job_run(ws, job_name_contains: str) -> dict:
    try:
        job = next((j for j in ws.jobs.list() if job_name_contains in (j.settings.name or "")), None)
        if job is None:
            return {"error": f"No job whose name contains '{job_name_contains}'"}
        run = next(iter(ws.jobs.list_runs(job_id=job.job_id, limit=1)), None)
    except Exception as exc:
        return _err(exc)
    out = {"job_id": job.job_id, "job_name": job.settings.name, "run": None}
    if run is not None:
        state = run.state
        out["run"] = {
            "life_cycle": state.life_cycle_state.value if state and state.life_cycle_state else None,
            "result": state.result_state.value if state and state.result_state else None,
            "start": _from_ms(run.start_time),
            "end": _from_ms(run.end_time),
            "url": run.run_page_url,
        }
    return out


# COMMAND ----------
# Files added/changed AFTER the last ingest run started - i.e. not yet processed.
# (Approximate: based on modification time, while the job itself detects changes
# by content hash, so a re-uploaded identical file shows here but is skipped.)
def pending_files(files: list[dict], last_run: dict | None) -> list[dict]:
    run = (last_run or {}).get("run")
    if not run or not run.get("start"):
        return list(files)               # never ingested -> everything is pending
    return [f for f in files if f["modified"] and f["modified"] > run["start"]]
