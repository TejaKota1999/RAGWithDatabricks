"""Unit tests for src.serving.flow_status (the app's live "How it works" view).

The Databricks SDK client is replaced with FAKE objects, so these tests do NOT
hit a live workspace. They check response parsing and that failures become an
{"error": ...} dict instead of an exception.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.serving import flow_status

# 2026-01-01 00:00:00 UTC in epoch milliseconds.
_T0 = 1767225600000


def _file(name, minutes, is_dir=False):
    return SimpleNamespace(name=name, file_size=10, last_modified=_T0 + minutes * 60_000, is_directory=is_dir)


def _ws(files=None, index=None, jobs=(), runs=(), fail=None):
    def boom(*a, **k):
        raise PermissionError("PERMISSION_DENIED: no access")

    return SimpleNamespace(
        files=SimpleNamespace(list_directory_contents=boom if fail == "files" else (lambda p: iter(files or []))),
        vector_search_indexes=SimpleNamespace(get_index=boom if fail == "index" else (lambda n: index)),
        jobs=SimpleNamespace(list=lambda: iter(jobs), list_runs=lambda job_id, limit: iter(runs)),
    )


# Directories are skipped and files come back newest first, with UTC datetimes.
def test_volume_files_lists_newest_first_and_skips_dirs():
    ws = _ws(files=[_file("a.docx", 0), _file("sub", 5, True), _file("b.docx", 10)])
    out = flow_status.volume_files(ws, "/Volumes/c/s/v")
    assert [f["name"] for f in out["files"]] == ["b.docx", "a.docx"]
    assert out["files"][0]["modified"].tzinfo is timezone.utc


# A permission error is reported, not raised.
def test_volume_files_error_is_returned():
    out = flow_status.volume_files(_ws(fail="files"), "/Volumes/c/s/v")
    assert "PERMISSION_DENIED" in out["error"]


def test_index_status_parses_ready_and_spec():
    idx = SimpleNamespace(
        endpoint_name="rag_poc_vs",
        status=SimpleNamespace(ready=True, indexed_row_count=8, message="ok"),
        delta_sync_index_spec=SimpleNamespace(
            source_table="c.s.document_chunks_ai",
            embedding_source_columns=[SimpleNamespace(embedding_model_endpoint_name="databricks-gte-large-en")],
        ),
    )
    out = flow_status.index_status(_ws(index=idx), "c.s.idx")
    assert out["ready"] is True and out["row_count"] == 8
    assert out["embedding_endpoint"] == "databricks-gte-large-en"
    assert out["source_table"] == "c.s.document_chunks_ai"


def test_index_status_error_is_returned():
    assert "error" in flow_status.index_status(_ws(fail="index"), "c.s.idx")


# The job is matched by substring (bundle deployments prefix the name).
def test_last_job_run_matches_prefixed_name():
    job = SimpleNamespace(job_id=7, settings=SimpleNamespace(name="[dev me] [dev] RAG Ingest (AI Functions)"))
    run = SimpleNamespace(
        state=SimpleNamespace(life_cycle_state=SimpleNamespace(value="TERMINATED"),
                              result_state=SimpleNamespace(value="SUCCESS")),
        start_time=_T0, end_time=_T0 + 1000, run_page_url="https://x/run/1",
    )
    out = flow_status.last_job_run(_ws(jobs=[job], runs=[run]), "RAG Ingest (AI Functions)")
    assert out["job_id"] == 7
    assert out["run"]["result"] == "SUCCESS"
    assert out["run"]["start"] == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_last_job_run_missing_job():
    assert "error" in flow_status.last_job_run(_ws(jobs=[]), "RAG Ingest (AI Functions)")


# Only files modified after the last run started are "pending".
def test_pending_files_after_last_run():
    files = flow_status.volume_files(_ws(files=[_file("old.docx", 0), _file("new.docx", 30)]), "/v")["files"]
    last = {"run": {"start": datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)}}
    assert [f["name"] for f in flow_status.pending_files(files, last)] == ["new.docx"]
    # Never ingested -> everything is pending.
    assert len(flow_status.pending_files(files, {"run": None})) == 2
