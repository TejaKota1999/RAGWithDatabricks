"""'How it works' tab: a live, three-column flow diagram of the RAG pipeline.

Layout mirrors a  TRIGGER  ->  PIPELINE STEPS  ->  DATA / SERVICES  view, for
the two flows in this project:
  1) Ingestion - what happens when a new file lands in the source-docs volume.
  2) Question  - what happens when you ask the chatbot something (traced from
                 your LAST question in this session: real timings, scores, ids).

Live numbers come from src.serving.flow_status (cheap metadata reads only, cached
for 60 s). Nothing here calls a model or the SQL warehouse.
"""

from __future__ import annotations

import html
import os
from datetime import datetime, timezone

import streamlit as st

from src.serving import flow_status
from src.serving.databricks_client import get_workspace_client

# Where to look for live status (overridable per environment via app.yaml).
VOLUME_PATH = os.environ.get("RAG_VOLUME_PATH", "/Volumes/databricks_rag_dev/rag/source_docs")
INGEST_JOB_NAME = os.environ.get("RAG_INGEST_JOB_NAME", "RAG Ingest (AI Functions)")

# Badge colors per status (chosen to read on both light and dark themes).
_TONES = {
    "ok": "#16a34a",       # done / ready
    "warn": "#d97706",     # waiting / needs action
    "run": "#2563eb",      # in progress
    "err": "#dc2626",      # failed / unavailable
    "idle": "#6b7280",     # not run / skipped
}

_CSS = """
<style>
.rf-wrap{display:grid;grid-template-columns:minmax(150px,1fr) minmax(230px,1.7fr) minmax(190px,1.3fr);
  gap:14px 28px;align-items:start;margin:4px 0 18px}
.rf-head{font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;opacity:.6;font-weight:600}
.rf-col{display:flex;flex-direction:column;gap:10px;position:relative}
.rf-card{border:1px solid rgba(127,127,127,.28);border-radius:10px;padding:10px 12px;
  background:rgba(127,127,127,.06);position:relative;font-size:.86rem;line-height:1.35;
  overflow-wrap:anywhere;min-width:0}
.rf-card.rf-lit{border-color:var(--tone);box-shadow:0 0 0 1px var(--tone) inset}
.rf-trigger{border:2px solid #6366f1;background:rgba(99,102,241,.10)}
.rf-title{font-weight:650;font-size:.92rem;margin-bottom:2px}
.rf-sub{opacity:.68;font-size:.76rem}
.rf-stat{margin-top:6px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.76rem}
.rf-num{display:inline-block;min-width:20px;height:20px;border-radius:50%;background:#6366f1;color:#fff;
  font-size:.72rem;font-weight:700;text-align:center;line-height:20px;margin-right:6px}
.rf-badge{float:right;font-size:.64rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;
  color:#fff;background:var(--tone);border-radius:4px;padding:1px 6px;margin-left:6px}
.rf-chip{display:inline-block;margin:6px 4px 0 0;padding:1px 7px;border-radius:4px;font-size:.7rem;
  font-family:ui-monospace,Menlo,Consolas,monospace;border:1px solid rgba(127,127,127,.35)}
.rf-down{text-align:center;opacity:.45;font-size:.9rem;line-height:.6}
.rf-col.rf-mid::before,.rf-col.rf-right::before{content:"\\2192";position:absolute;left:-22px;top:28px;
  opacity:.45;font-size:1.1rem}
.rf-run .rf-badge{animation:rf-pulse 1.4s ease-in-out infinite}
@keyframes rf-pulse{50%{opacity:.45}}
@media (max-width:720px){.rf-wrap{grid-template-columns:1fr}.rf-col.rf-mid::before,
  .rf-col.rf-right::before{content:none}.rf-head{margin-top:6px}}
</style>
"""


# --------------------------------------------------------------------------
# Small HTML builders (every dynamic string is escaped).
def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _card(title, sub="", stat="", tone=None, badge=None, num=None, chips=(), extra_cls=""):
    cls = f"rf-card {extra_cls}"
    style = ""
    if tone:
        cls += " rf-lit" + (" rf-run" if tone == "run" else "")
        style = f' style="--tone:{_TONES[tone]}"'
    badge_html = f'<span class="rf-badge">{_e(badge)}</span>' if badge and tone else ""
    num_html = f'<span class="rf-num">{num}</span>' if num else ""
    chips_html = "".join(f'<span class="rf-chip">{_e(c)}</span>' for c in chips)
    return (
        f'<div class="{cls}"{style}>{badge_html}<div class="rf-title">{num_html}{_e(title)}</div>'
        f'<div class="rf-sub">{_e(sub)}</div>'
        + (f'<div class="rf-stat">{stat}</div>' if stat else "")
        + chips_html + "</div>"
    )


def _steps(cards: list[str]) -> str:
    return '<div class="rf-down">&#8595;</div>'.join(cards)


def _diagram(left: list[str], mid: list[str], right: list[str]) -> str:
    return (
        _CSS + '<div class="rf-wrap">'
        f'<div class="rf-col"><div class="rf-head">Trigger</div>{"".join(left)}</div>'
        f'<div class="rf-col rf-mid"><div class="rf-head">Pipeline steps</div>{_steps(mid)}</div>'
        f'<div class="rf-col rf-right"><div class="rf-head">Data &amp; services</div>{"".join(right)}</div>'
        "</div>"
    )


def _ago(dt: datetime | None) -> str:
    if not dt:
        return "never"
    secs = int((datetime.now(timezone.utc) - dt).total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{secs // size}{unit} ago"
    return f"{max(secs, 0)}s ago"


# --------------------------------------------------------------------------
# Live status, cached so switching tabs / reruns do not re-hit the APIs.
@st.cache_data(ttl=60, show_spinner="Reading live pipeline status ...")
def _live_status(profile: str | None, index_name: str) -> dict:
    ws = get_workspace_client(profile)
    vol = flow_status.volume_files(ws, VOLUME_PATH)
    job = flow_status.last_job_run(ws, INGEST_JOB_NAME)
    idx = flow_status.index_status(ws, index_name)
    pending = flow_status.pending_files(vol.get("files", []), job) if "error" not in vol else []
    return {"volume": vol, "job": job, "index": idx, "pending": pending,
            "read_at": datetime.now(timezone.utc)}


# --------------------------------------------------------------------------
# Flow 1: a new file lands in the volume.
def _ingest_flow(s: dict, chat_endpoint: str) -> str:
    vol, job, idx, pending = s["volume"], s["job"], s["index"], s["pending"]
    run = job.get("run") if "error" not in job else None

    # Trigger card: the volume and how many files are not yet processed.
    if "error" in vol:
        trig = _card("New file in volume", VOLUME_PATH, _e(vol["error"]), "err", "unavailable", extra_cls="rf-trigger")
    else:
        n, p = len(vol["files"]), len(pending)
        stat = f"{n} file(s) &middot; {p} new since last run"
        if p:
            stat += "<br>" + "<br>".join(f"&#9679; {_e(f['name'])}" for f in pending[:4])
        trig = _card("New file in volume", "Upload a .docx / .pdf to source_docs", stat,
                     "warn" if p else "ok", "pending" if p else "up to date", extra_cls="rf-trigger")

    # Step 1: the job run itself drives the badge for steps 1-3.
    if "error" in job:
        jt, jb, js = "err", "unavailable", _e(job["error"])
    elif not run:
        jt, jb, js = "idle", "never run", "No runs yet"
    elif run["life_cycle"] in ("PENDING", "QUEUED", "RUNNING", "BLOCKING", "TERMINATING"):
        jt, jb, js = "run", "running", f"started {_ago(run['start'])}"
    elif run["result"] == "SUCCESS":
        jt, jb, js = "ok", "success", f"last run {_ago(run['end'] or run['start'])}"
    else:
        jt, jb, js = "err", (run["result"] or run["life_cycle"] or "failed").lower(), f"last run {_ago(run['start'])}"
    if pending and jt == "ok":
        jt, jb = "warn", "run needed"
        js += " &middot; new files not processed yet"
    stage_tone = jt if jt in ("run", "err") else ("ok" if jt == "ok" else None)

    mid = [
        _card("Run ingest job", job.get("job_name") or "rag_ingest_ai (Lakeflow Job)",
              js, jt, jb, num=1, chips=["serverless", "incremental=true"]),
        _card("ai_parse_document", "Extracts text + layout from each NEW/CHANGED file (keyed on content_hash)",
              tone=stage_tone, badge=jb if stage_tone else None, num=2, chips=["reads: source_docs"]),
        _card("ai_prep_search", "Cleans and chunks the parsed text, upserts by document",
              tone=stage_tone, badge=jb if stage_tone else None, num=3, chips=["writes: document_chunks_ai"]),
        _card("Vector Search sync", "Index embeds chunk_to_embed with the managed embedding endpoint",
              f"{idx.get('row_count')} chunks in index" if "error" not in idx else _e(idx["error"]),
              ("ok" if idx.get("ready") else "warn") if "error" not in idx else "err",
              ("ready" if idx.get("ready") else "syncing") if "error" not in idx else "unavailable",
              num=4, chips=["writes: " + idx.get("name", "index").split(".")[-1]]),
    ]

    right = [
        _card("source_docs volume", VOLUME_PATH,
              f"{len(vol['files'])} file(s)" if "error" not in vol else "unavailable"),
        _card("document_chunks_ai", "Delta table (Unity Catalog)",
              _e(idx.get("source_table") or "databricks_rag_dev.rag.document_chunks_ai")),
        _card(idx.get("name", "Vector Search index").split(".")[-1],
              f"Vector Search · endpoint {idx.get('endpoint') or '?'}" if "error" not in idx else "Vector Search",
              f"{idx.get('row_count')} chunks indexed" if "error" not in idx else "unavailable",
              ("ok" if idx.get("ready") else "warn") if "error" not in idx else "err",
              ("ready" if idx.get("ready") else "not ready") if "error" not in idx else "error"),
        _card(idx.get("embedding_endpoint") or "embedding endpoint", "Foundation Model API (managed embeddings)"),
    ]
    return _diagram([trig], mid, right)


# --------------------------------------------------------------------------
# Flow 2: a question from the chatbot (traced from the last question asked).
def _question_flow(s: dict, trace: dict | None, chat_endpoint: str, top_k: int, min_score: float) -> str:
    idx = s["index"]
    idx_short = idx.get("name", "index").split(".")[-1] if "error" not in idx else "Vector Search index"

    if not trace:
        trig = _card("You ask a question", "Ask something in the Chat tab to trace it here",
                     extra_cls="rf-trigger")
        mid = [
            _card("Condense follow-up", "Rewrites a follow-up into a standalone question (only if there is history)", num=1,
                  chips=[chat_endpoint]),
            _card("Vector Search query", f"Query by TEXT, top_k={top_k}, min_score={min_score}", num=2,
                  chips=["reads: " + idx_short]),
            _card("Evidence check", "No chunk above min_score -> abstain (no model call)", num=3),
            _card("Generate answer", "Grounded prompt: question + retrieved chunks + recent turns", num=4,
                  chips=[chat_endpoint]),
        ]
    else:
        d = trace["diagnostics"]
        t = d.get("timings") or {}
        hits = trace["sources"]
        trig = _card("You asked", "Last question in this session",
                     f"&ldquo;{_e(trace['question'][:160])}&rdquo;", "ok", "traced", extra_cls="rf-trigger")
        if d.get("condensed"):
            c1 = _card("Condense follow-up", "Rewritten into a standalone question for retrieval",
                       f"&rarr; &ldquo;{_e(d.get('search_query', '')[:160])}&rdquo;<br>{t.get('condense_ms')} ms",
                       "ok", "ran", num=1, chips=[chat_endpoint])
        else:
            c1 = _card("Condense follow-up", "First question (no history), so this step is skipped",
                       "0 model calls", "idle", "skipped", num=1)
        top = d.get("top_score")
        c2 = _card("Vector Search query", f"Query by TEXT, top_k={top_k}, min_score={min_score}",
                   f"{d.get('retrieved', 0)} chunk(s) kept &middot; top score "
                   f"{top:.3f}" if top is not None else f"{d.get('retrieved', 0)} chunk(s) kept",
                   "ok" if hits else "warn", f"{t.get('retrieve_ms')} ms", num=2, chips=["reads: " + idx_short])
        if hits:
            c3 = _card("Evidence check", "At least one chunk above min_score, so continue",
                       tone="ok", badge="pass", num=3)
        else:
            c3 = _card("Evidence check", "No chunk above min_score, so ABSTAIN (no model call)",
                       tone="warn", badge="abstain", num=3)
        if trace.get("error"):
            c4 = _card("Generate answer", "Chat endpoint call failed", _e(trace["error"]), "err", "error",
                       num=4, chips=[chat_endpoint])
        elif t.get("generate_ms") is None:
            c4 = _card("Generate answer", "Skipped because the chain abstained", tone="idle", badge="skipped", num=4)
        else:
            files = sorted({os.path.basename(h.get("source_path") or "") for h in hits} - {""})
            c4 = _card("Generate answer", "Answer grounded in the retrieved chunks",
                       "sources: " + ", ".join(_e(f) for f in files) if files else "",
                       "ok", f"{t['generate_ms']} ms", num=4, chips=[chat_endpoint])
        mid = [c1, c2, c3, c4]

    right = [
        _card(idx_short, "Vector Search (managed embeddings)",
              f"{idx.get('row_count')} chunks" if "error" not in idx else "unavailable"),
        _card(idx.get("embedding_endpoint") or "embedding endpoint",
              "Embeds the question text inside Vector Search (same model as the chunks)"),
        _card(chat_endpoint, "Foundation Model API (chat), pay-per-token"),
    ]
    return _diagram([trig], mid, right)


# --------------------------------------------------------------------------
# Entry point called from chatbot.py inside the "How it works" tab.
def render(profile: str | None, index_name: str, chat_endpoint: str, top_k: int, min_score: float) -> None:
    head, btn = st.columns([5, 1])
    with head:
        st.caption("Live view of the two flows in this app. Status reads are cached for 60 s.")
    with btn:
        if st.button("Refresh", key="flow_refresh"):
            _live_status.clear()

    s = _live_status(profile, index_name)

    st.subheader("1 · When you add a file to the volume")
    st.html(_ingest_flow(s, chat_endpoint))

    st.subheader("2 · When you ask a question")
    trace = st.session_state.get("last_trace")
    st.html(_question_flow(s, trace, chat_endpoint, top_k, min_score))

    # The exact chunks the last question retrieved (what the model saw).
    if trace and trace["sources"]:
        st.markdown("**Chunks retrieved for the last question**")
        st.dataframe(
            [
                {
                    "rank": i + 1,
                    "source file": os.path.basename(h.get("source_path") or ""),
                    "chunk_id": h.get("chunk_id"),
                    "score": round(h["score"], 4) if h.get("score") is not None else None,
                }
                for i, h in enumerate(trace["sources"])
            ],
            hide_index=True,
        )

    st.caption(f"Status read at {s['read_at']:%Y-%m-%d %H:%M:%S} UTC")
