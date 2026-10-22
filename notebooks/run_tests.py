# Databricks notebook source
# MAGIC %md
# MAGIC # Run unit tests (pytest) on serverless
# MAGIC
# MAGIC Runs the project's `tests/` suite inside Databricks on serverless compute.
# MAGIC The tests mock the Databricks clients, so they need no endpoints or quota.
# MAGIC
# MAGIC This notebook is driven by the `rag_tests` Job in `databricks.yml`. It
# MAGIC receives `project_root` (the bundle's synced files path) so it can import
# MAGIC `src` and collect the tests. It fails the task if any test fails.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 - Parameter: where the project files live

# COMMAND ----------

# The Job passes ${workspace.file_path} here (e.g. /Workspace/Users/<you>/.bundle/rag-poc/dev/files).
dbutils.widgets.text("project_root", "", "Project root (bundle files path)")
# Where to write the test-run audit log (a Delta table in the rag schema).
dbutils.widgets.text("catalog", "databricks_rag_dev", "Catalog")
dbutils.widgets.text("schema", "rag", "Schema")
dbutils.widgets.text("test_log_table", "test_log", "Test log table")

PROJECT_ROOT = dbutils.widgets.get("project_root").strip()
CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
TEST_LOG_TABLE = dbutils.widgets.get("test_log_table").strip()
# Full name of the audit table, e.g. databricks_rag_dev.rag.test_log
TEST_LOG_FQN = f"{CATALOG}.{SCHEMA}.{TEST_LOG_TABLE}"
# Must be set, or we cannot find src/ and tests/.
assert PROJECT_ROOT, "project_root must be set (the Job passes ${workspace.file_path})"
print("PROJECT_ROOT:", PROJECT_ROOT)
print("TEST_LOG_FQN:", TEST_LOG_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 - Install test dependencies (in-process, no restart)
# MAGIC Only what the tests import: pytest, plus faiss/docx/pypdf used by a few tests.

# COMMAND ----------

# subprocess installs packages into the current Python without restarting it.
import subprocess
import sys

# Pinned versions so test runs are reproducible.
for pkg in ["pytest==8.3.3", "faiss-cpu==1.9.0", "python-docx==1.1.2", "pypdf==5.1.0"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
print("test dependencies installed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 - Run pytest against the tests folder

# COMMAND ----------

# os/shutil/tempfile for copying; sys.path so `import src...` works.
import os
import shutil
import tempfile

# The Workspace files mount does NOT allow writing __pycache__, which pytest's
# assertion rewriting needs. So copy src/ and tests/ to a writable temp dir and
# run the suite from there. tempfile picks a location the runtime allows us to write.
WORK = tempfile.mkdtemp(prefix="ragtests_")
for sub in ("src", "tests"):
    shutil.copytree(os.path.join(PROJECT_ROOT, sub), os.path.join(WORK, sub))
# Run from the writable copy and make `import src...` resolve there.
os.chdir(WORK)
if WORK not in sys.path:
    sys.path.insert(0, WORK)
print("copied src/ and tests/ to", WORK)

# pytest.main runs the suite in-process and returns an exit code (0 = all passed).
import contextlib
import io
import time

import pytest


# Helper to strip ANSI colour codes so stored text is clean.
import re as _re

_ANSI = _re.compile(r"\x1b\[[0-9;]*m")


def _clean(text):
    return _ANSI.sub("", text or "")


# A pytest plugin that records EACH test: its description (docstring or a
# humanized name), pass/fail status, duration, and any failure message.
class _PerTest:
    def __init__(self):
        self.desc = {}   # nodeid -> one-line description
        self.rows = {}   # nodeid -> (STATUS, duration_sec, message)

    # Called once after collection: grab each test's docstring as its description.
    def pytest_collection_modifyitems(self, items):
        for it in items:
            doc = ""
            try:
                fn = getattr(it, "obj", None)
                doc = (fn.__doc__ or "").strip() if fn is not None else ""
            except Exception:
                doc = ""
            first = doc.splitlines()[0].strip() if doc else ""
            if not first:
                # No docstring -> make a readable description from the test name,
                # e.g. "test_stable_chunk_id" -> "stable chunk id".
                nm = it.name[5:] if it.name.startswith("test_") else it.name
                first = nm.replace("_", " ").strip()
            self.desc[it.nodeid] = first

    # Called for each test phase: record the outcome of the "call" phase (or an
    # error/skip during setup) so we know PASSED / FAILED / SKIPPED per test.
    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            msg = "" if report.outcome == "passed" else _clean(str(report.longrepr))[:1000]
            self.rows[report.nodeid] = (report.outcome.upper(), round(report.duration, 4), msg)
        elif report.when == "setup" and report.outcome in ("skipped", "failed") \
                and report.nodeid not in self.rows:
            self.rows[report.nodeid] = (report.outcome.upper(), round(report.duration, 4),
                                        _clean(str(report.longrepr))[:1000])


pert = _PerTest()
tests_dir = os.path.join(WORK, "tests")
print("running:", tests_dir)

# Capture pytest's own output so we can see it (and surface it on failure), since
# the Job run output only shows the raised exception, not the cell's stdout.
_buf = io.StringIO()
_start = time.time()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    exit_code = int(pytest.main(
        ["-rA", "-p", "no:cacheprovider", "--rootdir", WORK, tests_dir],
        plugins=[pert],
    ))
duration_sec = round(time.time() - _start, 3)
pytest_output = _buf.getvalue()
print(pytest_output)                    # show full output in the cell

# Derive the headline numbers from the per-test results.
tests_total = len(pert.desc) if pert.desc else len(pert.rows)
tests_failed = sum(1 for (st, _d, _m) in pert.rows.values() if st == "FAILED")
tests_passed = sum(1 for (st, _d, _m) in pert.rows.values() if st == "PASSED")
run_status = "SUCCESS" if exit_code == 0 else "FAILURE"
summary_lines = [ln for ln in pytest_output.splitlines() if ln.strip()]
summary = _clean(summary_lines[-1])[:500] if summary_lines else ""
print(f"run_status={run_status} total={tests_total} passed={tests_passed} "
      f"failed={tests_failed} exit={exit_code} duration={duration_sec}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Write the result to the audit log (ALWAYS, pass or fail)
# MAGIC Appends one row to the `test_log` Delta table before failing on errors, so
# MAGIC every run is recorded whether it succeeded or failed.

# COMMAND ----------

# datetime for the timestamp; the pyspark types describe the audit table.
from datetime import datetime, timezone

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Best-effort: capture the Job run id from the notebook context (for traceability).
job_run_id = None
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _tags = _ctx.tags()
    for _key in ("multitaskParentRunId", "jobRunId", "runId"):
        _opt = _tags.get(_key)
        if _opt.isDefined():
            job_run_id = _opt.get()
            break
except Exception as _e:
    print("could not read job run id:", _e)

# uuid groups all the per-test rows that belong to one run.
import uuid

RUN_ID = str(uuid.uuid4())
RUN_TS = datetime.now(timezone.utc).isoformat()

# One row PER TEST: name, description, status, duration, and any failure message.
log_schema = StructType(
    [
        StructField("run_id", StringType(), False),        # groups the tests of one run
        StructField("run_timestamp", StringType(), False), # when the run happened (UTC)
        StructField("run_status", StringType(), False),    # overall SUCCESS / FAILURE
        StructField("test_file", StringType(), False),     # e.g. test_chunker.py
        StructField("test_name", StringType(), False),     # e.g. test_stable_chunk_id
        StructField("description", StringType(), True),     # docstring or humanized name
        StructField("status", StringType(), False),        # PASSED / FAILED / SKIPPED
        StructField("duration_sec", DoubleType(), False),  # this test's runtime
        StructField("message", StringType(), True),         # failure detail (empty if passed)
        StructField("job_run_id", StringType(), True),      # Databricks Job run id (if known)
    ]
)

# Build one row for every collected test (fall back to any that ran but weren't
# in collection, e.g. on a collection error).
rows = []
nodeids = list(pert.desc.keys()) or list(pert.rows.keys())
for nodeid in nodeids:
    st, dur, msg = pert.rows.get(nodeid, ("NOTRUN", 0.0, ""))
    test_file = nodeid.split("::", 1)[0].replace("\\", "/").split("/")[-1]
    test_name = nodeid.split("::")[-1]
    rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TS,
            "run_status": run_status,
            "test_file": test_file,
            "test_name": test_name,
            "description": pert.desc.get(nodeid, ""),
            "status": st,
            "duration_sec": float(dur),
            "message": msg,
            "job_run_id": str(job_run_id) if job_run_id is not None else None,
        }
    )

# If nothing was collected (e.g. a collection/import error), still log one row so
# the failure is never silent.
if not rows:
    rows.append(
        {
            "run_id": RUN_ID, "run_timestamp": RUN_TS, "run_status": run_status,
            "test_file": "(collection)", "test_name": "(no tests collected)",
            "description": "pytest could not collect tests", "status": "ERROR",
            "duration_sec": float(duration_sec), "message": summary,
            "job_run_id": str(job_run_id) if job_run_id is not None else None,
        }
    )

# Append all per-test rows to the Delta table (creates it if it does not exist).
spark.createDataFrame(rows, schema=log_schema).write.mode("append").saveAsTable(TEST_LOG_FQN)
print(f"Logged {len(rows)} test row(s) to {TEST_LOG_FQN} "
      f"(run_id={RUN_ID}, run_status={run_status}, passed={tests_passed}/{tests_total})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 - Fail the task if any test failed
# MAGIC Runs AFTER logging, so the audit row is written even on failure.

# COMMAND ----------

# A non-zero exit code means tests failed or could not be collected -> fail the task.
# Include the tail of pytest's output in the error so it shows in the Job run output.
if exit_code != 0:
    tail = "\n".join(pytest_output.splitlines()[-40:])
    raise AssertionError(f"pytest failed (exit code {exit_code})\n\n{tail}")
# Report success back to the Job run output.
dbutils.notebook.exit(f"All tests passed (pytest exit {exit_code}).")
