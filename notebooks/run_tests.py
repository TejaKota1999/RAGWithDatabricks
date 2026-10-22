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


# A tiny pytest plugin to record how many tests were collected and how many failed.
class _Stats:
    collected = 0
    failed = 0

    def pytest_sessionfinish(self, session, exitstatus):
        self.collected = int(getattr(session, "testscollected", 0))
        self.failed = int(getattr(session, "testsfailed", 0))


stats = _Stats()
tests_dir = os.path.join(WORK, "tests")
print("running:", tests_dir)

# Capture pytest's own output so we can see it (and surface it on failure), since
# the Job run output only shows the raised exception, not the cell's stdout.
_buf = io.StringIO()
_start = time.time()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    exit_code = int(pytest.main(
        ["-rA", "-p", "no:cacheprovider", "--rootdir", WORK, tests_dir],
        plugins=[stats],
    ))
duration_sec = round(time.time() - _start, 3)
pytest_output = _buf.getvalue()
print(pytest_output)                    # show full output in the cell

# Derive the headline numbers for the audit log.
tests_total = stats.collected
tests_failed = stats.failed
tests_passed = max(tests_total - tests_failed, 0)
status = "SUCCESS" if exit_code == 0 else "FAILURE"
# The last non-empty line of pytest output is its summary (e.g. "46 passed in 1.2s").
# Strip ANSI colour codes so the stored text is clean.
import re as _re

_ansi = _re.compile(r"\x1b\[[0-9;]*m")
summary_lines = [ln for ln in pytest_output.splitlines() if ln.strip()]
summary = _ansi.sub("", summary_lines[-1])[:500] if summary_lines else ""
print(f"status={status} total={tests_total} passed={tests_passed} "
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

# Describe the audit table's columns (created automatically on first write).
log_schema = StructType(
    [
        StructField("run_timestamp", StringType(), False),   # when the run happened (UTC)
        StructField("status", StringType(), False),          # SUCCESS or FAILURE
        StructField("exit_code", IntegerType(), False),      # pytest exit code (0 = all passed)
        StructField("tests_total", IntegerType(), False),    # number of tests collected
        StructField("tests_passed", IntegerType(), False),
        StructField("tests_failed", IntegerType(), False),
        StructField("duration_sec", DoubleType(), False),    # how long the suite took
        StructField("summary", StringType(), True),          # pytest's summary line
        StructField("job_run_id", StringType(), True),       # Databricks Job run id (if known)
    ]
)

# Build the single audit row for this run.
log_row = [
    {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "exit_code": exit_code,
        "tests_total": tests_total,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "duration_sec": float(duration_sec),
        "summary": summary,
        "job_run_id": str(job_run_id) if job_run_id is not None else None,
    }
]

# Append the row to the Delta table (creates it if it does not exist yet).
spark.createDataFrame(log_row, schema=log_schema).write.mode("append").saveAsTable(TEST_LOG_FQN)
print(f"Logged test run to {TEST_LOG_FQN}: status={status}, "
      f"passed={tests_passed}/{tests_total}, exit={exit_code}")

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
