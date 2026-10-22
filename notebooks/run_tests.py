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

PROJECT_ROOT = dbutils.widgets.get("project_root").strip()
# Must be set, or we cannot find src/ and tests/.
assert PROJECT_ROOT, "project_root must be set (the Job passes ${workspace.file_path})"
print("PROJECT_ROOT:", PROJECT_ROOT)

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

import pytest

tests_dir = os.path.join(WORK, "tests")
print("running:", tests_dir)

# Capture pytest's own output so we can see it (and surface it on failure), since
# the Job run output only shows the raised exception, not the cell's stdout.
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    exit_code = int(pytest.main(["-rA", "-p", "no:cacheprovider", "--rootdir", WORK, tests_dir]))
pytest_output = _buf.getvalue()
print(pytest_output)                    # show full output in the cell
print("pytest exit code:", exit_code)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 - Fail the task if any test failed

# COMMAND ----------

# A non-zero exit code means tests failed or could not be collected -> fail the task.
# Include the tail of pytest's output in the error so it shows in the Job run output.
if exit_code != 0:
    tail = "\n".join(pytest_output.splitlines()[-40:])
    raise AssertionError(f"pytest failed (exit code {exit_code})\n\n{tail}")
# Report success back to the Job run output.
dbutils.notebook.exit(f"All tests passed (pytest exit {exit_code}).")
