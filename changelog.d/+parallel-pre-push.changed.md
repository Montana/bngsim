- **The pre-push hook now runs the test suite on every core, in about five
  minutes instead of about 45.** It still runs all of `python/tests`, now
  under pytest-xdist with one test file per worker (`ci/pre_push_pytest.py`).
  On a 16-core Mac that took the run from 44m29s to 4m42s with the same
  result. pytest-xdist joins the `test` extra, so run
  `uv sync --extra dev` to get it; without it the hook says so and runs
  serially. `BNGSIM_PREPUSH_WORKERS` sets the worker count, and `0` runs the
  suite serially in one process.
