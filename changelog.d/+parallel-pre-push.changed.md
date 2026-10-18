- **The pre-push hook now runs the test suite in parallel, in about five
  minutes instead of about 45.** It still runs all of `python/tests`, now
  under pytest-xdist with one test file per worker (`ci/pre_push_pytest.py`);
  on a 16-core Mac that took the run from 44m29s to 4m42s with the same
  result. From a worktree it uses the main checkout's test cache, because
  macOS checks every new compiled library on first load, one at a time. That
  made a worktree's first push take 19m15s. pytest-xdist joins the `test`
  extra: without it the push stops and says to run `uv sync --extra dev`.
  `BNGSIM_PREPUSH_WORKERS` sets the worker count, and `0` runs serially.
