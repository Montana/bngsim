- **A push that changes only files the test suite never reads no longer runs
  it.** The pre-push hook passes at once when every file the push changes,
  deleted ones included, is under `changelog.d/` or is one of `LICENSE`,
  `NOTICE`, `ACKNOWLEDGMENTS.md`, `CONTRIBUTORS.md`, `REPRODUCING.md`,
  `.readthedocs.yaml` or `.clang-format`; the `changelog-fragments` hook still
  checks the fragments. Docs still run it, because the suite runs their
  examples. A push that changes only `third_party/`, or only deletes files,
  now runs it too, where it used to skip.
