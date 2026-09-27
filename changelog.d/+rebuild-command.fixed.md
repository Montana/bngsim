- **The stale-extension message now gives a rebuild command that uses the
  project venv.** The import banner, the missing-feature note and the docs said
  `python scripts/rebuild_editable.py`. That fails where no `python` is on
  `PATH`, and otherwise rebuilds for whichever Python it finds. They now say
  `uv run --no-sync python scripts/rebuild_editable.py`, where `--no-sync` keeps
  uv from syncing the venv, and possibly reinstalling bngsim, before the rebuild
  starts.
