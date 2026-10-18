#!/usr/bin/env python3
"""The pre-push hook's pytest run: all of ``python/tests``, on every core.

Run serially in one process, the suite took 44m29s on a 16-core Apple Silicon
Mac (2026-10-18, 16,834 tests, warm artifact cache), and every push paid it.
Under pytest-xdist with one test file per worker the same run took 4m42s on 12
workers, with the same outcome test for test. Nothing is dropped to get there:
this is still the whole directory, so the hook and python-tests.yml keep
running one suite.

``--dist loadfile`` and not xdist's default per-test ``load``: each file's tests
run in one process, in file order, so a module- or class-scoped fixture is
built once and no test runs after a different predecessor than it does
serially. The price is balance, since the slowest file (about 200 s) is the
floor.

The default is one worker per performance core on Apple Silicon and xdist's
``auto`` (one per CPU) everywhere else. On the same 12+4-core Mac, ``auto``'s 16
workers took 5m27s against 12 workers' 4m42s: a file dealt to an efficiency
core runs slower, and with ``loadfile`` the slowest file sets the finish.
``BNGSIM_PREPUSH_WORKERS`` overrides the count; ``0`` runs the suite serially
in this process, as the hook used to, for a failure that might be about what
ran before it.

A venv provisioned before pytest-xdist joined the ``test`` extra does not have
it, and there ``-n`` is pytest's "unrecognized arguments" error with no hint of
the fix. So without xdist this says how to get it and runs serially, rather
than refusing the push.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

WORKERS_ENV = "BNGSIM_PREPUSH_WORKERS"


def command(have_xdist: bool, workers: str) -> list[str]:
    """The pytest invocation, with or without xdist's arguments."""
    cmd = [sys.executable, "-m", "pytest", "-q", "python/tests"]
    if have_xdist:
        cmd += ["-n", workers, "--dist", "loadfile"]
    return cmd


def default_workers() -> str:
    """One worker per performance core on Apple Silicon, else xdist's ``auto``."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        # Intel Macs have no perflevel0 and print nothing here.
        if out.isdigit() and int(out) > 0:
            return out
    return "auto"


def main() -> int:
    have_xdist = importlib.util.find_spec("xdist") is not None
    workers = os.environ.get(WORKERS_ENV, "").strip() or default_workers()
    if not have_xdist:
        print(
            "pre-push: pytest-xdist is not installed, so the suite runs serially "
            "(about 45 minutes). `uv sync --extra dev` installs it.",
            file=sys.stderr,
            flush=True,
        )
    return subprocess.call(command(have_xdist, workers))


if __name__ == "__main__":
    sys.exit(main())
