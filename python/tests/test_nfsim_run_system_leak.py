"""GH #577 — run() must free the NFsim System on the throwing paths too.

``NfsimSimulator::run()`` held the parsed ``NFcore::System`` in a bare pointer
and freed it in exactly two places: a ``catch`` around the stepping loop, and a
``delete`` before the return. Everything between the System's creation and that
try block was unguarded — ``prepare_system()`` (which raises "NFsim setup
failed" on a model NFsim rejects after parsing, or a failed override bake),
``times.output_times()``, ``result.allocate()``, ``resolve_output_functions()``
— so a throw from any of them walked out of the function and left the whole
System behind.

That is not a handful of bytes: the System holds the parsed model, and the leak
scales with it (~3 KB per call at ``molecule_limit`` 1,000, ~5 MB at 100,000).
A parameter scan that trips ``prepare_system`` repeats it once per scan point.

The System is now owned by a ``unique_ptr`` for the whole function, so every
path out frees it — and the catch-and-rethrow that existed only to run one of
the two deletes is gone with it.

The measurement below runs in a subprocess under an ``RLIMIT_AS`` cap, so the
allocation that triggers the throw fails at a fixed size rather than depending
on how much memory the machine happens to have. Linux only, for ``/proc`` and
the address-space limit; skipped elsewhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import bngsim
import pytest

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False),
    reason="NFsim not compiled in",
)

_LINUX_ONLY = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not Path("/proc/self/statm").exists(),
    reason="needs /proc/self/statm and RLIMIT_AS (Linux)",
)

# Throws inside the window this issue is about: allocate() runs after
# create_system() has parsed the model, and 100M output points x n_obs doubles
# is far past the cap the subprocess sets for itself.
_N_POINTS = 100_000_000
_ADDRESS_SPACE_CAP = 1536 * 1024 * 1024
_CALLS = 40

# A leak of the System is megabytes per call; the baseline a *successful* call
# leaves behind is ~70 KB on this model. Anything under a megabyte says the
# System is not among what stayed.
_MAX_BYTES_PER_CALL = 1024 * 1024

_PROBE = f"""
import os, resource, sys
resource.setrlimit(resource.RLIMIT_AS, ({_ADDRESS_SPACE_CAP},) * 2)
from bngsim._bngsim_core import NfsimSimulator, TimeSpec

def rss():
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

sim = NfsimSimulator(sys.argv[1])

def throwing_call():
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 1.0
    ts.n_points = {_N_POINTS}
    try:
        sim.run(ts, 42, 0.0)
    except MemoryError:
        return True
    return False

# The probe is only meaningful if the call throws where the issue says it does.
assert throwing_call(), "run() did not raise on the oversized allocation"
before = rss()
for _ in range(({_CALLS})):
    throwing_call()
print((rss() - before) / ({_CALLS}))
"""


@_LINUX_ONLY
def test_a_throwing_run_does_not_retain_the_system(nfsim_xml: Path):
    """The defect: ~6 MB per throwing call on this model, unbounded over a
    scan. The System is what that is — the leak tracked molecule_limit."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(nfsim_xml)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    per_call = float(proc.stdout.strip().splitlines()[-1])
    assert per_call < _MAX_BYTES_PER_CALL, (
        f"{per_call / 1024:.0f} KB retained per throwing run() call; "
        "the parsed System is not being freed on the throwing path"
    )


# ── The behaviour around it, which needs no measurement ──────────────────────
#
# These two drive the same oversized allocation, so they run in a child process
# too. In the pytest process they relied on 2e9 points failing to allocate
# immediately; under Linux overcommit a ~16 GB request can instead *succeed*,
# get zero-filled page by page, and take the runner down with it (exit 143 on
# ubuntu-latest-lapack, run 35893818826). In a child, under the same RLIMIT_AS
# cap as the probe above, the allocation fails at a fixed size. Where RLIMIT_AS
# is not enforced (macOS) the child still runs uncapped, as these did before, so
# no platform loses coverage — but the worst case now kills the child, not the
# session.

_CAPPED_PRELUDE = f"""
import resource, sys
if sys.platform.startswith("linux"):
    resource.setrlimit(resource.RLIMIT_AS, ({_ADDRESS_SPACE_CAP},) * 2)
from bngsim._bngsim_core import NfsimSimulator, TimeSpec

def oversized():
    ts = TimeSpec()
    ts.t_start, ts.t_end, ts.n_points = 0.0, 1.0, 2_000_000_000
    return ts

def raises(sim):
    try:
        sim.run(oversized(), 42, 0.0)
    except (MemoryError, RuntimeError, ValueError) as exc:
        return type(exc).__name__
    return None
"""


def _run_capped(body: str, nfsim_xml: Path) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-c", _CAPPED_PRELUDE + body, str(nfsim_xml)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()


def test_the_oversized_run_still_raises(nfsim_xml: Path):
    """Freeing the System must not swallow the error that freed it."""
    out = _run_capped(
        "print(raises(NfsimSimulator(sys.argv[1])))\n",
        nfsim_xml,
    )
    assert out[-1] != "None", "run() returned instead of raising on 2e9 points"


def test_a_simulator_still_runs_after_a_throw(nfsim_xml: Path):
    """The throw leaves nothing half-owned behind it: the same simulator
    parses a fresh System and runs."""
    out = _run_capped(
        "sim = NfsimSimulator(sys.argv[1])\n"
        "print(raises(sim))\n"
        "good = TimeSpec()\n"
        "good.t_start, good.t_end, good.n_points = 0.0, 1.0, 3\n"
        "print(sim.run(good, 42, 0.0).n_times)\n",
        nfsim_xml,
    )
    assert out[-2] != "None", "run() returned instead of raising on 2e9 points"
    assert out[-1] == "3"


def test_the_ordinary_run_is_unchanged(nfsim_xml: Path):
    """Two successive runs from one simulator agree: the System is still built
    per call and freed after it, exactly as before."""
    import numpy as np
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    sim = NfsimSimulator(str(nfsim_xml))
    ts = TimeSpec()
    ts.t_start, ts.t_end, ts.n_points = 0.0, 5.0, 6
    first = np.asarray(sim.run(ts, 7, 0.0).observable_data)
    second = np.asarray(sim.run(ts, 7, 0.0).observable_data)
    assert first.shape == (6, first.shape[1])
    assert np.array_equal(first, second)
