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

Two throws exercise it. The first is an allocation in ``result.allocate()``,
measured in a subprocess under an ``RLIMIT_AS`` cap so it fails at a fixed size
rather than depending on how much memory the machine has — Linux only, for
``/proc`` and the address-space limit. The second is NFsim's own setup failing
on a model whose function names a missing observable: deterministic on every
platform, and the only path where a *half-prepared* System is freed.
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


# ── A throw from inside NFsim's own setup, on every platform ─────────────────
#
# The probe above throws from result.allocate(), after prepare_system() has
# finished. prepare_system() is the other way out of the window, and the one a
# user actually reaches: NFsim parses the model, then throws "Quitting" part-way
# through System::prepareForSimulation(). The fixture's global function names an
# observable that does not exist, which gets there cheaply and deterministically
# on every platform — no oversized allocation, no address-space cap. It is also
# the one path where the unique_ptr frees a System that was only half prepared;
# before GH #577 that System was leaked, so nothing had ever freed one.

_SETUP_FAILURE_CALLS = 200

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-specific: peak RSS comes from resource.getrusage, which Windows lacks",
)

_SETUP_FAILURE_PROBE = f"""
import resource, sys
from bngsim._bngsim_core import NfsimSimulator, TimeSpec

def peak_bytes():
    kb_or_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_b if sys.platform == "darwin" else kb_or_b * 1024

ts = TimeSpec()
ts.t_start, ts.t_end, ts.n_points = 0.0, 1.0, 3
sim = NfsimSimulator(sys.argv[1])

def failing_call():
    try:
        sim.run(ts, 42, 0.0)
    except RuntimeError:
        return
    raise AssertionError("run() did not raise on the setup-failure fixture")

for _ in range(20):  # let the allocator and NFsim's statics settle first
    failing_call()
before = peak_bytes()
for _ in range({_SETUP_FAILURE_CALLS}):
    failing_call()
print(peak_bytes() - before)
"""


@pytest.fixture
def setup_failure_xml(data_dir: Path) -> Path:
    return data_dir / "nfsim" / "setup_failure_unknown_observable.xml"


def _three_points():
    from bngsim._bngsim_core import TimeSpec

    ts = TimeSpec()
    ts.t_start, ts.t_end, ts.n_points = 0.0, 1.0, 3
    return ts


def test_a_setup_failure_still_raises_every_time(setup_failure_xml: Path):
    """Freeing the half-prepared System must not swallow the error that freed
    it, nor leave anything behind that changes the next call."""
    from bngsim._bngsim_core import NfsimSimulator

    sim = NfsimSimulator(str(setup_failure_xml))
    for _ in range(_SETUP_FAILURE_CALLS):
        with pytest.raises(RuntimeError, match="NFsim setup failed"):
            sim.run(_three_points(), 42, 0.0)


def test_a_simulator_still_runs_after_a_setup_failure(
    setup_failure_xml: Path, nfsim_funccols_xml: Path
):
    """The same model with the reference intact runs, after a string of failed
    setups in the same process: the teardown corrupted nothing."""
    from bngsim._bngsim_core import NfsimSimulator

    failing = NfsimSimulator(str(setup_failure_xml))
    for _ in range(20):
        with pytest.raises(RuntimeError):
            failing.run(_three_points(), 42, 0.0)
    assert NfsimSimulator(str(nfsim_funccols_xml)).run(_three_points(), 42, 0.0).n_times == 3


@_POSIX_ONLY
def test_a_setup_failure_does_not_retain_the_half_prepared_system(setup_failure_xml: Path):
    """Peak RSS is a ceiling, not a live count, but a leak can only raise it —
    and in a fresh process past a warm-up there is nothing else to. Before the
    fix it grew ~27 MB over these calls on macOS (~150 KB per call, this small
    model's System); after it, not at all."""
    proc = subprocess.run(
        [sys.executable, "-c", _SETUP_FAILURE_PROBE, str(setup_failure_xml)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    grew = float(proc.stdout.strip().splitlines()[-1])
    assert grew < 4 * 1024 * 1024, (
        f"peak RSS grew {grew / 1e6:.1f} MB over {_SETUP_FAILURE_CALLS} setup-failure "
        "calls; the half-prepared System is not being freed"
    )


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
