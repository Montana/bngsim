"""GH #570 — an empty output schedule must be refused, not walked off the end.

``NfsimSimulator::run()`` and ``::simulate()`` both size a ``Result`` by the
number of output points and then record the initial row unconditionally: they
read ``t_out[0]`` and write ``record(0, ...)`` before the loop over the
remaining points is ever entered. With ``n_points <= 0`` there is no
``t_out[0]`` and no row 0, so both ran past the end of a zero-row buffer and the
process died with a segmentation fault — exit 139, not a Python exception, so
nothing downstream could catch it or say what had gone wrong. It reached the
public ``bngsim.NfsimSession`` API, not only the internal bindings.

The sibling RuleMonkey backend has refused this since it was written
(``rulemonkey_interval_count``: "n_points must be positive"). Both NFsim entry
points now say the same thing, naming the count they were given, and
``run()`` checks before it builds the NFsim ``System`` so nothing is
constructed or leaked on the way out.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False),
    reason="NFsim not compiled in",
)

BAD = [0, -1, -7]


# ── The session path (the public API the issue reports) ──────────────────────


@pytest.mark.parametrize("n_points", BAD)
def test_session_simulate_refuses_an_empty_schedule(nfsim_xml: Path, n_points: int):
    """``s.simulate(0.0, 10.0, 0)`` used to take the interpreter down with it."""
    with bngsim.NfsimSession(str(nfsim_xml)) as session:
        session.initialize(42)
        with pytest.raises(Exception, match="n_points must be positive"):
            session.simulate(0.0, 10.0, n_points)


def test_the_session_message_names_the_count(nfsim_xml: Path):
    with bngsim.NfsimSession(str(nfsim_xml)) as session:
        session.initialize(42)
        with pytest.raises(Exception, match=r"got 0"):
            session.simulate(0.0, 10.0, 0)


# ── The stateless path ───────────────────────────────────────────────────────


@pytest.mark.parametrize("n_points", BAD)
def test_run_refuses_an_empty_schedule(nfsim_xml: Path, n_points: int):
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 10.0
    ts.n_points = n_points
    with pytest.raises(RuntimeError, match="n_points must be positive"):
        NfsimSimulator(str(nfsim_xml)).run(ts, 42, 0.0)


# ── The schedules that must keep working ─────────────────────────────────────


@pytest.mark.parametrize("n_points", [1, 2, 5])
def test_a_positive_schedule_still_runs(nfsim_xml: Path, n_points: int):
    """One point is the boundary the guard sits next to: it records the initial
    row and nothing else, which is a legitimate request."""
    with bngsim.NfsimSession(str(nfsim_xml)) as session:
        session.initialize(42)
        result = session.simulate(0.0, 10.0, n_points)
    assert result.observables.shape[0] == n_points


def test_explicit_sample_times_are_unaffected(nfsim_xml: Path):
    """A non-empty sample_times list supplies its own schedule, so the guard
    counts those instants rather than n_points."""
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 10.0
    ts.n_points = 0  # ignored: sample_times wins
    ts.sample_times = [0.0, 2.5, 7.5]
    assert NfsimSimulator(str(nfsim_xml)).run(ts, 42, 0.0).n_times == 3
