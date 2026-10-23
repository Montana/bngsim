"""Issue #1059 — an SSA run far from t = 0 kept its waiting times.

The SSA placed each firing at ``t + tau`` in doubles. Far from the origin the
doubles near ``t`` are coarse, 1.2e-4 apart at ``t = 1e12`` and 0.125 apart at
1e15, so a waiting time below half that spacing was lost whole and one above
it rounded to a whole spacing. ``A -> B`` at k = 0.5 from 1e5 copies, read at
``t0 + 2`` over ten seeds against the mean 1e5·e⁻¹ = 36788, came out at
z = −289 from ``t0 = 1e12`` and z = −762 from 1e15, with no error.

What a firing rounds away is now carried into the next one, and a sample, an
event or a breakpoint is weighed against the firing's exact time. A run of a
model whose rates do not read the clock, or read it only through a step held
between breakpoints, is then the same run wherever it starts: these hold it to
the run from ``t0 = 0`` seed for seed, to the bit.

Not covered: a rate that reads ``time()`` otherwise, or a rate rule, runs
through the continuous loop, whose panels are placed in absolute time, and
keeps the limit there. So does an event trigger that reads the clock, within
half a spacing of the time it fires at: a firing that rounds onto that time
from before it leaves the clock reading the event's time.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

OFFSETS = [0.5, 1.0, 2.0]


def _decay(t0: float = 0.0, kind: str = "plain", k: float = 0.5, n: float = 1e5) -> bngsim.Model:
    b = ModelBuilder()
    b.add_parameter("k", k)
    a = b.add_species("A", n)
    b.add_species("B", 0.0)
    if kind == "step":
        # A rate that switches off at t0 + 1, held between breakpoints.
        b.add_function("kt", f"if(time() > {t0 + 1.0!r}, 0, k)")
        b.add_reaction([a], [1], "functional", "kt")
    else:
        b.add_reaction([a], [1], "elementary", "k")
    if kind == "watched":
        # An event that never fires puts the run on the loop that watches
        # triggers, rather than the plain one.
        c = b.add_species("C", 0.0)
        b.add_event("late", "time() >= 1e300", [(c, "1")], initial_value=False)
    elif kind == "dose":
        b.add_event("dose", f"time() >= {t0 + 1.0!r}", [(a, "A + 50000")], initial_value=False)
    return bngsim.Model(_core=b.build())


def _run(model, t0, method, seed):
    kw = {"poplevel": 100.0} if method == "psa" else {}
    result = bngsim.Simulator(model, method=method, **kw).run(
        sample_times=[t0, *(t0 + d for d in OFFSETS)], seed=seed
    )
    return np.asarray(result.species)


@pytest.mark.parametrize(
    "method, kind",
    [("ssa", "plain"), ("ssa", "watched"), ("ssa", "step"), ("psa", "plain")],
    ids=["ssa", "ssa with an event", "ssa with a step rate", "psa"],
)
@pytest.mark.parametrize("t0", [1e9, 1e12, 1e15])
def test_a_run_far_from_the_origin_is_the_run_from_it(method, kind, t0):
    for seed in range(5):
        near = _run(_decay(0.0, kind), 0.0, method, seed)
        far = _run(_decay(t0, kind), t0, method, seed)
        np.testing.assert_array_equal(far, near, err_msg=f"seed {seed}")


def test_the_mean_far_from_the_origin_is_the_exact_mean():
    """The issue's own measure: the mean of A at t0 + 2 over ten seeds, from
    t0 = 1e12, against 1e5·e⁻¹ (the issue's seeds gave 22,928, z = −287)."""
    finals = [_run(_decay(), 1e12, "ssa", seed)[-1, 0] for seed in range(10)]
    exact = 1e5 * np.exp(-1.0)
    sd = np.sqrt(1e5 * np.exp(-1.0) * (1.0 - np.exp(-1.0)) / 10)
    assert abs(np.mean(finals) - exact) < 5 * sd


@pytest.mark.parametrize("t0", [1e12, 1e15])
def test_the_row_at_an_event_reads_the_event(t0):
    """A dose of 50,000 at t0 + 1, read at t0 + 1. A firing whose exact time is
    just past the dose but rounds onto it was weighed against the rounded
    time, came first, and the row read the state without the dose."""
    for seed in range(40):
        a = _run(_decay(t0, "dose"), t0, "ssa", seed)[:, 0]
        assert a[2] > 1e5, f"seed {seed}: A = {a[2]} at the dose"


@pytest.mark.parametrize("kind", ["plain", "watched"])
def test_a_wait_too_long_for_a_double_fires_nothing(kind):
    """k = 1e-310 on one copy: a0 is subnormal and -ln(r1)/a0 overflows to inf,
    so nothing fires in the run. The carry of that wait was inf - inf, a NaN no
    sample was ever read past, and the reaction fired at every seed."""
    for seed in range(5):
        b = _run(_decay(kind=kind, k=1e-310, n=1.0), 0.0, "ssa", seed)[:, 1]
        np.testing.assert_array_equal(b, 0.0, err_msg=f"seed {seed}")
