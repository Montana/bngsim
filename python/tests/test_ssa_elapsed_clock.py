"""Issue #1059: an SSA run from a large t_start keeps its waiting times.

Far from the origin the doubles are coarse, 1.2e-4 apart at t = 1e12, and a
waiting time placed at ``t + tau`` loses what is below that spacing. #1075 made
the two discrete loops carry what rounding drops. The continuous loop, which
runs a model whose rate reads the clock, places its firings on panels in
absolute time, and kept the loss: a decay ``k*A*(1 + 1e-30*time())`` from 1e5
copies, whose exact mean at t0 + 2 is 1e5/e = 36788, came back 22829 (z = -289)
from t0 = 1e12. The run now counts its own clock from t_start, and hands the
model t_start + elapsed where the model reads the clock; rows report the times
asked for, and the state and events carried to a next run their absolute times.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

T0 = 1e12
EXACT = 1e5 * np.exp(-1.0)


def _ssa(text, **run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_antimony_string(text)
        return bngsim.Simulator(model, method="ssa").run(**run)


def _z(values):
    values = np.asarray(values, dtype=float)
    se = np.sqrt(EXACT * (1 - np.exp(-1.0)) / len(values))
    return (values.mean() - EXACT) / se


@pytest.mark.parametrize(
    "law",
    [
        pytest.param("k*A*(1 + 1e-30*time())", id="continuous loop (reads the clock)"),
        pytest.param("k*A", id="discrete loop (#1075)"),
    ],
)
def test_a_decay_from_a_far_start_keeps_its_rate(law):
    text = f"species A = 100000; k = 0.5; R: A -> ; {law};"
    ends = [
        np.asarray(_ssa(text, sample_times=[T0, T0 + 1, T0 + 2], seed=s + 1).species)[-1, 0]
        for s in range(10)
    ]
    assert abs(_z(ends)) < 5.0, _z(ends)


def test_the_rows_are_the_times_asked_for():
    times = [T0, T0 + 0.5, T0 + 2]
    r = _ssa(
        "species A = 1000; k = 0.5; R: A -> ; k*A*(1 + 1e-30*time());", sample_times=times, seed=1
    )
    assert list(np.asarray(r.time)) == times


def test_an_event_at_an_absolute_time_fires_there():
    """The clock the trigger reads is absolute: halving A at t0 + 1 shows
    between the rows at t0 + 0.9 and t0 + 1.1, nowhere else."""
    text = (
        "species A = 1000000; k = 1e-12; R: A -> ; k*A*(1 + 1e-30*time());\n"
        f"E: at (time > {T0!r} + 1): A = A/2;\n"
    )
    r = _ssa(text, sample_times=[T0, T0 + 0.9, T0 + 1.1, T0 + 2], seed=1)
    a = np.asarray(r.species)[:, 0]
    assert a[1] > 9.9e5 and 4.9e5 < a[2] < 5.1e5 and abs(a[3] - a[2]) < 10


def test_a_rate_that_switches_at_an_absolute_time_switches_there():
    """A rate on from t0 + 1 (a breakpoint, shifted onto the run's clock):
    nothing decays before it, and after it the decay is e^-k(t - t0 - 1)."""
    text = f"species A = 100000; k = 1; R: A -> ; k*A*piecewise(1, time() > {T0!r} + 1, 0);"
    ends = []
    for s in range(10):
        r = _ssa(text, sample_times=[T0, T0 + 1, T0 + 2], seed=s + 1)
        a = np.asarray(r.species)[:, 0]
        assert a[1] == 100000
        ends.append(a[2])
    assert abs(_z(ends)) < 5.0, _z(ends)


def test_a_run_continues_from_where_the_last_one_ended():
    """The state, the time and an execution still waiting are carried to the
    next run at their absolute times: a delayed event that fires at t0 + 1 and
    applies at t0 + 3 applies in the second leg, at t0 + 3."""
    text = (
        "species A = 1000; k = 1e-12; R: A -> ; k*A*(1 + 1e-30*time());\n"
        f"E: at 2 after (time > {T0!r} + 1): A = 7;\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_antimony_string(text)
        sim = bngsim.Simulator(model, method="ssa")
        first = sim.run(sample_times=[T0, T0 + 1, T0 + 2], seed=1)
        second = sim.run(sample_times=[T0 + 2, T0 + 2.9, T0 + 3.1], seed=2)
    a1, a2 = np.asarray(first.species)[:, 0], np.asarray(second.species)[:, 0]
    assert a1[-1] == 1000 and a2[0] == 1000 and a2[1] == 1000 and a2[2] == 7
    assert list(np.asarray(second.time)) == [T0 + 2, T0 + 2.9, T0 + 3.1]


def test_a_long_run_hands_on_its_exact_end():
    """A run longer than its start, t_end > 2*t_start, where t_start + (t_end -
    t_start) is not t_end in doubles. The next run continues from the exact
    t_end, and so finds the execution still waiting: one that fires at t0 + 1
    with a delay of t_end - t0 applies a second into the second leg."""
    t0, te = 1146994.9100613801, 4086777.6982969143
    assert t0 + (te - t0) != te
    text = (
        "species A = 1000; k = 1e-12; R: A -> ; k*A*(1 + 1e-30*time());\n"
        f"E: at {te - t0!r} after (time > {t0!r} + 1): A = 7;\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ssa")
        sim.run(sample_times=[t0, t0 + 2, te], seed=1)
        second = sim.run(sample_times=[te, te + 0.5, te + 1.5], seed=2)
    np.testing.assert_array_equal(np.asarray(second.species)[:, 0], [1000, 1000, 7])
