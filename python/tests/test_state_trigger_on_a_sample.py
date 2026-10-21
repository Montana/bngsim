"""An event whose trigger on the state crosses on an output sample fires in
that sample's row (issue #1124, item 2).

``q' = 1; at (q >= 2.5): S = 1`` with a sample at 2.5: CVODE reaches
q(2.5) = 2.4999999999999996, a few doubles short, so the trigger read false
at the row, the root was found a few doubles after it, and the row held the
state from before the event ([S] = 3; libRoadRunner and SSA give 1). The root
search places a crossing no closer than its own tolerance, so a root that
near is not after the row by anything the run can tell. The triggers at an
output are now also read with the state half that tolerance ahead, and the
clock at the row.

A trigger on the clock is exact, and keeps #950's answer: an event a double
after a sample leaves that row before it.

Oracles: closed forms. q = t, so the event is at t = 2.5; its count of
executions; and for the clock, the order of two doubles.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

_RISE = "species S = 3; q = 0; q' = 1; E1: at (q {op} 2.5): S = 1;"

_GRIDS = {
    "a_sample_on_it": [0.0, 1.0, 2.0, 2.5, 3.0],
    "only_it": [0.0, 2.5],
    "half_steps": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "linspace": list(np.linspace(0.0, 5.0, 11)),
}


def _column(text, times, name, params=None, **kw):
    model = bngsim.Model.from_antimony_string(text)
    for key, value in (params or {}).items():
        model.set_param(key, value)
    r = bngsim.Simulator(model, method="ode").run(sample_times=times, **kw)
    return [float(v) for v in np.asarray(r.species)[:, list(r.species_names).index(name)]]


def _row(times, t):
    return [float(x) for x in times].index(t)


@pytest.mark.parametrize("grid", sorted(_GRIDS))
def test_the_row_on_the_crossing_holds_the_event(grid):
    """The issue's repro, on four grids. Each was 3 at t = 2.5."""
    times = _GRIDS[grid]
    s = _column(_RISE.format(op=">="), times, "S")
    at = _row(times, 2.5)
    assert s[:at] == [3.0] * at
    assert s[at:] == [1.0] * (len(times) - at)


def test_it_agrees_with_ssa():
    times = _GRIDS["a_sample_on_it"]
    model = bngsim.Model.from_antimony_string(_RISE.format(op=">="))
    r = bngsim.Simulator(model, method="ssa").run(sample_times=times, seed=1)
    ssa = [float(v) for v in np.asarray(r.species)[:, list(r.species_names).index("S")]]
    assert _column(_RISE.format(op=">="), times, "S") == ssa


def test_it_fires_once():
    """Fired at the row, the root a few doubles later is behind the run."""
    text = "species n = 0; q = 0; q' = 1; E1: at (q >= 2.5): n = n + 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0, 4.0], "n") == [0, 0, 0, 1, 1, 1]


def test_a_falling_trigger():
    text = "species S = 3; q = 5; q' = -1; E1: at (q <= 2.5): S = 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S") == [3, 3, 3, 1, 1]


def test_a_delayed_execution_runs_its_delay_from_the_row():
    text = "species S = 3; q = 0; q' = 1; E1: at 1 after (q >= 2.5): S = 1;"
    assert _column(text, [0.0, 2.5, 3.4, 3.5, 4.0], "S") == [3, 3, 3, 1, 1]


def test_at_a_tight_tolerance_as_well():
    s = _column(_RISE.format(op=">="), _GRIDS["a_sample_on_it"], "S", rtol=1e-10, atol=1e-12)
    assert s == [3, 3, 3, 1, 1]


def test_an_event_on_the_row_that_turns_the_state_back_is_read_ahead_from():
    """An event on a stop at the sample turns q back before it reaches 2.5.

    q tops out a few doubles short of 2.5 and falls, so E1 never fires. Read
    ahead on the derivative from before E0 (q' = +1), q crossed 2.5 and E1 fired
    in the row.
    """
    text = (
        "species S = 3; q = 0; k = 1; q' = k;"
        " E0: at (time >= 2.5): k = -1; E1: at (q >= 2.5): S = 1;"
    )
    for times in ([0.0, 1.0, 2.0, 2.5, 3.0], [0.0, 2.5, 3.0]):
        assert _column(text, times, "S") == [3.0] * len(times)
        assert max(_column(text, times, "q")) < 2.5


def test_a_crossing_after_the_sample_by_more_than_rounding_is_after_it():
    """1e-9 after the sample is far past the root tolerance: the row is 3."""
    text = "species S = 3; q = 0; q' = 1; E1: at (q >= 2.5 + 1e-9): S = 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S") == [3, 3, 3, 3, 1]


@pytest.mark.parametrize("ulps", [1, 8])
def test_a_clock_trigger_a_double_after_the_sample_is_after_it(ulps):
    """Only the state is read ahead: #950's row before the event stands."""
    tau = 2.5
    for _ in range(ulps):
        tau = math.nextafter(tau, math.inf)
    text = "species S = 3; tau = 0; E1: at (time >= tau): S = 1;"
    s = _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S", params={"tau": tau})
    assert s == [3, 3, 3, 3, 1]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1124: fixed for plain runs only. A sensitivity run's fire needs a located "
    "crossing for its jump, so it still records the row at 2.5 before the event (S = 3).",
)
def test_a_sensitivity_run_agrees_with_a_plain_run_at_the_row():
    """The same model and grid, with and without sensitivities, at the crossing row."""
    text = "species S = 3; q = 0; k = 1; q' = k; E1: at (q >= 2.5): S = 1;"
    times = _GRIDS["a_sample_on_it"]
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(sample_times=times)
    s = [float(v) for v in np.asarray(r.species)[:, list(r.species_names).index("S")]]
    assert s == _column(text, times, "S")
