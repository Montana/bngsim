"""Issue #1158 — an event that re-arms its own trigger as it fires is refused.

``at (B > 0): B = 0, F = F + 1`` with ``B' = 1``: each fire puts B back on its
threshold, the flow makes the trigger true again at once, and the event fires
infinitely often at one instant (Zeno behaviour). No trajectory continues past
that instant, in bngsim or anywhere else; libRoadRunner stops at F = 2, which is
where its own event handling gives up, not an answer.

bngsim fired at every root CVODE reported, each one root tolerance (100 ulps of
|t| + |h|) after the last, about 4e5 fires a second, and ``run`` never returned:
over t in [0, 1e-12] as over [0, 2]. The chatter guard (GH #95) did not see it,
since each fire changes the state (F, or B and A) by more than the tolerance.

Those fires are the same instant to the solver, so a run of them is the
same-instant algebraic loop ``CASCADE_LIMIT`` (100000) already refuses, one
rounding at a time: a fire within twice the root tolerance of the same event's
previous one is counted, and past the limit the run is refused, naming the
event. A pulse train, however fast, is spaced by more than the solver's root
resolution and never counts.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_ZENO = {
    "reset B, first order": "species B = 0; B' = 1;\nF = 0; E: at (B > 0): F = F + 1, B = 0;\n",
    "reset B, second order": (
        "species B = 0, A = 0; A' = 1; B' = A;\nF = 0; E: at (B > 0): F = F + 1, B = 0;\n"
    ),
    "reset B and A": (
        "species B = 0, A = 0; A' = 1; B' = A;\nF = 0; E: at (B > 0): F = F + 1, B = 0, A = 0;\n"
    ),
    "put on the threshold mid-run": (
        "species B = -1, A = 0; A' = 1; B' = A;\nF = 0;\n"
        "E0: at (time > 0.5): B = 0, A = 0;\nE: at (B > 0): F = F + 1, B = 0, A = 0;\n"
    ),
    "a clamp with a counter": "species X = 1; X' = -1; c = 0;\nE: at (X < 0): X = 0, c = c + 1;\n",
}


def _run(text, t_span, n_points=5):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode").run(
        t_span=t_span, n_points=n_points, rtol=1e-10, atol=1e-12
    )
    last = dict(zip(run.species_names, np.asarray(run.species)[-1], strict=True))
    last.update(zip(run.observable_names, np.asarray(run.observables)[-1], strict=True))
    return last


@pytest.mark.parametrize("shape", sorted(_ZENO))
def test_an_event_that_rearms_itself_as_it_fires_is_refused_by_name(shape):
    """Each of these hung: `run` did not return. Now it is refused, naming E."""
    with pytest.raises(bngsim.SimulationError, match=r"Event 'E' fired 100000 times") as err:
        _run(_ZENO[shape], (0.0, 2.0))
    assert "Zeno behaviour" in str(err.value)
    assert "issue #1158" in str(err.value)


@pytest.mark.parametrize("t0", [300.0, 1e6])
def test_a_self_reset_is_refused_from_any_start(t0):
    """From a start that is not 0 the fires are as close, to that time's own
    rounding."""
    with pytest.raises(bngsim.SimulationError, match="Zeno behaviour"):
        _run(_ZENO["reset B, first order"], (t0, t0 + 2.0))


@pytest.mark.parametrize(
    "t_start, period, n_expected",
    [(0.0, 1e-6, 105000), (1.0, 2e-9, 105001)],
    ids=["1e-6 from 0", "2e-9 at t=1"],
)
def test_a_fast_pulse_train_past_the_limit_is_not_refused(t_start, period, n_expected):
    """More than CASCADE_LIMIT fires of one event, each a real period after the
    last (far more than the solver's root tolerance): a train, not a loop."""
    text = (
        f"species Z = 1; Z' = -Z; n = 0; tn = {t_start!r};\n"
        f"P: at (time >= tn), t0=false: tn = tn + {period!r}, n = n + 1;\n"
    )
    last = _run(text, (t_start, t_start + 105000 * period), 3)
    assert last["n"] == pytest.approx(n_expected, abs=1)


def test_a_decaying_clamp_is_still_stepped_over_by_the_chatter_guard():
    """`at (X < 0): X = 0` with X' = -X fires on noise once X is far below
    atol, changing it by less than the tolerance: the chatter guard (GH #95)
    makes it dormant long before the Zeno guard would count to its limit.
    X(3) = e^-3 as in libRoadRunner (0.0497873)."""
    last = _run("species X = 1; X' = -X;\nC: at (X < 0): X = 0;\n", (0.0, 3.0), 4)
    assert last["X"] == pytest.approx(np.exp(-3.0), rel=1e-5)
