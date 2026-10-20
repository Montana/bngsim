"""An event that re-arms itself onto its threshold is refused by name (#1158).

`at (B > 0): B = 0` with the flow carrying B straight back above 0 fires
infinitely often, at times that accumulate at the first fire (Zeno). No finite
trajectory exists, and `Simulator.run` used to never return. It now stops after
the event fires ZENO_LIMIT times in a row, each within 1e-9 of max(|t|,
horizon) of the last, with an error naming the event. A fast but real event
train is not caught: its fires are real time apart.
"""

from __future__ import annotations

import time

import bngsim
import numpy as np
import pytest

ZENO = {
    "reset_B_o1": "species B = 0; B' = 1;\nF = 0; E: at (B > 0): F = F + 1, B = 0;\n",
    "reset_B_o2": (
        "species B = 0, A = 0; A' = 1; B' = A;\nF = 0; E: at (B > 0): F = F + 1, B = 0;\n"
    ),
    "reset_both_o2": (
        "species B = 0, A = 0; A' = 1; B' = A;\nF = 0; E: at (B > 0): F = F + 1, B = 0, A = 0;\n"
    ),
    "reset_mid_o2": (
        "species B = -1, A = 0; A' = 1; B' = A;\nF = 0;\n"
        "E0: at (time > 0.5): B = 0, A = 0;\nE: at (B > 0): F = F + 1, B = 0, A = 0;\n"
    ),
}


@pytest.mark.parametrize("name", list(ZENO))
def test_a_self_re_arming_event_is_refused_by_name(name):
    model = bngsim.Model.from_antimony_string(ZENO[name])
    start = time.monotonic()
    with pytest.raises(bngsim.SimulationError, match=r"event 'E' fired 1000 times.*Zeno"):
        bngsim.Simulator(model, method="ode").run(
            t_span=(0, 2), n_points=5, rtol=1e-10, atol=1e-12
        )
    assert time.monotonic() - start < 20  # it used to never return


def test_a_fast_real_event_train_is_not_caught():
    """A sawtooth reset every 1e-6: 2000 fires, twice ZENO_LIMIT, each a real
    1e-6 apart, 500,000 times the 1e-9 · 2e-3 the guard looks for."""
    model = bngsim.Model.from_antimony_string(
        "species X = 0, F = 0; X' = 1; F' = 0;\nE: at (X > 1e-6): X = 0, F = F + 1;\n"
    )
    run = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 2e-3), n_points=3, rtol=1e-10, atol=1e-14
    )
    fires = float(np.asarray(run.as_roadrunner(["F"])["F"])[-1])
    assert fires == pytest.approx(2000, abs=2)
