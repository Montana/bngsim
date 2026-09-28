"""Issue #869: a periodic dose written with floor() or mod() of time() is not
stepped over.

A `.net` pulse train has a zero rate law between pulses, so CVODE's step grows
past a whole pulse unless something stops it at the edges. Only one spelling,
``(time()-t0) - P*floor((time()-t0)/P) < w``, was recognized as a schedule and
stopped at; the difference of two floors (bare or compared) and ``mod()`` were
not, and each returned X = 0 where 11 pulses were given. Every edge of a
floor/ceil/rint/mod of time is now placed as a crossing stop.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import fixed_time_crossings

_NET = """begin parameters
    1 P {P}
    2 w {w}
    3 t0 {t0}
end parameters
begin species
    1 X() 0
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
begin groups
    1 Xtot 1
end groups
"""

# The issue's table: every spelling is a unit pulse on [t0 + nP, t0 + nP + w).
DOSES = {
    "floor_difference": "floor((time()-t0)/P) - floor((time()-t0-w)/P)",
    "floor_difference_compared": "if(floor((time()-t0)/P) - floor((time()-t0-w)/P) > 0.5, 1, 0)",
    "mod_compared": "if(mod(time()-t0+10*P, P) < w, 1, 0)",
    "ceil_difference": "ceil((time()-t0)/P) - ceil((time()-t0-w)/P)",
    "remainder_compared": "if((time()-t0) - P*floor((time()-t0)/P) < w, 1, 0)",
}

# (P, w, t0): daily, daily from t = 0, weekly, and fast.
SCHEDULES = [(24, 1, 5), (24, 1, 0), (168, 1, 5), (10, 0.5, 5)]
T_END = 252.0


def _model(tmp_path, dose: str, P: float, w: float, t0: float) -> bngsim.Model:
    path = tmp_path / "m.net"
    path.write_text(_NET.format(P=P, w=w, t0=t0, dose=dose), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _pulses_begun(P: float, t0: float) -> int:
    return math.floor((T_END - t0) / P) + 1


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
@pytest.mark.parametrize("P, w, t0", SCHEDULES)
@pytest.mark.parametrize("dose", DOSES.values(), ids=DOSES.keys())
def test_every_pulse_is_integrated(tmp_path, dose, P, w, t0, codegen):
    sim = bngsim.Simulator(_model(tmp_path, dose, P, w, t0), method="ode", codegen=codegen)
    if codegen:
        assert sim.codegen_backend in ("cc", "mir")
    r = sim.run(t_span=(0.0, T_END), n_points=int(T_END) + 1)
    x_end = float(np.asarray(r.species)[-1, 0])
    assert x_end == pytest.approx(w * _pulses_begun(P, t0), rel=1e-6)


def test_mod_of_a_negative_argument_keeps_fmods_sign(tmp_path):
    # fmod(time()-5, 24) is negative, so below w, for the whole of [0, 5): a
    # sixth hour of dose before the first pulse, which the stops must also see.
    m = _model(tmp_path, "if(mod(time()-t0, P) < w, 1, 0)", 24, 1, 5)
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, T_END), n_points=253)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(5 + 11, rel=1e-6)


@pytest.mark.parametrize("dose", DOSES.values(), ids=DOSES.keys())
def test_the_stops_are_the_pulse_edges(tmp_path, dose):
    m = _model(tmp_path, dose, 24, 1, 5)
    stops = fixed_time_crossings(m._core, 0.0, 60.0, m.time_discontinuity_conditions())
    assert stops == pytest.approx([5.0, 6.0, 29.0, 30.0, 53.0, 54.0], abs=1e-9)


def test_a_condition_that_never_changes_places_no_stop(tmp_path):
    # `rem >= 0` holds for every t. Its floor jumps once a period, but the
    # truth does not, and a stop where nothing changes only perturbs stepping.
    m = _model(tmp_path, "if(time() - P*floor(time()/P) >= 0, 1, 0)", 24, 1, 5)
    assert fixed_time_crossings(m._core, 0.0, 252.0, m.time_discontinuity_conditions()) == []


def test_a_step_of_state_is_not_a_stop(tmp_path):
    # floor(X) jumps at a time no one knows before the run. (+2, not +1: a
    # species fed at rate 1 would read as a counter clock, issue #443.)
    m = _model(tmp_path, "floor(Xtot/2) + 2", 24, 1, 5)
    assert m.time_discontinuity_conditions() == ()
