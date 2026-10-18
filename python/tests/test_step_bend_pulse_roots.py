"""Issue #1068: a pulse written with step or bend calls over a counter that is not a clock.

``100*(floor((Tobs-50)/100) - floor((Tobs-50.01)/100))`` is a pulse of width
0.01 in T at T = 50, written with no comparison, and so is the triangle
``100*max(0, 1 - abs(Tobs-50.005)/0.005)``. With T counting at 1 it is a
clock, and #869/#904 stop at each edge. At any other rate T is state: no stop
could be placed, no comparison was there to root on, the rate law is flat on
both sides of the pulse, and one step spanned it. Y came back 0 with no error.

Each call is now rooted where it jumps or bends: a rounding call where
``sin(pi*u)`` changes sign, a bend where its arguments meet, and an ``abs()``
or ``min()`` inside such a turn unfolded into its branches, which cross at the
pulse's feet.

T = k*t, so the pulse lasts 0.01/k and Y = 10(1 - exp(-1/k)) for the
rectangle, 10(1 - exp(-0.5/k)) for the triangle.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import _bend_branches, _turn_atoms

NET = """begin parameters
    1 k_clock {k}
end parameters
begin functions
    1 pulse() {law}
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 pulse
end reactions
begin groups
    1 Tobs 1
end groups
"""

# (rate law, its area in T)
PULSES = {
    "floor": ("100*(floor((Tobs-50)/100)-floor((Tobs-50.01)/100))", 1.0),
    "ceil": ("100*(ceil((Tobs-50)/100)-ceil((Tobs-50.01)/100))", 1.0),
    "sign": ("50*(sign(Tobs-50)-sign(Tobs-50.01))", 1.0),
    "inrange": ("100*inrange(50,Tobs,50.01)", 1.0),
    "max-abs triangle": ("100*max(0,1-abs(Tobs-50.005)/0.005)", 0.5),
    "min-max triangle": ("100*max(0,min((Tobs-50)/0.005,(50.01-Tobs)/0.005))", 0.5),
}


def _model(tmp_path, law: str, k: float) -> bngsim.Model:
    path = tmp_path / "pulse.net"
    path.write_text(NET.format(k=k, law=law))
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize("k", [2.0, 0.5])
@pytest.mark.parametrize("name", list(PULSES))
def test_the_pulse_is_not_stepped_over(tmp_path, name, k):
    law, area = PULSES[name]
    run = bngsim.Simulator(_model(tmp_path, law, k), method="ode").run(
        t_span=(0.0, 60.0 / k), n_points=3
    )
    y = float(np.asarray(run.species)[-1, 2])
    assert y == pytest.approx(10 * (1 - np.exp(-area / k)), rel=1e-5)


@pytest.mark.parametrize("name", ["floor", "max-abs triangle"])
def test_a_unit_rate_clock_keeps_its_stops_and_gets_no_roots(tmp_path, name):
    """At k = 1 the stops of #869/#904 cover every jump and bend, the
    abs() inside the max() included; a root there would be a second restart."""
    model = _model(tmp_path, PULSES[name][0], 1.0)
    assert model.time_discontinuity_conditions()
    assert model.state_switch_root_conditions() == ()


def test_the_turns_of_each_call():
    pi = repr(np.pi)
    assert _turn_atoms("floor(u)") == [f"sin({pi}*(u))<0"]
    assert _turn_atoms("round(u)") == [f"cos({pi}*(u))<0"]
    assert _turn_atoms("mod(a, b)") == [f"sin({pi}*(a)/(b))<0"]
    assert _turn_atoms("sgn(u)") == ["(u)<0"]
    assert _turn_atoms("clamp(lo,x,hi)") == ["(x)<(lo)", "(x)<(hi)"]
    assert _turn_atoms("max(a,b,c)") == ["(a)<(b)", "(a)<(c)", "(b)<(c)"]
    assert _turn_atoms("floor(a,b)") == []


def test_a_bend_in_a_turn_unfolds_into_its_branches():
    assert _bend_branches("0<1-abs(T-5)/w") == ["0<1-(T-5)/w", "0<1-(-(T-5))/w"]
    assert _bend_branches("(0)<(min(p, q))") == ["(0)<((p))", "(0)<((q))"]
    assert len(_bend_branches("abs(a)<abs(b)")) == 4
    assert _bend_branches("abs(a)<max(b,c,d)") == []  # six branches
    assert _bend_branches("a<b") == []
    assert _bend_branches("abs(abs(a)-1)<0.1") == []
