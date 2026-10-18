"""The steady-state march roots the rate-law conditions a plain run roots (issue #904).

A rate law switched by a comparison over model state is constant on each side
of its switch, so a step that spans a narrow window reads it as off at both
ends, and the window is lost at every tolerance. ``run()`` roots each such
condition since #897. The march ``steady_state()`` integrates by built its own
integrator and rooted nothing: with Y made at k while ``0.5 < S < 0.505`` and S
decaying from 1, ``steady_state()`` returned Y = 0 with ``converged=True``,
where a run gives 2·k·ln(1.01) = 0.1990.

Each condition is now rooted in the march too, and the march restarts on each
crossing, as a run does. Every expected value is a closed form.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

K, KD, LO, HI = 10.0, 0.5, 0.5, 0.505
# S = exp(-kd·t), so S is in (LO, HI) for ln(1/HI)/kd < t < ln(1/LO)/kd, and Y
# gains k over that stretch.
Y_SS = K * math.log(HI / LO) / KD

NET = """begin parameters
    1 k   {k!r}
    2 kd  {kd!r}
    3 lo  {lo!r}
    4 hi  {hi!r}
end parameters
begin functions
    1 r() {law}
end functions
begin species
    1 S() 1
    2 Y() 0
end species
begin reactions
    1 1 0 kd
    2 0 2 r
end reactions
begin groups
    1 Sobs 1
end groups
"""

LAWS = {
    "two-atoms": "if((Sobs>lo)&&(Sobs<hi),k,0)",
    "abs-window": "if(abs(Sobs-(lo+hi)/2)<(hi-lo)/2,k,0)",
    "as-a-number": "k*((Sobs>lo)*(Sobs<hi))",
}


def _model(tmp_path, law):
    path = tmp_path / "window.net"
    path.write_text(NET.format(k=K, kd=KD, lo=LO, hi=HI, law=law))
    return bngsim.Model.from_net(path)


@pytest.mark.parametrize("spelling", sorted(LAWS))
def test_a_window_on_the_approach_is_integrated_over(tmp_path, spelling):
    """Y was 0 with converged=True, in every spelling."""
    out = bngsim.Simulator(_model(tmp_path, LAWS[spelling]), method="ode").steady_state()
    assert out.converged
    y = np.asarray(out.concentrations)
    assert y[1] == pytest.approx(Y_SS, rel=1e-5)
    assert abs(y[0]) < 1e-6
    assert out.n_state_switch_roots >= 2
    assert not out.state_switch_roots_dropped


@pytest.mark.parametrize("spelling", sorted(LAWS))
def test_the_march_agrees_with_a_run(tmp_path, spelling):
    """The same model run long enough to settle."""
    model = _model(tmp_path, LAWS[spelling])
    run = bngsim.Simulator(model, method="ode").run(t_span=(0, 80), n_points=2)
    out = bngsim.Simulator(_model(tmp_path, LAWS[spelling]), method="ode").steady_state()
    assert np.asarray(out.concentrations)[1] == pytest.approx(
        np.asarray(run.species)[-1, 1], rel=1e-5
    )


def test_a_batch_entry_roots_its_own_window(tmp_path):
    """``steady_state_batch`` builds an integrator per entry; each one is
    rooted, at its own k."""
    sim = bngsim.Simulator(_model(tmp_path, LAWS["two-atoms"]), method="ode")
    out = sim.steady_state_batch([{"k": 10.0}, {"k": 20.0}])
    for r, k in zip(out, (10.0, 20.0), strict=True):
        assert np.asarray(r.concentrations)[1] == pytest.approx(
            k * math.log(HI / LO) / KD, rel=1e-5
        )


def test_a_model_with_no_condition_roots_nothing(tmp_path):
    """Control."""
    path = tmp_path / "plain.net"
    path.write_text(NET.format(k=K, kd=KD, lo=LO, hi=HI, law="k*Sobs"))
    out = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state()
    assert out.converged
    assert out.n_state_switch_roots == 0
    # dY/dt = k·S, integrated: k/kd.
    assert np.asarray(out.concentrations)[1] == pytest.approx(K / KD, rel=1e-5)


SLIDE = """begin parameters
    1 kon  2.0
    2 koff 1.0
    3 thr  0.5
end parameters
begin functions
    1 up() if(Sobs<thr,kon,0)
end functions
begin species
    1 S() 0
end species
begin reactions
    1 0 1 up
    2 1 0 koff
end reactions
begin groups
    1 Sobs 1
end groups
"""


@pytest.mark.parametrize("jacobian", ["analytical", "fd"])
def test_a_state_held_on_its_threshold_ends_where_it_did(tmp_path, monkeypatch, jacobian):
    """Control. S is made at 2 below 0.5 and not above, and removed at 1: the
    flow points at the threshold from both sides, and there is no steady
    state, only S held on 0.5. With the closed-form Jacobian the march parks
    on it, where the residual is exactly 0 and CVODE asks no more of that
    root. With the difference quotient it crosses back and forth, and past
    1000 crossings it roots them no more and goes on over the switch. Either
    way it ends where the march with nothing rooted ended: S on the threshold,
    not converged. How near the threshold a march that slides along it stops
    is the integrator's, and differs by platform (2.4e-5 above it on Linux CI,
    1e-9 here), so it is held to 1e-3."""
    path = tmp_path / "slide.net"
    path.write_text(SLIDE)
    out = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", jacobian=jacobian)
    out = out.steady_state()
    if jacobian == "analytical":
        assert out.n_state_switch_roots <= 2
        assert not out.state_switch_roots_dropped
    else:
        assert out.n_state_switch_roots == 1001
        assert out.state_switch_roots_dropped
    monkeypatch.setattr(bngsim.Model, "state_switch_root_conditions", lambda self: ())
    before = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", jacobian=jacobian)
    before = before.steady_state()
    assert before.n_state_switch_roots == 0
    assert not out.converged and not before.converged
    assert np.asarray(out.concentrations)[0] == pytest.approx(0.5, abs=1e-3)
    assert np.asarray(before.concentrations)[0] == pytest.approx(0.5, abs=1e-3)
