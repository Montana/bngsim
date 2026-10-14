"""Issue #726: the cached time-condition scan follows the clock's rate.

A BNGL counter ``0 -> T`` at rate ``k_clock`` is a unit-rate clock at
``k_clock = 1`` and not otherwise, so ``Tobs >= on`` is a condition on time at
one value and a condition on state at another. The model cached the scan the
first time it was asked, and the state-root scan cached what it left out as
covered beside it. So a model whose clock rate moved after its first run kept
the first answer:

- first run at 0, then at 1: no crossing stop, and a 0.01-wide window was
  stepped over (issue #726 as filed; since #897 a state root catches it);
- first run at 1, then at 0.5: the thresholds stayed "covered" by a stop that
  no longer resolves at 0.5, so they had neither a stop nor a root, and the
  window was stepped over: Y = 0 for 10(1 - e^-2) = 8.6466, with no error.

Both caches are now keyed: the scan by which species are clocks, the roots by
what the scan covers. ``X -> Y`` at rate 100 inside the window
``on <= Tobs < on + w``, from X = 10: Y = 10(1 - exp(-100 w / k_clock)) once
the window has passed.
"""

from __future__ import annotations

import logging

import bngsim
import numpy as np
import pytest

_NET = """begin parameters
    1 k_clock 0
    2 on 50
    3 w 0.01
end parameters
begin functions
    1 rate_Y() if(Tobs>=on && Tobs<(on+w),100,0)
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 rate_Y
end reactions
begin groups
    1 Tobs 1
end groups
"""

SPAN = dict(t_span=(0.0, 200.0), n_points=3)
# (first clock rate, then this one): the window passes before t = 200 at both.
MOVES = [(0.0, 1.0), (1.0, 0.5), (0.5, 1.0)]
IDS = ["0-then-1", "1-then-half", "half-then-1"]


def _exact(k_clock: float) -> float:
    return 10.0 * (1.0 - np.exp(-100.0 * 0.01 / k_clock)) if k_clock > 0 else 0.0


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "clock_window.net"
    path.write_text(_NET)
    return str(path)


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def _Y(result) -> float:
    return float(np.asarray(result.species)[-1, 2])


def _ran_at(net, k_clock):
    model = bngsim.Model.from_net(net)
    model.set_param("k_clock", k_clock)
    bngsim.Simulator(model, method="ode").run(**SPAN)
    model.reset()
    return model


def _fresh(net, k_clock):
    model = bngsim.Model.from_net(net)
    model.set_param("k_clock", k_clock)
    return model


@pytest.mark.parametrize(("first", "then"), MOVES, ids=IDS)
def test_set_param_after_a_run(net, first, then):
    model = _ran_at(net, first)
    model.set_param("k_clock", then)
    got = _Y(bngsim.Simulator(model, method="ode").run(**SPAN))
    assert got == pytest.approx(_exact(then), rel=1e-5)
    fresh = _fresh(net, then)
    assert model.time_discontinuity_conditions() == fresh.time_discontinuity_conditions()
    assert model.state_switch_root_conditions() == fresh.state_switch_root_conditions()


@pytest.mark.parametrize(("first", "then"), MOVES, ids=IDS)
def test_parameter_scan(net, first, then):
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    got = [_Y(r) for r in sim.parameter_scan("k_clock", [first, then], **SPAN)]
    assert got == pytest.approx([_exact(first), _exact(then)], rel=1e-5, abs=1e-12)


@pytest.mark.parametrize(("first", "then"), MOVES, ids=IDS)
def test_a_clone_of_a_model_that_ran(net, first, then):
    clone = _ran_at(net, first).clone()
    clone.set_param("k_clock", then)
    got = _Y(bngsim.Simulator(clone, method="ode").run(**SPAN))
    assert got == pytest.approx(_exact(then), rel=1e-5)


@pytest.mark.parametrize(("first", "then"), MOVES, ids=IDS)
def test_run_batch_after_the_parent_ran(net, first, then):
    model = _fresh(net, first)
    sim = bngsim.Simulator(model, method="ode")
    sim.run(**SPAN)
    model.reset()
    got = [_Y(r) for r in sim.run_batch(params=[{"k_clock": then}], **SPAN)]
    assert got == pytest.approx([_exact(then)], rel=1e-5)


def test_the_counter_is_a_time_condition_only_while_it_is_a_clock(net):
    # Not covered by a stop, the thresholds are rooted on; covered, they are not.
    model = bngsim.Model.from_net(net)
    window = ("Tobs>=on", "Tobs<(on+w)")
    for k_clock, conditions, roots in [
        (1.0, window, ()),
        (0.5, (), window),
        (1.0, window, ()),
        (0.0, (), window),
    ]:
        model.set_param("k_clock", k_clock)
        assert model.time_discontinuity_conditions() == conditions
        assert model.state_switch_root_conditions() == roots
