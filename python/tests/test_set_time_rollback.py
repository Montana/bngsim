"""``Simulator.set_time`` and ``set_state(state, time=...)``: rolling a step back.

A predictor-corrector coupling loop advances a step, corrects its coupling input,
and redoes the step from the saved state *and time*. The state half always had a
public setter; the clock half had none, so the redo either wrote the private
``_current_time`` or integrated the corrector over the next interval instead of
the same one. That is invisible on an autonomous model and wrong on any model
that reads the clock — a time-indexed table function, ``time()`` in a rate law —
because the corrector then sees the forcing a step late.

``restore`` rolls both back but rebuilds the backend, discarding the warm ODE
solver a coupling loop exists to keep; ``set_time`` only moves the clock.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.kernel import ReactionKernel

#: A -> B at a rate that follows a time-indexed table, B -> A at kb: the step
#: from t to t+dt depends on which t it starts at.
_NET = textwrap.dedent(
    """
    begin parameters
      1 kb 0.3
    end parameters
    begin species
      1 A() 10
      2 B() 0
    end species
    begin functions
      1 F() tfun([0,5,10,20],[0.1,2,0.5,1],time)
    end functions
    begin reactions
      1 1 2 F
      2 2 1 kb
    end reactions
    """
).strip()


@pytest.fixture
def forced_net(tmp_path: Path) -> str:
    path = tmp_path / "forced.net"
    path.write_text(_NET + "\n")
    return str(path)


def _kernel(net: str) -> ReactionKernel:
    return ReactionKernel(bngsim.Model.from_net(net), method="ode")


class TestSetTime:
    def test_moves_the_clock(self, forced_net):
        sim = bngsim.Simulator(bngsim.Model.from_net(forced_net), method="ode")
        sim.run_until(4.0)
        sim.set_time(1.5)
        assert sim.current_time == 1.5
        r = sim.run_until(2.0, n_points=3)
        np.testing.assert_allclose(r.time, [1.5, 1.75, 2.0])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_a_non_finite_time(self, forced_net, bad):
        sim = bngsim.Simulator(bngsim.Model.from_net(forced_net), method="ode")
        sim.run_until(3.0)
        with pytest.raises(ValueError, match="finite"):
            sim.set_time(bad)
        assert sim.current_time == 3.0

    def test_set_state_with_time_is_all_or_nothing(self, forced_net):
        sim = bngsim.Simulator(bngsim.Model.from_net(forced_net), method="ode")
        sim.run_until(3.0)
        state = sim.get_state().copy()

        with pytest.raises(ValueError):
            sim.set_state(np.array([1.0, 2.0, 3.0]), time=1.0)  # wrong length
        assert sim.current_time == 3.0

        with pytest.raises(ValueError, match="finite"):
            sim.set_state(np.array([7.0, 8.0]), time=float("nan"))
        assert sim.current_time == 3.0
        np.testing.assert_array_equal(sim.get_state(), state)

        sim.set_state(np.array([7.0, 8.0]), time=1.0)
        assert sim.current_time == 1.0
        np.testing.assert_array_equal(sim.get_state(), [7.0, 8.0])


class TestKernelRollback:
    def test_redone_step_equals_a_straight_advance(self, forced_net):
        """Each step is advanced, rolled back to its saved state and time, and
        advanced again; the loop must land where a straight loop does. Rolling
        back the state alone does not, which is what makes this a test of the
        clock: the forcing is read at the wrong times."""
        dt, n_steps = 0.75, 16
        straight = _kernel(forced_net)
        for _ in range(n_steps):
            straight.advance(dt)

        redone = _kernel(forced_net)
        for _ in range(n_steps):
            saved, t0 = redone.get_state().copy(), redone.time
            redone.advance(dt)  # predictor
            redone.set_state(saved, time=t0)
            assert redone.time == t0
            redone.advance(dt)  # corrector, over the same interval

        assert redone.time == pytest.approx(straight.time)
        np.testing.assert_allclose(redone.get_state(), straight.get_state(), rtol=1e-10)

        state_only = _kernel(forced_net)
        for _ in range(n_steps):
            saved = state_only.get_state().copy()
            state_only.advance(dt)
            state_only.set_state(saved)  # the clock stays a step ahead
            state_only.advance(dt)
        assert not np.allclose(state_only.get_state(), straight.get_state(), rtol=1e-3)

    def test_reset_rewinds_through_set_time(self, forced_net):
        k = _kernel(forced_net)
        first = k.advance(6.0).copy()
        k.reset()
        assert k.time == 0.0
        np.testing.assert_array_equal(k.advance(6.0), first)
