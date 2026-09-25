"""run_replicates leaves the interactive Simulator as it found it (issue #748).

The sequential path (``num_processors`` None or 1) reuses the live model and
simulator, and each replicate resets the model and runs it to ``t_end``. Nothing
put the model back afterwards, and the interactive clock, which only ``run()``
advances, stayed where it was. So after ``run_until(2)`` then
``run_replicates(...)``, the model held the last replicate's t = 10 state while
``current_time`` still said 2, and the next ``run_until(3)`` integrated from that
state and labelled it t = 2, with no warning. The parallel path runs on
per-thread clones and never had the problem; both now leave the model and clock
untouched.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest

_NET = """
begin parameters
    1 k1 0.3
    2 k2 0.1
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1 2 k1
    2 2 0 k2
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""

_METHODS = [("ssa", {}), ("psa", {"poplevel": 10})]


def _sim(tmp_path: Path, method: str, kw: dict) -> bngsim.Simulator:
    net = tmp_path / "decay.net"
    net.write_text(textwrap.dedent(_NET).strip() + "\n")
    return bngsim.Simulator(bngsim.Model.from_net(str(net)), method=method, **kw)


@pytest.mark.parametrize("procs", [None, 1, 2], ids=["default", "one", "parallel"])
@pytest.mark.parametrize(("method", "kw"), _METHODS, ids=[m for m, _ in _METHODS])
def test_replicates_leave_the_model_and_clock_untouched(tmp_path, method, kw, procs):
    sim = _sim(tmp_path, method, kw)
    sim.run_until(2, n_points=2, seed=1)
    state, clock = np.array(sim.get_state()), sim.current_time

    sim.run_replicates(3, t_span=(0, 10), n_points=3, seed=5, num_processors=procs)

    assert sim.current_time == clock
    np.testing.assert_array_equal(sim.get_state(), state)

    # ...so the next leg continues exactly as if no replicates had run.
    control = _sim(tmp_path, method, kw)
    control.run_until(2, n_points=2, seed=1)
    expected = control.run_until(3, n_points=2, seed=1)
    got = sim.run_until(3, n_points=2, seed=1)
    np.testing.assert_array_equal(got.time, expected.time)
    np.testing.assert_array_equal(got.species, expected.species)


@pytest.mark.parametrize(("method", "kw"), _METHODS, ids=[m for m, _ in _METHODS])
def test_sequential_and_parallel_replicates_agree(tmp_path, method, kw):
    """The restore is around the loop, not inside it: each replicate still
    starts from reset(), so the trajectories themselves are unchanged."""
    seq = _sim(tmp_path, method, kw).run_replicates(4, t_span=(0, 10), n_points=5, seed=7)
    par = _sim(tmp_path, method, kw).run_replicates(
        4, t_span=(0, 10), n_points=5, seed=7, num_processors=2
    )
    for a, b in zip(seq, par, strict=True):
        assert a.seed == b.seed
        np.testing.assert_array_equal(a.species, b.species)
