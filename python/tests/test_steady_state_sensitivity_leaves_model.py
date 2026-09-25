"""steady_state(sensitivity_params=...) leaves the model as it found it (issue #705).

The sensitivity solve is taken at the steady state, so ``find_steady_state``
moved the model's species there for it, and left them there, with
``ic_state_dirty`` still false. The next ``run()`` then integrated from x_ss with
a fresh zero seed (A flat at its steady value instead of relaxing from A0, and a
wrong dA/dkf), and a second sensitivity solve started from the first one's
answer, so an accumulator drifted call to call. The plain solve already put the
state back; the sensitivity solve now does too.

Model: A <-> B (kf = 2, kr = 1, A0 = 3) plus an accumulator B -> B + P. A + B = 3
is conserved, so A_ss = A0*kr/(kf+kr) = 1 and dA_ss/dkf = -A0*kr/(kf+kr)^2 = -1/3,
and from A0 = 3, A(t) = 1 + 2*exp(-3t).
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

_NET = """
begin parameters
    1 kf 2
    2 kr 1
    3 ks 0.5
    4 A0 3
end parameters
begin species
    1 A() A0
    2 B() 0
    3 P() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
    3 2 2,3 ks
end reactions
begin groups
    1 Atot 1
    2 Btot 2
    3 Ptot 3
end groups
"""


@pytest.fixture
def sim(tmp_path: Path) -> bngsim.Simulator:
    net = tmp_path / "ab.net"
    net.write_text(textwrap.dedent(_NET).strip() + "\n")
    return bngsim.Simulator(bngsim.Model.from_net(str(net)), sensitivity_params=["kf"])


def _ss(sim: bngsim.Simulator):
    with warnings.catch_warnings():
        # P is masked out on purpose, so its dP_ss/dkf row is NaN and says so.
        warnings.simplefilter("ignore")
        return sim.steady_state(sensitivity_params=["kf"], mask=~sim._model.is_pure_sink())


def test_the_model_is_restored_after_a_sensitivity_solve(sim):
    before = np.array(sim._model.get_state())
    ss = _ss(sim)
    np.testing.assert_array_equal(sim._model.get_state(), before)
    # ...and the solve itself is unchanged: taken at x_ss, as before.
    assert float(np.asarray(ss.concentrations)[0]) == pytest.approx(1.0, rel=1e-8)
    assert float(np.asarray(ss.sensitivity)[0, 0]) == pytest.approx(-1.0 / 3.0, rel=1e-6)


def test_repeated_sensitivity_solves_do_not_drift(sim):
    """P, which only accumulates, drifted upward on every call."""
    p = [float(np.asarray(_ss(sim).concentrations)[2]) for _ in range(3)]
    assert p[0] == p[1] == p[2]


def test_a_following_run_starts_from_the_initial_condition(sim):
    """It started from x_ss: A = [1, 1, 1, 1] and dA/dkf off by 0.1 at t = 1."""
    _ss(sim)
    r = sim.run(t_span=(0, 5), n_points=6, rtol=1e-10, atol=1e-12)
    t = np.asarray(r.time)
    a = np.asarray(r.species)[:, 0]
    np.testing.assert_allclose(a, 1.0 + 2.0 * np.exp(-3.0 * t), rtol=1e-7)
    # dA/dkf from A0 = 3 (the closed form of the reported values).
    expected = [0.0, -0.416312, -0.342422, -0.333336]
    np.testing.assert_allclose(
        np.asarray(r.sensitivities)[[0, 1, 2, 5], 0, 0], expected, rtol=1e-5, atol=1e-7
    )
