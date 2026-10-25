"""A small conservation pool beside a large one keeps its total (issue #1265).

``_project_onto_laws`` places every start ``steady_states()`` solves from, and
since #1195 the seed from ``steady_state()``'s root too. It judged each law on
the largest law's scale:

* it pinned every member of a law of non-negative coefficients to 0 where the
  total was at most 1e-12 of the largest total, so a pool of 1e-5 beside one of
  1e8 was taken for an empty one, and every root came back with its total at 0;
* it stopped where each law was met to 1e-15 of the largest total, 1e-7 here,
  so the small pool's total was off by up to that much at each start. Newton
  keeps a start's totals, so each start reached a root at totals of its own: at
  a pool of 1e-3 the call returned four roots, totals 1.0000006e-3 to
  1.0000693e-3, where there is one, and at 1e-1 its one root was 1.4e-7 off.

Each law is judged on its own scale now. A law of non-negative coefficients has
a positive point exactly when its total is positive, so it is pinned only where
its total is 0 to the round-off of its own members.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._simulator import _project_onto_laws

_TWO_POOLS = """begin parameters
    1 k1 1
    2 k2 2
end parameters
begin species
    1 A() 1e8
    2 B() 0
    3 C() {c0}
    4 D() 0
end species
begin reactions
    1 1 2 k1
    2 2 1 k2
    3 3 4 k1
    4 4 3 k2
end reactions
begin groups
    1 Atot 1
end groups
"""
_LAWS = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])


def _two_pools(tmp_path, c0):
    path = tmp_path / "two_pools.net"
    path.write_text(_TWO_POOLS.format(c0=c0))
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")


def test_the_projection_keeps_a_small_total_beside_a_large_one():
    w = np.array([3e7, 5e7, 2e7, 7e7])
    b = np.array([1e8, 1e-5])
    y = _project_onto_laws(w, _LAWS, b)
    assert y is not None and (y > 0).all()
    np.testing.assert_allclose(_LAWS @ y, b, rtol=1e-12)


@pytest.mark.parametrize("small", [0.0, -1e-30])
def test_a_law_whose_total_is_zero_is_still_pinned(small):
    """nfkb's laws of one member at a total of 0, and a total that is 0 to the
    round-off of its own members: their members are set to 0."""
    w = np.array([3e7, 5e7, 2e7, 7e7])
    y = _project_onto_laws(w, _LAWS, np.array([1e8, small]))
    assert y is not None
    assert y[2] == 0.0 and y[3] == 0.0
    np.testing.assert_allclose(y[:2].sum(), 1e8, rtol=1e-12)


@pytest.mark.parametrize("c0", [1e-5, 1e-3, 1e-1])
def test_every_root_holds_both_totals(tmp_path, c0):
    """One root, at steady_state()'s, holding 1e8 and C0 to 1e-9."""
    sim = _two_pools(tmp_path, c0)
    ordinary = sim.steady_state()
    roots = sim.steady_states(n_starts=50, seed=0)
    assert len(roots) == 1
    np.testing.assert_allclose(_LAWS @ roots.states[0], [1e8, c0], rtol=1e-9)
    np.testing.assert_allclose(roots.states[0], ordinary.concentrations, rtol=1e-6)


# A switch in the small pool: X + Y = T, with Y -> X at b + x^4/(K^4 + x^4)
# per Y for x = X/T, and X -> Y at g, so x' = (1 - x)(b + x^4/(K^4 + x^4)) - g x
# has three roots, at x = 0.0386, 0.2639 and 0.6396. Beside it, A <-> B at 1e8.
_SWITCH = """begin parameters
    1 b   0.02
    2 K   0.4
    3 g   0.5
    4 T   {total}
    5 k1  1
    6 k2  2
end parameters
begin species
    1 X() 0
    2 Y() T
    3 A() 1e8
    4 B() 0
end species
begin functions
    1 act() b+(Xobs/T)^4/(K^4+(Xobs/T)^4)
end functions
begin reactions
    1 2 1 act
    2 1 2 g
    3 3 4 k1
    4 4 3 k2
end reactions
begin groups
    1 Xobs 1
end groups
"""


@pytest.mark.parametrize("total", [1e-3, 1.0])
def test_a_switch_in_a_small_pool_keeps_its_three_roots(tmp_path, total):
    """The counter-case to grouping the small pool's roots coarsely: its three
    roots, at the small total and at 1, each where the scalar equation puts it."""
    path = tmp_path / "switch.net"
    path.write_text(_SWITCH.format(total=total))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    roots = sim.steady_states(n_starts=100, seed=0)
    x = np.sort(roots.states[:, 0] / total)
    np.testing.assert_allclose(x, [0.03862, 0.26388, 0.63959], rtol=1e-3)
    assert sorted(roots.stability) == ["stable", "stable", "unstable"]
    np.testing.assert_allclose(roots.states[:, :2].sum(axis=1), total, rtol=1e-9)
