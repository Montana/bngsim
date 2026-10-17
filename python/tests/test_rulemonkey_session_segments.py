"""Issue #1022: a RuleMonkey session advanced in short calls fires at its rate.

RuleMonkey 3.10.1 rolled the RNG back to redraw a waiting time that outlasted a
call, so the next call reused a draw already known to outlast the hazard up to
its start. A session advanced in calls shorter than its waiting times all but
stopped firing, with no warning (richardposner/RuleMonkey#89).
RuleMonkeySession.simulate and step_to take exactly that path. RuleMonkey 3.11.0
spends the draw and takes a fresh one at the call boundary, which is exact
because the waiting time has no memory.

first_order_switch.xml: A(s~0) -> A(s~1) at k = 0.5 from 20000 copies, a total
propensity of 10000, so a mean waiting time of 1e-4. Each molecule leaves on its
own, so A0 at t is binomial with p = exp(-0.5 t).
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

XML = str(
    Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim" / "first_order_switch.xml"
)


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_rulemonkey(), reason="RuleMonkey not compiled in")

T_END, N_CALLS = 0.02, 2000  # calls of 1e-5, a tenth of the mean waiting time


def _assert_binomial(a0: float, t: float) -> None:
    p = math.exp(-0.5 * t)
    assert abs(a0 - 20000 * p) < 4.5 * math.sqrt(20000 * p * (1 - p)), a0


@pytest.mark.parametrize("seed", [1, 2])
def test_many_short_segments_fire_at_the_rate_of_one_run(seed):
    grid = np.linspace(0.0, T_END, N_CALLS + 1)
    with bngsim.RuleMonkeySession(XML) as s:
        s.initialize(seed=seed)
        for t0, t1 in zip(grid[:-1], grid[1:], strict=True):
            r = s.simulate(float(t0), float(t1), n_points=2)
    # About 200 events; 3.10.1 left A0 at about 20000.
    _assert_binomial(float(r.observables["A0"][-1]), T_END)


@pytest.mark.parametrize("seed", [1, 2])
def test_many_short_steps_fire_at_the_rate_of_one_run(seed):
    grid = np.linspace(0.0, T_END, N_CALLS + 1)
    with bngsim.RuleMonkeySession(XML) as s:
        s.initialize(seed=seed)
        for t in grid[1:]:
            s.step_to(float(t))
        a0 = s.get_observable_values()[s.get_observable_names().index("A0")]
    _assert_binomial(float(a0), T_END)
