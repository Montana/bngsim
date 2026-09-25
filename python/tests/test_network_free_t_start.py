"""GH #703: method='nf' and method='rm' start from the initial state at t_start.

Both network-free backends ignored t_start: NFsim stepped to absolute times and
labelled the t = 0 state as t_start, and RuleMonkey's stateless run simulated
[0, t_start] before its first row. A bngsim run starts from the initial state at
t_start and never sees [0, t_start], as ode and ssa do and as BNG2.pl's
simulate_nf does (it runs NFsim for t_end - t_start).

A(s~0) -> A(s~1) at k = 0.5 from 20000 copies, so A0 = 20000*exp(-0.5 (t - t0)).
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
XML = DATA / "nfsim" / "first_order_switch.xml"


def _available(method: str) -> bool:
    if method == "nf":
        return bool(getattr(bngsim, "HAS_NFSIM", False))
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


# Literal reasons, the ones the other network-free tests give: the skip audit
# (test_skip_audit.py) reads each reason from the source and cannot read an
# f-string's.
METHODS = [
    pytest.param(
        "nf", marks=pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
    ),
    pytest.param(
        "rm",
        marks=pytest.mark.skipif(not _available("rm"), reason="RuleMonkey not compiled in"),
    ),
]


def _run(method: str, seed: int = 3, **times):
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    sim = bngsim.Simulator(m, method=method, xml_path=str(XML))
    r = sim.run(seed=seed, **times)
    return np.asarray(r.time), np.asarray(r.observables["A0"])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "times", [{"t_span": (5.0, 7.0), "n_points": 3}, {"sample_times": [5.0, 6.0, 7.0]}]
)
def test_a_late_start_begins_at_the_initial_state(method, times):
    t, a = _run(method, **times)
    np.testing.assert_array_equal(t, [5.0, 6.0, 7.0])
    assert a[0] == 20000.0
    # Within SSA noise (sd ~ 100) of the closed form; before the fix the rows
    # were ~1000 and ~600 (a decay from t = 0).
    np.testing.assert_allclose(a, 20000.0 * np.exp(-0.5 * (t - 5.0)), rtol=0.05)


@pytest.mark.parametrize("method", METHODS)
def test_a_late_start_is_the_same_run_shifted(method):
    """Same seed, same elapsed schedule: the counts are identical to a run
    from t = 0, only the labels move."""
    _, a0 = _run(method, t_span=(0.0, 2.0), n_points=3)
    t5, a5 = _run(method, t_span=(5.0, 7.0), n_points=3)
    np.testing.assert_array_equal(t5, [5.0, 6.0, 7.0])
    np.testing.assert_array_equal(a5, a0)
