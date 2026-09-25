"""GH #824: a comparison used as a number is ExprTk's 1/0 in sympy too.

ExprTk has no boolean type, so a condition may do arithmetic or ordering on a
comparison: `(a==b)<1` (BNG2.pl's parenthesization of `a==b<1`), `(a<b)*2>1`,
`(a<b)+(b<a)>0`. sympy has no such coercion: `Eq(a, b) < 1` raised TypeError,
the rate law could not be parsed for differentiation, and the analytic
sensitivity RHS was declined for the whole model with a warning claiming a
moving crossing. Such a truth value is now `Piecewise((1, cond), (0, True))`.
"""

from __future__ import annotations

import logging

import bngsim
import numpy as np
import pytest
from bngsim._jacobian import _preprocess_exprtk

_NET = """begin parameters
    1 a 0.5
    2 b 2
    3 k1 0.5
    4 k2 0.1
end parameters
begin species
    1 A() 10
end species
begin functions
    1 f() if({cond}, k1, k2)
end functions
begin reactions
    1 1 0 f #_R1
end reactions
begin groups
    1 Atot 1
end groups
"""

# All three hold at a = 0.5, b = 2, so A decays at k1.
CONDITIONS = ["(a==b)<1", "(a<b)*2>1", "(a<b)+(b<a)>0"]


@pytest.mark.parametrize("cond", CONDITIONS)
def test_the_analytic_sensitivity_rhs_is_used_and_exact(tmp_path, cond, caplog):
    path = tmp_path / "m.net"
    path.write_text(_NET.format(cond=cond), encoding="utf-8")
    m = bngsim.Model.from_net(str(path))
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        r = bngsim.Simulator(m, method="ode", sensitivity_params=["k1", "k2", "a"]).run(
            t_span=(0.0, 2.0), n_points=3
        )
    assert not [rec for rec in caplog.records if "could not be parsed" in rec.getMessage()]
    t = np.asarray(r.time)
    a = 10.0 * np.exp(-0.5 * t)
    np.testing.assert_allclose(r.species[:, 0], a, rtol=1e-6)
    s = np.asarray(r.sensitivities)[:, 0, :]
    np.testing.assert_allclose(s[:, 0], -t * a, rtol=1e-5, atol=1e-9)  # k1: the branch taken
    np.testing.assert_allclose(s[:, 1:], 0.0, atol=1e-9)  # k2 and a: no effect


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A condition in its own place is untouched.
        ("if(a<b,2,1)", "Piecewise((2, a<b), (1, True))"),
        # Used as a number, it is ExprTk's 1/0.
        ("k*(A>1)", "k * Piecewise((1, A > 1), (0, True))"),
        ("if(c, a<b, 0)", "Piecewise((Piecewise((1, a < b), (0, True)), c), (0, True))"),
    ],
)
def test_rewrite(text, expected):
    assert _preprocess_exprtk(text) == expected
