"""GH #754: ``(cond)==1`` / ``(cond)!=1`` keep their condition in sympy.

ExprTk has no boolean type, so ``(a>1)==1`` is simply ``a>1``. The sympy
rewrite turned it into ``Eq(a>1, 1)``, which sympy folds to ``False`` because a
Boolean never equals Integer 1 (and ``Ne`` to ``True``). The if() then took the
wrong branch in every sympy-derived quantity while the trajectory, evaluated by
ExprTk, stayed right: the sensitivity landed in the other parameter's column.

With a = 2 the condition holds, so ``==1`` decays at k1 and ``!=1`` at k2:
A = 10 exp(-k t), dA/dk = -t A for the branch taken and 0 for the other.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _preprocess_derived_expr

_NET = """begin parameters
    1 a 2
    2 k1 0.5
    3 k2 0.1
end parameters
begin species
    1 A() 10
end species
begin reactions
    1 1 0 f1 #_R1
end reactions
begin groups
    1 Atot 1
end groups
begin functions
    1 f1() if({cond},k1,k2)
end functions
"""


@pytest.mark.parametrize(
    ("cond", "taken", "rate"),
    [("(a>1)==1", 0, 0.5), ("1==(a>1)", 0, 0.5), ("(a>1)!=1", 1, 0.1), ("(a>1)==1.0", 0, 0.5)],
)
def test_sensitivity_follows_the_branch_the_trajectory_takes(tmp_path, cond, taken, rate):
    path = tmp_path / "cond.net"
    path.write_text(_NET.format(cond=cond), encoding="utf-8")
    m = bngsim.Model.from_net(str(path))
    r = bngsim.Simulator(m, method="ode", sensitivity_params=["k1", "k2"]).run(
        t_span=(0.0, 2.0), n_points=3
    )
    t = np.asarray(r.time)
    a = r.species[:, 0]
    np.testing.assert_allclose(a, 10.0 * np.exp(-rate * t), rtol=1e-6)
    np.testing.assert_allclose(r.sensitivities[:, 0, taken], -t * a, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(r.sensitivities[:, 0, 1 - taken], 0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("if((a>1)==1,k1,k2)", "Piecewise((k1, (a>1)), (k2, True))"),
        ("if((a>1)!=1,k1,k2)", "Piecewise((k1, Not((a>1))), (k2, True))"),
        # Not claimed: a numeric operand, and a literal other than 0 or 1.
        ("if(p==1,k1,k2)", "Piecewise((k1, Eq(p, 1)), (k2, True))"),
        # Compared with any other number the truth value is its 1/0 (issue #824),
        # which never equals 2: sympy folds the branch away, as ExprTk would.
        (
            "if((a>1)==2,k1,k2)",
            "Piecewise((k1, Eq(Piecewise((1, a > 1), (0, True)), 2)), (k2, True))",
        ),
    ],
)
def test_rewrite(text, expected):
    assert _preprocess_derived_expr(text) == expected
