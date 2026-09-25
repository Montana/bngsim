"""GH #756: an observable named ``E`` or ``pi`` is a model value, not a constant.

``_prepare_derived_expr`` bound only parameter names in ``parse_expr``'s
``local_dict``, so any other bare identifier resolved through sympy's own
namespace: an observable ``E`` became Euler's number and ``pi`` became π. A
constant adds no free symbol, so ``time() > k*E`` passed as a parameter-only
clock threshold at the phantom time t = e*k. The real crossing's switch root
and saltation jump were never registered, and dX/dk came back an exact zero.

X grows at rate 1 once ``time() > k*<obs>``, with the observable held at 5, so
X = max(0, t - 5k) and dX/dk = -5 after the crossing.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _derived_expr_value_numeric

_NET = """begin parameters
    1 k 1.0
end parameters
begin species
    1 $Ez() 5
    2 $Src() 1
    3 X() 0
end species
begin reactions
    1 2 2,3 f1 #_R1
end reactions
begin groups
    1 {obs} 1
    2 Xo 3
end groups
begin functions
    1 f1() if(time()>k*{obs},1,0)
end functions
"""


@pytest.mark.parametrize("obs", ["E", "pi"])
def test_crossing_on_an_observable_named_like_a_sympy_constant(tmp_path, obs):
    path = tmp_path / f"threshold_{obs}.net"
    path.write_text(_NET.format(obs=obs), encoding="utf-8")
    m = bngsim.Model.from_net(str(path))
    r = bngsim.Simulator(m, method="ode", sensitivity_params=["k"]).run(
        t_span=(0.0, 10.0), n_points=11
    )
    x = list(m.species_names).index("X()")
    t = np.asarray(r.time)
    np.testing.assert_allclose(r.species[:, x], np.maximum(0.0, t - 5.0), atol=1e-6)
    after = t > 5.0
    np.testing.assert_allclose(r.sensitivities[after, x, 0], -5.0, rtol=1e-5)
    np.testing.assert_allclose(r.sensitivities[t < 5.0, x, 0], 0.0, atol=1e-8)


@pytest.mark.parametrize("name", ["E", "pi", "S", "N"])
def test_non_parameter_name_is_not_evaluated_as_a_sympy_builtin(name):
    # Only `k` is a parameter; the other name must leave the value unresolved.
    assert _derived_expr_value_numeric(f"k*{name}", {"k"}, set(), {"k": 0}, [1.0]) is None
    assert _derived_expr_value_numeric("k*5", {"k"}, set(), {"k": 0}, [1.0]) == 5.0
