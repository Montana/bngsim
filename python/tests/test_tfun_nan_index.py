"""A NaN table-function index is refused, not answered (issue #778).

`TableFunction::evaluate_at` had no NaN test. Every comparison with NaN is
false, so a NaN index fell through both endpoint checks and `upper_bound()`
returned `end()`. A step table then answered its LAST value: the run finished
with finite, plausible, wrong numbers and no warning. A linear table read
`xs_[n]` and `ys_[n]`, one past the end (heap-buffer-overflow under ASan).

Both now propagate NaN, so the RHS non-finite guard refuses the run the way it
refuses any other uncomputable rate (#580). The interpreter and the compiled
path share this one evaluator (codegen calls back into it), so both are pinned.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest

# d = sqrt(-1) is NaN, and it indexes the table that sets A's decay rate. With
# the old step fallback the run gave A = 10*exp(-7t): the last table value.
_NET = """
begin parameters
    1 k -1.0
    2 d sqrt(k)
end parameters
begin functions
    1 F() tfun([0,1,2],[5,6,7],d{method})
end functions
begin species
    1 A() 10
end species
begin reactions
    1 1 0 F
end reactions
begin groups
    1 Atot 1
end groups
"""

_METHODS = {"linear": "", "step": ', method=>"step"'}


def _nan_indexed_model(tmp_path: Path, method: str) -> bngsim.Model:
    net = tmp_path / f"nan_{method}.net"
    net.write_text(textwrap.dedent(_NET.format(method=_METHODS[method])).strip() + "\n")
    return bngsim.Model.from_net(str(net))


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
@pytest.mark.parametrize("method", sorted(_METHODS))
def test_a_nan_index_refuses_the_run(tmp_path, monkeypatch, method, codegen):
    if codegen:
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    m = _nan_indexed_model(tmp_path, method)
    with pytest.raises(bngsim.SimulationError, match="non-finite"):
        bngsim.Simulator(m, "ode").run((0.0, 1.0), 3)


@pytest.mark.parametrize("method", sorted(_METHODS))
def test_a_finite_index_is_unchanged(tmp_path, method):
    """The guard must not touch an ordinary index: d = 1.5 sits between the
    table's points, so step gives 6 and linear gives 6.5."""
    m = _nan_indexed_model(tmp_path, method)
    m.set_param("k", 2.25)  # d = sqrt(k) = 1.5
    r = bngsim.Simulator(m, "ode").run((0.0, 1.0), 2, rtol=1e-10, atol=1e-12)
    rate = {"linear": 6.5, "step": 6.0}[method]
    atot = float(np.asarray(r.observables)[-1, 0])
    assert atot == pytest.approx(10.0 * math.exp(-rate), rel=1e-8)


def test_a_nan_x_value_is_refused_when_the_table_is_built(tmp_path):
    """A NaN x passes the monotonicity check (both `<=` tests are false) and then
    breaks the binary search the same way a NaN index does."""
    m = _nan_indexed_model(tmp_path, "linear")
    with pytest.raises(bngsim.ModelError, match=r"x\[1\] is NaN"):
        m.add_table_function("T", times=[0.0, float("nan"), 2.0], values=[1.0, 2.0, 3.0])
