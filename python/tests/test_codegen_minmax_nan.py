"""GH #660: compiled ``max``/``min`` mean what ExprTk's do.

ExprTk's ``max``/``min`` are ``std::max``/``std::min``: ``(a < b) ? b : a``,
which returns the FIRST argument whenever the comparison is false, so a NaN in
first position wins. Codegen used to rename them to C's ``fmax``/``fmin``,
which return the non-NaN argument. A rate law the interpreter refuses as a NaN
right-hand side then ran to completion compiled, with plausible numbers.
"""

from __future__ import annotations

import os

import bngsim
import numpy as np
import pytest

_NET = """begin parameters
    1 k       0.3  # Constant
end parameters
begin functions
    1 law() {body}
end functions
begin species
    1 A() 5.0
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 A                    1
end groups
"""


def _run(tmp_path, body: str, codegen: bool):
    path = os.path.join(tmp_path, f"m_{abs(hash((body, codegen)))}.net")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_NET.format(body=body))
    m = bngsim.Model.from_net(path)
    sim = bngsim.Simulator(m, method="ode", codegen=codegen)
    return np.asarray(sim.run(t_span=(0.0, 2.0), n_points=3, timeout=60.0).observables["A"])


# ln(A - 10) is NaN for the whole run (A starts at 5 and decays).
@pytest.mark.parametrize("body", ["k*max(ln(A-10),1)", "k*min(ln(A-10),1)"])
def test_nan_first_argument_is_refused_by_both_engines(tmp_path, body):
    with pytest.raises(bngsim.SimulationError):
        _run(tmp_path, body, codegen=False)
    # fmax/fmin returned 1 here, so the compiled run used to complete.
    with pytest.raises(bngsim.SimulationError):
        _run(tmp_path, body, codegen=True)


@pytest.mark.parametrize("body", ["k*max(1,ln(A-10))", "k*min(1,ln(A-10))"])
def test_nan_second_argument_agrees_across_engines(tmp_path, body):
    interp = _run(tmp_path, body, codegen=False)
    compiled = _run(tmp_path, body, codegen=True)
    np.testing.assert_allclose(compiled, interp, rtol=1e-6)
    np.testing.assert_allclose(interp, 5.0 * np.exp(-0.3 * np.array([0.0, 1.0, 2.0])), rtol=1e-5)
