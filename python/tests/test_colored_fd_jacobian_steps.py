"""KLU's colored FD Jacobian takes each column's step on the error test's scale
(issue #1254).

A model on the KLU route without an analytic Jacobian has its Jacobian filled
by colored finite differences (``colored_fd_jacobian``), CVODE having no
difference quotient of its own for a sparse matrix. Each column's step was
``sqrt(eps)*max(|y_j|, 1)``: a floor of 1 in absolute terms, so a species of
1e-9 was moved by fifteen times itself, and the matrix KLU factored was the
Jacobian somewhere else. CVODE's own dense difference quotient, on the dense
route, takes ``max(sqrt(eps)*|y_j|, min_inc/ewt_j)``: the error weights' scale.
The colored one now does too.

* #1254's equality, ``if(Bobs==0, kb, 0)``: Y(10) = 2 ln 5 exactly. KLU with
  the FD Jacobian gave 3.215642 at rtol 1e-8 and 3.203796 at 1e-10 (the error
  grew as the tolerance tightened), where dense with the same Jacobian was
  within 6e-6. It is now as close as dense is.
* A dimerization chain at nanomolar scale: right on KLU either way, but the
  wrong Jacobian cost 781 steps and 100 Jacobians, where dense took 299 and 40.

Found by #1257's ``klu_vs_dense`` check, which also had four BioModels fail or
read 3% off on this route (BIOMD0000000827, MODEL1006230034, MODEL2002030001,
MODEL1912090001); those now agree with dense to 2e-4 or better.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")

_EQUALITY = """begin parameters
    1 a  0.5
    2 q  2.0
    3 kb 1.0
    4 kp 1.0
end parameters
begin species
    1 A() 10
    2 B() 0
    3 Y() 0
end species
begin functions
    1 fB() kp*if(q-Aobs>0,q-Aobs,0)
    2 fY() if(Bobs==0,kb,0)
end functions
begin reactions
    1 1 0 a
    2 0 2 fB
    3 0 3 fY
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
end groups
"""

KLU_FD = {"jacobian": "fd", "force_sparse_linear_solver": True}


def _net(tmp_path, text):
    path = tmp_path / "m.net"
    path.write_text(text)
    return str(path)


def _y10(path, rtol, n_points, **kw):
    r = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", **kw).run(
        t_span=(0.0, 10.0), n_points=n_points, rtol=rtol, atol=rtol / 100
    )
    return r.species[-1][list(r.species_names).index("Y()")]


@pytest.mark.parametrize("n_points", [101, 1001])
@pytest.mark.parametrize("rtol", [1e-8, 1e-10])
def test_the_equality_on_klu_is_as_right_as_on_dense(tmp_path, rtol, n_points):
    """B is 0 exactly until A falls below q at t* = 2 ln 5, and Y grows at kb
    while B == 0, so Y(10) = t*. KLU with the FD Jacobian was 1.0e-3 to 1.2e-2
    off it, where dense with the FD Jacobian was 1.4e-7 to 5.7e-6; it is now as
    close as dense is, or closer.

    Not asked: that either is exact. An equality on a state has no root to
    stop the solver at, and where its switch flips depends on where the steps
    fall, on every route: dense with the analytic Jacobian is 1.1e-3 off at
    rtol 1e-10 and 1001 samples."""
    path = _net(tmp_path, _EQUALITY)
    exact = 2 * math.log(5)
    dense = abs(_y10(path, rtol, n_points, jacobian="fd") - exact)
    klu = abs(_y10(path, rtol, n_points, **KLU_FD) - exact)
    assert klu <= max(2 * dense, 1e-9 * exact), (klu, dense)


# 2 A_i -> A_{i+1} at 1e9 /M/s from A_0 = 1 nM: O(1) dynamics in a second
_CHAIN = (
    "begin parameters\n    1 k 1e9\nend parameters\nbegin species\n"
    + "".join(f"    {i + 1} A{i}() {'1e-9' if i == 0 else '0'}\n" for i in range(6))
    + "end species\nbegin reactions\n"
    + "".join(f"    {i + 1} {i + 1},{i + 1} {i + 2} k\n" for i in range(5))
    + "end reactions\n"
)


def _run(path, **kw):
    return bngsim.Simulator(bngsim.Model.from_net(path), method="ode", **kw).run(
        t_span=(0.0, 20.0), n_points=21, rtol=1e-8, atol=1e-20
    )


def test_a_nanomolar_model_takes_the_steps_dense_takes(tmp_path):
    """The step on the error weights' scale gives the Jacobian dense's
    difference quotient gives, and Newton converges as it does there. Before,
    a species of 1e-9 was stepped by 1.5e-8: right in the end, at 2.6x the
    steps and 2.5x the Jacobians."""
    path = _net(tmp_path, _CHAIN)
    dense = _run(path, jacobian="fd")
    klu = _run(path, **KLU_FD)
    np.testing.assert_allclose(klu.species, dense.species, rtol=1e-6, atol=1e-20)
    assert klu.solver_stats["n_steps"] <= 1.2 * dense.solver_stats["n_steps"]
    assert klu.solver_stats["n_jac_evals"] <= 1.2 * dense.solver_stats["n_jac_evals"]


def test_the_analytic_jacobian_on_klu_is_untouched(tmp_path):
    """The control: the analytic route never reads the colored step."""
    path = _net(tmp_path, _CHAIN)
    a = _run(path, force_sparse_linear_solver=True)
    b = _run(path, jacobian="auto")
    np.testing.assert_allclose(a.species, b.species, rtol=1e-6, atol=1e-20)
