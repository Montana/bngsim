"""A dilution term that reads a species' rate keeps the analytic Jacobian and
the analytic sensitivity RHS (issue #1124, item 7).

``C1 := 2 + r*P`` sizes a compartment with S in it, so S's dilution term is
``-S·(r·rate_of__P)/C1``. ``rate_of__P`` is the engine's live dP/dt, which no
derivation could differentiate: the analytic Jacobian and the analytic
sensitivity RHS were declined, a sensitivity run went to CVODES' difference
quotient (291,511 steps at rtol 1e-11 for 284), and ``sensitivity_ic`` could
not run at all. dP/dt is the sum of P's rate laws, which is what the
derivations now read in its place.

Oracle: closed forms. P = e^(-0.3t), so C1 = 2 + r·e^(-0.3t). J2 takes S's
amount at 0.2 a unit amount, so the amount is 5·C1(0)·e^(-0.2t) and
[S] = 5(2 + r)e^(-0.2t)/C1; d[S]/dr = 10·e^(-0.2t)(1 − e^(-0.3t))/C1², and
[S] is linear in its start, so d[S]/d[S](0) = [S]/5.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_TEXT = """
compartment C0 = 1; compartment C1; r = 0.5; C1 := 2 + r*P;
species P in C0 = 1; species S in C1 = 5;
J1: P -> ; 0.3*P*C0; J2: S -> ; 0.2*S*C1;
"""
_TIGHT = {"rtol": 1e-11, "atol": 1e-13}


def _c1(t, r=0.5):
    return 2 + r * np.exp(-0.3 * t)


def _s(t, r=0.5):
    return 5 * (2 + r) * np.exp(-0.2 * t) / _c1(t, r)


def _ds_dr(t, r=0.5):
    return 10 * np.exp(-0.2 * t) * (1 - np.exp(-0.3 * t)) / _c1(t, r) ** 2


def _model():
    return bngsim.Model.from_antimony_string(_TEXT)


def test_the_sensitivity_rhs_is_analytic():
    sim = bngsim.Simulator(_model(), method="ode", sensitivity_params=["r"])
    assert sim.has_analytic_sens_rhs
    result = sim.run(t_span=(0.0, 10.0), n_points=11, **_TIGHT)
    t = np.asarray(result.time)
    w = list(result.species_names).index("S")
    np.testing.assert_allclose(np.asarray(result.species)[:, w], _s(t), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(result.sensitivities)[:, w, 0], _ds_dr(t), rtol=1e-6)
    # On the difference quotient this run took 291,511 steps.
    assert result.solver_stats["n_steps"] < 5_000


def test_an_initial_value_column_runs():
    """It was refused: no sensitivity RHS had been built for it."""
    result = bngsim.Simulator(_model(), method="ode", sensitivity_ic=["S"]).run(
        t_span=(0.0, 10.0), n_points=11, **_TIGHT
    )
    t = np.asarray(result.time)
    w = list(result.species_names).index("S")
    np.testing.assert_allclose(np.asarray(result.sensitivities_ic)[:, w, 0], _s(t) / 5, rtol=1e-7)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
def test_the_analytic_jacobian_attaches(codegen):
    model = _model()
    result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
        t_span=(0.0, 10.0), n_points=11, **_TIGHT
    )
    assert model.analytical_jacobian_status == "complete"
    t = np.asarray(result.time)
    w = list(result.species_names).index("S")
    np.testing.assert_allclose(np.asarray(result.species)[:, w], _s(t), rtol=1e-8)
