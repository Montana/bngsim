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


def test_a_mass_action_law_on_p_is_written_out_too():
    """The loader reads ``k1*P*C0`` as an Elementary reaction, k1·P, and left
    out, the model as usually written stayed on the difference quotient. The
    k1 column goes through P's rate: d[S]/dk1 = 5(2 + r)·r·t·e^(-0.2t)·P/C1²."""
    text = _TEXT.replace("r = 0.5;", "r = 0.5; k1 = 0.3;").replace("0.3*P*C0", "k1*P*C0")
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["r", "k1"]
    )
    assert sim.has_analytic_sens_rhs
    result = sim.run(t_span=(0.0, 10.0), n_points=11, **_TIGHT)
    t = np.asarray(result.time)
    w = list(result.species_names).index("S")
    sens = np.asarray(result.sensitivities)[:, w, :]
    ds_dk1 = 5 * 2.5 * 0.5 * t * np.exp(-0.2 * t) * np.exp(-0.3 * t) / _c1(t) ** 2
    np.testing.assert_allclose(sens[:, 0], _ds_dr(t), rtol=1e-6)
    np.testing.assert_allclose(sens[:, 1], ds_dk1, rtol=1e-6, atol=1e-12)
    assert result.solver_stats["n_steps"] < 5_000


def test_the_compiled_jacobian_reads_the_rate_too():
    """A cross-compartment reaction into a rate-ruled compartment makes the
    compiled Jacobian recompute func[], the dilution term's included, and that
    recomputation has no live dx/dt. Once the interpreted Jacobian attached, the
    token went into the C as an undeclared name: the build failed, and the
    sensitivity run raised. A and B leave P and S as they were."""
    text = _TEXT + (
        "compartment Cv = 2; Cv' = 0.1;\n"
        "species A in C0 = 1; species B in Cv = 1;\n"
        "Jx: A -> B; 0.1*A*C0;\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text),
        method="ode",
        codegen=True,
        sensitivity_params=["r"],
    )
    assert sim.has_analytic_sens_rhs
    result = sim.run(t_span=(0.0, 10.0), n_points=11, **_TIGHT)
    t = np.asarray(result.time)
    w = list(result.species_names).index("S")
    np.testing.assert_allclose(np.asarray(result.species)[:, w], _s(t), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(result.sensitivities)[:, w, 0], _ds_dr(t), rtol=1e-6)


def test_a_rate_beside_such_a_reaction_compiles_where_only_an_output_reads_it():
    """``y := rateOf(P)`` read by no rate law. The interpreted Jacobian never
    reads y, so it attached on main too, and the compiled one wrote
    ``func[0] = rate_of__P;``: the build failed under ``codegen=True``. y is
    -0.3·P."""
    text = (
        "compartment C0 = 1; compartment Cv = 2; Cv' = 0.1;\n"
        "species P in C0 = 1; species A in C0 = 1; species B in Cv = 1;\n"
        "J1: P -> ; 0.3*P*C0; Jx: A -> B; 0.1*A*C0;\n"
        "y := rateOf(P);\n"
    )
    result = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", codegen=True
    ).run(t_span=(0.0, 5.0), n_points=6, **_TIGHT)
    p = np.asarray(result.species)[:, list(result.species_names).index("P")]
    y = np.asarray(result.expressions)[:, result.expression_names.index("y")]
    np.testing.assert_allclose(p, np.exp(-0.3 * np.asarray(result.time)), rtol=1e-8)
    np.testing.assert_allclose(y, -0.3 * p, rtol=1e-8)


def test_an_amount_whose_rate_of_reports_an_amount():
    """P an amount in C0 = 2: its rateOf is d(amount)/dt, the engine's sum times
    C0. Left out, the run went to the difference quotient for millions of steps.
    P = e^(-k1·t), so [S] = 10·e^(-0.2t)/(1 + P) and
    d[S]/dk1 = 10·e^(-0.2t)·t·P/(1 + P)²; C0 moves neither."""
    text = """
compartment C0 = 2; compartment C; C := 1 + P; k1 = 0.3;
substanceOnly species P in C0 = 1; species S in C = 5;
J1: P -> ; k1*P; J2: S -> ; 0.2*S*C;
"""
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["k1", "C0"]
    )
    assert sim.has_analytic_sens_rhs
    result = sim.run(t_span=(0.0, 10.0), n_points=11, **_TIGHT)
    t = np.asarray(result.time)
    p = np.exp(-0.3 * t)
    w = list(result.species_names).index("S")
    sens = np.asarray(result.sensitivities)[:, w, :]
    np.testing.assert_allclose(
        np.asarray(result.species)[:, w], 10 * np.exp(-0.2 * t) / (1 + p), rtol=1e-8
    )
    np.testing.assert_allclose(sens[:, 0], 10 * np.exp(-0.2 * t) * t * p / (1 + p) ** 2, rtol=1e-6)
    np.testing.assert_allclose(sens[:, 1], 0.0, atol=1e-9)
    assert result.solver_stats["n_steps"] < 5_000
