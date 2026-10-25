"""rateOf of a species a cross-compartment reaction changes is written out (issue #1266).

``rateof_rate_laws`` writes ``rate_of__X`` out as X's dX/dt, so a law that reads
``rateOf(X)`` can be differentiated (#1143). It wrote nothing for a species a
cross-compartment reaction changes, and a law reading its rate kept the token:
the analytic Jacobian and the analytic sensitivity RHS declined, and a
sensitivity run fell back to CVODES' difference quotient. At rtol 1e-10 it took
59,927 steps where the same model with the rate written out by hand takes 389.

The engine (``compute_derivs_core``) divides such a reaction's rate by each
species' own divisor: the live volume where one is promoted, its size
otherwise. The written dX/dt does the same, the size by name where it is the
species' size parameter, so a write to it moves the rate as it moves the
engine's.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

READS = """
compartment C1 = 2; compartment C2 = 3; k = 0.3; g = 0.7;
species A in C1 = 1; species B in C2 = 0; species S in C1 = 0;
J: A -> B; k*A*C1;
S' = g*rateOf(B);
"""
# The same model, with B's rate (J's rate over C2) written out by hand.
WRITTEN = READS.replace("rateOf(B)", "(k*A*C1/C2)")
# C2 grows: B's divisor is its live volume, and B is diluted as it grows.
LIVE = READS.replace("compartment C2 = 3;", "compartment C2 = 3; C2' = 0.05;")

TIMES = list(np.linspace(0.0, 20.0, 21))


def _run(text, params, *, rtol=1e-10, atol=1e-14, sets=()):
    model = bngsim.Model.from_antimony_string(text)
    for name, value in sets:
        model.set_param(name, value)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    return sim, sim.run(sample_times=TIMES, rtol=rtol, atol=atol)


def _central(text, name, rel=1e-4, sets=()):
    """Central difference of plain runs, with respect to *name*, after *sets*.
    At rtol 1e-12 a run is good to about 1e-12, so the difference is good to
    about 1e-12 / (2·rel·|p|): 7e-9 here, under the 1e-7 it is judged to."""

    def run(scale):
        model = bngsim.Model.from_antimony_string(text)
        for n, v in sets:
            model.set_param(n, v)
        value = model.get_param(name)
        model.set_param(name, value * scale)
        r = bngsim.Simulator(model, method="ode").run(sample_times=TIMES, rtol=1e-12, atol=1e-15)
        return np.asarray(r.species), value

    (up, value), (down, _) = run(1 + rel), run(1 - rel)
    return (up - down) / (2 * rel * value)


@pytest.mark.parametrize("text", [READS, LIVE], ids=["static", "live"])
def test_the_jacobian_and_the_sensitivity_rhs_are_analytic(text):
    model = bngsim.Model.from_antimony_string(text)
    model.prepare_analytical_jacobian()
    assert model.analytical_jacobian_status == "complete"
    sim, _ = _run(text, ["k", "g"])
    assert sim.sens_rhs_decline_reason is None


def test_it_is_the_hand_written_model():
    """Same trajectory, same sensitivities, about the same steps: 59,927 on
    main, for the hand-written model's 389."""
    _, reads = _run(READS, ["k", "g"])
    _, written = _run(WRITTEN, ["k", "g"])
    np.testing.assert_allclose(reads.species, written.species, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(reads.sensitivities, written.sensitivities, rtol=1e-6, atol=1e-12)
    assert reads.solver_stats["n_steps"] <= 2 * written.solver_stats["n_steps"]


@pytest.mark.parametrize(
    ("text", "sets"),
    [(READS, ()), (READS, (("C2", 6.0),)), (LIVE, ())],
    ids=["static", "static-C2-set", "live"],
)
def test_the_sensitivities_are_the_derivative(text, sets):
    """Against central differences of plain runs, for k and g. With C2 written
    after loading, the divisor read by name has moved with it. (In the live
    model C2 is a species, its size the rate rule moves.)"""
    _, r = _run(text, ["k", "g"], sets=sets)
    s = np.asarray(r.sensitivities)
    for j, name in enumerate(["k", "g"]):
        np.testing.assert_allclose(
            s[:, :, j], _central(text, name, sets=sets), rtol=1e-5, atol=1e-7
        )


def test_the_central_differences_judge_the_hand_written_model():
    """The oracle's control: on the model with the rate written by hand, whose
    route was analytic before this fix, the same check passes."""
    _, r = _run(WRITTEN, ["k", "g"])
    s = np.asarray(r.sensitivities)
    for j, name in enumerate(["k", "g"]):
        np.testing.assert_allclose(s[:, :, j], _central(WRITTEN, name), rtol=1e-5, atol=1e-7)


def test_the_sensitivity_to_the_divisor_itself():
    """d/dC2: B's divisor is C2 read by name, so the -rate/C2^2 term is in the
    derivative; read as the number 3 it would be left out."""
    sim, r = _run(READS, ["C2"])
    assert sim.sens_rhs_decline_reason is None
    s = np.asarray(r.sensitivities)[:, :, 0]
    np.testing.assert_allclose(s, _central(READS, "C2"), rtol=1e-5, atol=1e-7)
    _, w = _run(WRITTEN, ["C2"])
    np.testing.assert_allclose(s, np.asarray(w.sensitivities)[:, :, 0], rtol=1e-6, atol=1e-12)


_J = "J: A -> B; k*A*C1;"
# Each: the reactions, the rule that reads rateOf, and the whole model with the
# rate written out by hand.
_CASES = {
    # rateOf of the reactant: its divisor is its own compartment's, C1
    "reactant": (_J, "S' = g*rateOf(A);", _J + " S' = g*(-k*A*C1/C1);"),
    # a coefficient of 2, both sides read
    "two-A": (
        "J: 2A -> B; k*A^2*C1;",
        "S' = g*rateOf(A) + rateOf(B);",
        "J: 2A -> B; k*A^2*C1; S' = g*(-2*k*A^2*C1/C1) + (k*A^2*C1/C2);",
    ),
    # a reversible rate, negative where B's term leads
    "reversible": (
        "J: A -> B; k*A*C1 - g*B*C2;",
        "S' = rateOf(B);",
        "J: A -> B; k*A*C1 - g*B*C2; S' = (k*A*C1 - g*B*C2)/C2;",
    ),
    # rateOf in a reaction's rate, not a rule
    "in-a-rate": (_J + " K: S -> ; g*S*rateOf(B)*C1;", "", _J + " K: S -> ; g*S*(k*A*C1/C2)*C1;"),
    # C2 grows: B's rate is J's over the live C2, less its dilution
    "live": (
        _J + " C2' = 0.05;",
        "S' = g*rateOf(B);",
        _J + " C2' = 0.05; S' = g*(k*A*C1/C2 - B*0.05/C2);",
    ),
}
_HEAD = """
compartment C1 = 2; compartment C2 = 3; k = 0.3; g = 0.7;
species A in C1 = 1; species B in C2 = 0.5; species S in C1 = 0;
"""


@pytest.mark.parametrize("case", list(_CASES))
def test_each_shape_is_the_hand_written_model(case):
    """A reactant's rate, a coefficient of 2, a reversible rate, rateOf in a
    rate, and a growing divisor: each the model with the rate written by hand,
    sizes included. Before, each declined to the difference quotient, which at
    rtol 1e-11 took 7 to 53 million steps and was 5e-3 off in places."""
    rxn, rule, hand = _CASES[case]
    params = ["k", "g", "C1"] + ([] if case == "live" else ["C2"])
    sim, reads = _run(_HEAD + rxn + "\n" + rule, params, rtol=1e-11, atol=1e-15)
    assert sim.sens_rhs_decline_reason is None
    _, written = _run(_HEAD + hand, params, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(reads.species, written.species, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(reads.sensitivities, written.sensitivities, rtol=1e-8, atol=1e-13)
