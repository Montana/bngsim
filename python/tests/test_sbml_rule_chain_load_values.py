"""An assignment rule that reads a later rule's target gets its load-time value (issue #994).

The loader evaluates assignment rules for their load-time values in one pass, in
document order, and repeated the pass to a fixed point only for a model with an
``initialAssignment``. Antimony writes ``v := 2*p; c2 := v`` as ``c2`` then
``v``, so ``c2 := v`` read ``v`` unset and the compartment loaded with volume
factor 1 instead of 4. Every reaction across compartments then divided by 1:
``run()`` and ``steady_state()`` ended at ``[0.8, 0.4]`` for ``[0.6, 0.3]``, with
nothing raised. The settle loop now runs for every model, and long enough for a
chain of any length.
"""

from __future__ import annotations

import bngsim
import libsbml
import numpy as np
import pytest

# A (in c1 = 1) <-> B (in c2, size 4 every way it is written). The exact steady
# state is A* = k2*T/(k1 + k2) = 0.6 and B* = (T - A*)/4 = 0.3, T = 1 + 0.2*4.
TEXT = (
    "compartment c1, c2; c1 = 1; {c2};\n"
    "species A in c1, B in c2; A = 1; B = 0.2; k1 = 1; k2 = 0.5;\n"
    "R1: A -> B; k1*A*c1\nR2: B -> A; k2*B*c2\n"
)
STEADY = [0.6, 0.3]

SPELLINGS = {
    "constant": "c2 = 4",
    "one rule": "p = 2; c2 := 2*p",
    "rule over a rule": "p = 2; v := 2*p; c2 := v",
    "chain of three": "p = 2; v := 2*p; w := v; c2 := w",
}


def _volume_factors(model: bngsim.Model) -> list[float]:
    return [s["volume_factor"] for s in model._core.codegen_data()["species"]]


def _end(model: bngsim.Model) -> np.ndarray:
    out = bngsim.Simulator(model, "ode").run(t_span=(0, 200.0), n_points=2, rtol=1e-11, atol=1e-13)
    return np.asarray(out.species)[-1]


def _sbml(c2: str, *, reverse_rules: bool) -> str:
    """The Antimony model as SBML, with its assignment rules in document order
    or reversed, so each test sees a rule both before and after what it reads."""
    import antimony

    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(TEXT.format(c2=c2)) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    model = doc.getModel()
    rules = [model.getRule(j).clone() for j in range(model.getNumRules())]
    while model.getNumRules():
        model.removeRule(0)
    for rule in reversed(rules) if reverse_rules else rules:
        assert model.addRule(rule) == libsbml.LIBSBML_OPERATION_SUCCESS
    return libsbml.writeSBMLToString(doc)


@pytest.mark.parametrize("c2", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_the_compartment_loads_at_its_size(c2: str) -> None:
    model = bngsim.Model.from_antimony_string(TEXT.format(c2=c2))
    assert model.get_param("c2") == pytest.approx(4.0)
    assert _volume_factors(model) == pytest.approx([1.0, 4.0])
    np.testing.assert_allclose(_end(model), STEADY, rtol=1e-6)


@pytest.mark.parametrize("reverse_rules", [False, True], ids=["as written", "reversed"])
@pytest.mark.parametrize(
    "c2", [SPELLINGS["rule over a rule"], SPELLINGS["chain of three"]], ids=["two", "three"]
)
def test_either_rule_order_in_the_document(c2: str, reverse_rules: bool) -> None:
    model = bngsim.Model.from_sbml_string(_sbml(c2, reverse_rules=reverse_rules))
    assert _volume_factors(model) == pytest.approx([1.0, 4.0])
    np.testing.assert_allclose(_end(model), STEADY, rtol=1e-6)


def test_a_chain_longer_than_ten_links_settles() -> None:
    # Written against document order, each pass settles one more link; the loop
    # used to stop after 10 passes whatever the chain's length.
    links = ["c2 := w13"] + [f"w{i} := w{i - 1}" for i in range(13, 1, -1)] + ["w1 := 2*p"]
    model = bngsim.Model.from_antimony_string(TEXT.format(c2="p = 2; " + "; ".join(links)))
    assert _volume_factors(model) == pytest.approx([1.0, 4.0])
    np.testing.assert_allclose(_end(model), STEADY, rtol=1e-6)


def test_steady_state_is_the_systems_own() -> None:
    model = bngsim.Model.from_antimony_string(TEXT.format(c2=SPELLINGS["rule over a rule"]))
    result = bngsim.Simulator(model, "ode").steady_state()
    assert result.converged
    np.testing.assert_allclose(np.asarray(result.concentrations).ravel(), STEADY, rtol=1e-6)


def test_steady_state_sensitivities_are_refused_not_wrong() -> None:
    # With both volume factors 1 the model looked like one with no law across
    # sizes, and steady_state returned columns [-0.2667, 0.2667] for
    # [-0.4, 0.1]. At the right sizes it is a rule-sized compartment under a law
    # across sizes, which #745/#992 refuse.
    model = bngsim.Model.from_antimony_string(TEXT.format(c2=SPELLINGS["rule over a rule"]))
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#745"):
        bngsim.Simulator(model, "ode").steady_state(sensitivity_params=["k1"])
