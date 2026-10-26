"""A moving size whose rate bngsim can know is diluted, not refused (issue #1281).

#1204 refuses, by name, an assignment-rule size whose rate V̇ bngsim cannot
derive, where main left its species undiluted with no word. Three shapes it
refused have a rate bngsim can know:

* ``rateOf`` of a rate-rule variable ``P``: its rate is the rate of ``P``'s
  right-hand side, by the chain rule the size's own rate is written with;
* ``rateOf`` of a species in a compartment whose size holds: the sum of its
  reactions' laws, each times its stoichiometry, over the size, whose rate is the
  same sum over the laws' rates;
* a rule on a cycle of rules that hold their values (#621), whose rate is 0.

The oracle is the amount. ``X`` leaves at ``0.25*X*cv``, an amount of
``0.25·A`` a unit of time, so ``A = [X]·cv`` is ``A(0)·exp(-t/4)`` however the
size moves, and ``[X]`` is that over ``cv(t)``, written below in closed form for
each model. Undiluted, ``[X]`` decays as ``exp(-t/4)`` whatever ``cv`` does.

What still cannot be derived is still refused: a cycle of rules that move, a
cycle of rates, and ``rateOf`` of a species whose own compartment moves.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

antimony = pytest.importorskip("antimony")

# S3's compartment c is of size 2, so that a rate written without its division
# by the size is not the right one by chance.
_HEAD = "compartment c = 2; species S3 in c = 2; species X in cv = 4; J: X -> ; 0.25*X*cv;\n"
TIMES = np.linspace(0.0, 4.0, 9)

# name -> (the rest of the model, cv(t))
CASES = {
    # an amount of 0.1 made a unit of time in c, so rateOf(S3) = 0.05 and cv
    # holds at 2.05
    "species-constant": ("cv := 2 + rateOf(S3); P0: -> S3; 0.1;", lambda t: 2.05 + 0 * t),
    # an amount of 0.1·[S3] a unit of time leaves c, so [S3] = 2 exp(-t/20)
    # and rateOf(S3) = -0.1 exp(-t/20)
    "species-decaying": (
        "cv := 2 + rateOf(S3); D: S3 -> ; 0.1*S3;",
        lambda t: 2 - 0.1 * np.exp(-t / 20),
    ),
    # rateOf(S3) stays in [-0.1, 0], so the max is 1
    "species-max": ("cv := 2 + max(rateOf(S3), 1); D: S3 -> ; 0.1*S3;", lambda t: 3 + 0 * t),
    "rate-rule-constant": ("cv := 2 + rateOf(P); P = 1; P' = 0.1;", lambda t: 2.1 + 0 * t),
    # P = exp(-t/10), so rateOf(P) = -0.1 exp(-t/10)
    "rate-rule-moving": (
        "cv := 2 + rateOf(P); P = 1; P' = -0.1*P;",
        lambda t: 2 - 0.1 * np.exp(-t / 10),
    ),
}


def _sbml(rest):
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString("model m\n" + _HEAD + rest + "\nend") >= 0, (
        antimony.getLastError()
    )
    return antimony.getSBMLString("m")


def _x(sbml):
    model = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(model, method="ode").run(sample_times=list(TIMES), rtol=1e-10, atol=1e-12)
    return np.asarray(r.species)[:, list(r.species_names).index("X")]


def _diluted(cv):
    """[X](t) for the amount A(0) exp(-t/4) in cv(t), from [X](0) = 4."""
    return 4 * cv(0.0) * np.exp(-TIMES / 4) / cv(TIMES)


@pytest.mark.parametrize("case", list(CASES))
def test_the_size_is_followed(case):
    rest, cv = CASES[case]
    np.testing.assert_allclose(_x(_sbml(rest)), _diluted(cv), rtol=1e-7, atol=1e-10)


def test_the_moving_rate_rule_is_libroadrunners():
    """Where libRoadRunner's start agrees (no rateOf read at the start of a
    species), its whole trajectory: 4, 3.0997, 2.4032 at t = 0, 1, 2."""
    roadrunner = pytest.importorskip("roadrunner")
    sbml = _sbml(CASES["rate-rule-moving"][0])
    rr = roadrunner.RoadRunner(sbml)
    rr.integrator.relative_tolerance = 1e-10
    rr.integrator.absolute_tolerance = 1e-12
    rr.timeCourseSelections = ["time", "[X]"]
    ref = rr.simulate(0.0, 4.0, len(TIMES))[:, 1]
    np.testing.assert_allclose(_x(sbml), ref, rtol=1e-8)


def _with_rules(rest, rules, rate=False):
    """The model, with rules Antimony will not write: a cycle of assignment
    rules, or of rate rules, each over a parameter of value 1 (declared where
    the model does not declare it yet)."""
    import libsbml

    doc = libsbml.readSBMLFromString(_sbml(rest))
    model = doc.getModel()
    for var, formula in rules:
        p = model.getParameter(var) or model.createParameter()
        p.setId(var)
        p.setConstant(False)
        p.setValue(1.0)
        rule = model.createRateRule() if rate else model.createAssignmentRule()
        rule.setVariable(var)
        rule.setMath(libsbml.parseL3Formula(formula))
    return libsbml.writeSBMLToString(doc)


def test_a_cycle_of_rules_that_hold():
    """p1 = 1 + 0.1·p2 and p2 = 1 + 0.1·p1 hold at p1 = 1.1/0.99 (#621), so cv
    = 1 + 0.1·t + 0.1·p1 moves with the clock alone. Before, the cycle made
    cv's rate unknown and the model was refused."""
    sbml = _with_rules(
        "cv := 1 + 0.1*time + 0.1*p1;",
        [("p1", "1 + 0.1*p2"), ("p2", "1 + 0.1*p1")],
    )
    p1 = 1.1 / 0.99
    np.testing.assert_allclose(
        _x(sbml), _diluted(lambda t: 1 + 0.1 * t + 0.1 * p1), rtol=1e-7, atol=1e-10
    )


def test_a_cycle_of_rules_that_hold_beside_a_jump():
    """The size's jump is found through a rule that holds: written out to find
    its jumps, the size keeps p1 by name where the cycle stops the writing.
    [S3] = 2 exp(-t/20) falls below 1 at t = 20 ln 2, where cv drops by 1 and [X]
    jumps up, its amount kept. Had the size not been written out, its jump
    would have been skipped with no word, and [X] carried across as it was."""
    sbml = _with_rules(
        "cv := piecewise(1, S3 < 1, 2) + 0.1*p1; D: S3 -> ; 0.1*S3;",
        [("p1", "1 + 0.1*p2"), ("p2", "1 + 0.1*p1")],
    )
    p1 = 1.1 / 0.99
    times = np.linspace(0.0, 20.0, 21)
    model = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(model, method="ode").run(sample_times=list(times), rtol=1e-10, atol=1e-12)
    x = np.asarray(r.species)[:, list(r.species_names).index("X")]
    cv = np.where(2 * np.exp(-times / 20) < 1, 1.0, 2.0) + 0.1 * p1
    np.testing.assert_allclose(x, 4 * cv[0] * np.exp(-times / 4) / cv, rtol=1e-7, atol=1e-10)


# Still refused, each by name: a rate bngsim cannot derive. (SBML rules out
# the cycle of rates, L3V2 4.9.5, so Antimony will not write it; bngsim must
# still stop, not recurse.)
STILL_REFUSED = {
    # a cycle of rules that move: p1 reads the clock
    "moving-cycle": (
        "cv := 1 + 0.1*p1;",
        [("p1", "time + 0.1*p2"), ("p2", "1 + 0.1*p1")],
        False,
        "(?s)cannot follow to its rate of change.*cycle",
    ),
    # a cycle of rates: P' = rateOf(Q), Q' = rateOf(P)
    "rate-cycle": (
        "cv := 2 + rateOf(P);",
        [("P", "-0.1*P + 0.01*rateOf(Q)"), ("Q", "rateOf(P)")],
        True,
        "assignment rule for 'cv' cannot be evaluated",
    ),
    # rateOf of a species whose own compartment moves, by a rate rule (SBML
    # rules one out whose compartment an assignment rule sets, L3V2 3.4.6)
    "species-in-a-moving-size": (
        "compartment c2 = 1; c2' = 0.1; species S4 in c2 = 1; D4: S4 -> ; 0.1*S4;"
        " cv := 2 + rateOf(S4);",
        [],
        False,
        r"(?s)cannot follow to its rate of change.*rateOf\(S4\)",
    ),
}


@pytest.mark.parametrize("case", list(STILL_REFUSED))
def test_what_cannot_be_derived_is_still_refused(case):
    rest, rules, rate, named = STILL_REFUSED[case]
    sbml = _with_rules(rest, rules, rate) if rules else _sbml(rest)
    with pytest.raises(bngsim.ModelError, match=named):
        bngsim.Model.from_sbml_string(sbml)


def _edited(rest, edit):
    import libsbml

    doc = libsbml.readSBMLFromString(_sbml(rest))
    edit(doc.getModel())
    return libsbml.writeSBMLToString(doc)


def _conversion_factor(model):
    cf = model.createParameter()
    cf.setId("cf")
    cf.setValue(3.0)
    cf.setConstant(True)
    model.getSpecies("S3").setConversionFactor("cf")


def _variable_stoichiometry(model):
    sr = model.getReaction("D").getReactant(0)
    sr.setId("nu")
    sr.setConstant(False)


@pytest.mark.parametrize(
    "edit",
    [_conversion_factor, _variable_stoichiometry],
    ids=["conversion-factor", "stoichiometry"],
)
def test_a_species_rate_the_writing_does_not_cover_is_still_refused(edit):
    """A conversion factor scales S3's change, and a stoichiometry that is not
    constant moves with the run: the sum of the laws is then not rateOf(S3),
    so its rate is left unknown, and the size is refused as before."""
    sbml = _edited("cv := 2 + rateOf(S3); D: S3 -> ; 0.1*S3;", edit)
    with pytest.raises(bngsim.ModelError, match=r"(?s)cannot follow to its rate.*rateOf\(S3\)"):
        bngsim.Model.from_sbml_string(sbml)


# An event that moves what rateOf reads jumps rateOf, and the size with it, at
# that instant; the size's jumps are found through its rules, not through a
# rate, so its concentrations would not be rescaled there. Loaded, S3 reset
# to 10 at t = 2 lost a fifth of X's amount at the event, with no word.
_EVENT_MOVES_THE_RATE = {
    "species": "cv := 2 + rateOf(S3); D: S3 -> ; 0.1*S3; at time >= 2: S3 = 10;",
    "rate-rule": "cv := 2 + rateOf(P); P = 1; P' = -0.1*P; at time >= 2: P = 5;",
    "through-a-rule": "cv := 2 + rateOf(S3); D: S3 -> ; k*S3; k := 0.1*q; q = 1;"
    " at time >= 2: q = 3;",
    "its-compartment": "cv := 2 + rateOf(S3); D: S3 -> ; 0.1*S3; at time >= 2: c = 4;",
    # a law that reads neither S3 nor c: rateOf(S3) = 0.1/c jumps with c alone
    "its-compartment-alone": "cv := 2 + rateOf(S3); P0: -> S3; 0.1; at time >= 2: c = 4;",
}


@pytest.mark.parametrize("case", list(_EVENT_MOVES_THE_RATE))
def test_a_rate_an_event_moves_is_still_refused(case):
    with pytest.raises(bngsim.ModelError, match=r"(?s)cannot follow to its rate.*rateOf"):
        bngsim.Model.from_sbml_string(_sbml(_EVENT_MOVES_THE_RATE[case]))


def test_an_event_that_moves_nothing_the_rate_reads():
    """An event elsewhere, on a q no law reads, leaves the rate alone: the
    model loads, and X's amount is A(0) exp(-t/4) across the event."""
    sbml = _sbml("cv := 2 + rateOf(S3); D: S3 -> ; 0.1*S3; q = 1; at time >= 2: q = 3;")
    np.testing.assert_allclose(
        _x(sbml), _diluted(lambda t: 2 - 0.1 * np.exp(-t / 20)), rtol=1e-7, atol=1e-10
    )
