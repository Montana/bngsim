"""One crossing reached two ways is one root (issue #1082).

The SBML loader splices a function-definition formal bound to a call into the
body as ``(time())``, where the same relational written directly reads
``time()``. Conditions were de-duplicated by text, so an event on
``abs(time - 2.5) < 0.005`` beside a rate law gated on ``near(time, 2.5,
0.005)``, with ``near := abs(x - c) < w``, registered each edge twice, and
``Model.time_discontinuity_conditions()`` listed four conditions for two. The
results were right; the roots and the listing were redundant.

Conditions are now de-duplicated on a key with the parentheses around a lone
name or a call dropped, and the first spelling met is the one registered.
"""

from __future__ import annotations

import numpy as np
import pytest

libsbml = pytest.importorskip("libsbml")
import bngsim  # noqa: E402
from bngsim._sbml_loader import _condition_key  # noqa: E402

NEAR = [("near", "lambda(x, c, w, abs(x - c) < w)")]
PEAK = [("peak", "lambda(s, sin(s) >= 1)")]


def _model(tmp_path, law, trigger, functions=(), species_rate=None, rule=None):
    doc = libsbml.SBMLDocument(3, 1)
    m = doc.createModel()
    c = m.createCompartment()
    c.setId("C")
    c.setSize(1)
    c.setConstant(True)
    for sid in ("Y", "Z", "A", "W"):
        s = m.createSpecies()
        s.setId(sid)
        s.setCompartment("C")
        s.setInitialConcentration(0)
        s.setHasOnlySubstanceUnits(False)
        s.setBoundaryCondition(False)
        s.setConstant(False)
    for fid, body in functions:
        f = m.createFunctionDefinition()
        f.setId(fid)
        f.setMath(libsbml.parseL3Formula(body))
    for rid, product, rate in (("R", "Z", law), ("RA", "A", species_rate or "2")):
        r = m.createReaction()
        r.setId(rid)
        r.setReversible(False)
        p = r.createProduct()
        p.setSpecies(product)
        p.setStoichiometry(1)
        p.setConstant(True)
        r.createKineticLaw().setMath(libsbml.parseL3Formula(rate))
    if rule is not None:
        ar = m.createAssignmentRule()
        ar.setVariable("W")
        ar.setMath(libsbml.parseL3Formula(rule))
    e = m.createEvent()
    e.setId("E")
    e.setUseValuesFromTriggerTime(True)
    t = e.createTrigger()
    t.setMath(libsbml.parseL3Formula(trigger))
    t.setInitialValue(False)
    t.setPersistent(True)
    a = e.createEventAssignment()
    a.setVariable("Y")
    a.setMath(libsbml.parseL3Formula("Y + 1"))
    path = tmp_path / "m.xml"
    path.write_text(libsbml.writeSBMLToString(doc))
    return bngsim.Model.from_sbml(str(path))


CASES = {
    # (law, trigger, functions, t_end): the issue's two pairs
    "window": (
        "piecewise(10, near(time, 2.5, 0.005), 0)",
        "abs(time - 2.5) < 0.005",
        NEAR,
        5.0,
    ),
    "touch": ("piecewise(1e-6, peak(time), 0)", "sin(time) >= 1", PEAK, 20.0),
}
DIRECT = {
    "window": "piecewise(10, abs(time - 2.5) < 0.005, 0)",
    "touch": "piecewise(1e-6, sin(time) >= 1, 0)",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_crossing_reached_through_a_function_definition_is_one_condition(tmp_path, case):
    """4 conditions for 2 edges, and 2 for 1 touch: now as many as written
    directly, and the event's trigger still reads every one of them."""
    law, trigger, functions, _ = CASES[case]
    through = _model(tmp_path, law, trigger, functions)
    direct = _model(tmp_path, DIRECT[case], trigger)
    assert len(through.time_discontinuity_conditions()) == len(
        direct.time_discontinuity_conditions()
    )
    assert set(through._event_disc_conditions) == set(through.time_discontinuity_conditions())


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_run_is_what_it_was(tmp_path, case):
    """The same trajectory either way, to the bit; the duplicate roots never
    changed a result."""
    law, trigger, functions, t_end = CASES[case]
    runs = [
        bngsim.Simulator(m, method="ode").run(t_span=(0, t_end), n_points=2)
        for m in (
            _model(tmp_path, law, trigger, functions),
            _model(tmp_path, DIRECT[case], trigger),
        )
    ]
    np.testing.assert_array_equal(np.asarray(runs[0].species), np.asarray(runs[1].species))


@pytest.mark.parametrize(
    "law, rule",
    [
        (DIRECT["window"], CASES["window"][0]),
        (CASES["window"][0], DIRECT["window"]),
    ],
    ids=["law direct, rule through", "law through, rule direct"],
)
def test_the_recovery_reads_the_registered_spelling_as_its_own(tmp_path, law, rule):
    """One spelling registered, the other in a rate law or an assignment rule
    the recovery reads: it compared text with the outer parentheses dropped
    only, found no match, and added the whole window again as a stop. The
    run then took 127 steps for 14 and Z moved by 4e-8. Compared on the
    loader's key, as the loader registers, the recovery finds it covered."""
    through = _model(tmp_path, law, "abs(time - 2.5) < 0.005", NEAR, rule=rule)
    direct = _model(tmp_path, DIRECT["window"], "abs(time - 2.5) < 0.005", rule=DIRECT["window"])
    assert len(through.time_discontinuity_conditions()) == len(
        direct.time_discontinuity_conditions()
    )
    assert through._derived_time_disc_conditions == ()
    runs = [
        bngsim.Simulator(m, method="ode").run(t_span=(0, 5), n_points=2) for m in (through, direct)
    ]
    np.testing.assert_array_equal(np.asarray(runs[0].species), np.asarray(runs[1].species))
    assert runs[0].solver_stats["n_steps"] == runs[1].solver_stats["n_steps"]


@pytest.mark.parametrize("first", ["written directly", "through the function"])
def test_two_rate_laws_on_one_crossing_list_it_once(tmp_path, first):
    """One rate law on ``near(time, 2.5, 0.005)`` and one on ``abs(time - 2.5)
    < 0.005``, in either order. One spelling of each edge is registered, and
    the recovery knew it by that spelling alone: it listed the other rate
    law's again, whole, and the run took 134 steps where both written
    directly take 15."""
    direct, via = DIRECT["window"], CASES["window"][0]
    laws = (direct, via) if first == "written directly" else (via, direct)
    both = _model(tmp_path, laws[0], "time > 100", NEAR, species_rate=laws[1])
    listed = both.time_discontinuity_conditions()
    reference = _model(tmp_path, direct, "time > 100", species_rate=direct)
    assert len(listed) == len(reference.time_discontinuity_conditions())
    runs = [
        bngsim.Simulator(m, method="ode").run(t_span=(0, 5), n_points=11)
        for m in (both, reference)
    ]
    assert runs[0].solver_stats["n_steps"] == runs[1].solver_stats["n_steps"]
    np.testing.assert_array_equal(np.asarray(runs[0].species), np.asarray(runs[1].species))


def test_a_state_window_is_untouched(tmp_path):
    """Control. A plain name spells the same both ways, so ``near(A, 5,
    0.01)`` beside ``abs(A - 5) < 0.01`` registered 2 conditions before, and
    still does."""
    m = _model(
        tmp_path,
        "piecewise(10, near(A, 5, 0.01), 0)",
        "abs(A - 5) < 0.01",
        NEAR,
    )
    run = bngsim.Simulator(m, method="ode").run(t_span=(0, 5), n_points=2)
    y, z = np.asarray(run.species)[-1, :2]
    assert y == 1.0
    assert z == pytest.approx(0.1, rel=1e-6)


@pytest.mark.parametrize(
    "a, b",
    [
        ("((((time())-2.5))<(0.005))", "(((time()-2.5))<(0.005))"),
        ("(sin((time())))>=1", "(sin(time()))>=1"),
        ("((x))<(y)", "(x)<y"),
    ],
)
def test_two_spellings_of_one_condition_have_one_key(a, b):
    assert _condition_key(a) == _condition_key(b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("((a+b))<c", "(a+b*1)<c"),
        ("(x)*(y)<1", "x*(y+1)<1"),
        ("f(x)<2", "g(x)<2"),
    ],
)
def test_different_conditions_keep_different_keys(a, b):
    assert _condition_key(a) != _condition_key(b)


def test_a_group_with_an_operator_keeps_its_parentheses():
    """``(a+b)*c`` and ``a+b*c`` are not one condition."""
    assert _condition_key("((a+b))*c<1") == "((a+b))*c<1"
