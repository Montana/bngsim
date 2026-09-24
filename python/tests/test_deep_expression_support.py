"""A deeply nested event expression keeps its variables (issue #779).

``referenced_variable_addresses()`` tells the sensitivity machinery which model
variables an expression reads. It re-parsed the expression with ExprTk's
``collect_variables()``, which builds a *default* parser: stack depth 400, where
the evaluator that compiled the expression allows 4096. An expression between
the two limits compiled and evaluated fine, the re-parse failed, and the failure
came back as an empty list, which every caller reads as "depends on nothing":

* a state-dependent trigger was classed as fixed-time and the crossing term
  dt*/dp was dropped (case a: ~197+ nested ``piecewise`` levels);
* an event assignment's jump got zero sensitivity (case b: deep parenthesized
  arithmetic).

Trajectories were unaffected, so nothing looked wrong.

Both models are S' = -k S with k = 1, built so that S(1) = 20 e^-k:

* (a) S(0) = 20; trigger ``S < <deep expression equal to 10>``, assignment
  S := 10. The crossing t* = ln 2 / k moves with k.
* (b) S(0) = 10; trigger ``time > 0.5``, assignment ``S := S * <deep expression
  equal to 2>``, so the jump carries s+ = 2 s-.

So dS(1)/dk = -20 e^-1 in both, which a central difference confirms.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pytest
from bngsim import Model, Simulator

libsbml = pytest.importorskip("libsbml")

EXPECTED = -20.0 * math.exp(-1.0)


def _deep_piecewise(depth: int) -> str:
    """Nested piecewise equal to 10 (p = 0, so no branch condition holds)."""
    s = "10"
    for i in range(depth):
        s = f"piecewise({99 - i % 50}, p > {1000 + i}, {s})"
    return s


def _deep_parens(depth: int) -> str:
    """Right-nested (1 + p*(1 + p*(...))) plus 1: equal to 2 while p = 0."""
    s = "1"
    for _ in range(depth):
        s = f"(1 + p*{s})"
    return f"(1 + {s})"


def _sbml(tmp_path: Path, case: str, depth: int) -> str:
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId(f"deep_{case}_{depth}")
    c = m.createCompartment()
    c.setId("c")
    c.setSize(1)
    c.setConstant(True)
    s = m.createSpecies()
    s.setId("S")
    s.setCompartment("c")
    s.setHasOnlySubstanceUnits(True)
    s.setBoundaryCondition(False)
    s.setConstant(False)
    s.setInitialAmount(20 if case == "a" else 10)
    for pid, value in (("k", 1.0), ("p", 0.0)):
        p = m.createParameter()
        p.setId(pid)
        p.setValue(value)
        p.setConstant(True)
    r = m.createReaction()
    r.setId("decay")
    r.setReversible(False)
    sr = r.createReactant()
    sr.setSpecies("S")
    sr.setStoichiometry(1)
    sr.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula("k*S"))
    e = m.createEvent()
    e.setId("e1")
    e.setUseValuesFromTriggerTime(True)
    t = e.createTrigger()
    t.setInitialValue(False)
    t.setPersistent(True)
    ea = e.createEventAssignment()
    ea.setVariable("S")
    if case == "a":
        t.setMath(libsbml.parseL3Formula(f"S < {_deep_piecewise(depth)}"))
        ea.setMath(libsbml.parseL3Formula("10"))
    else:
        t.setMath(libsbml.parseL3Formula("time > 0.5"))
        ea.setMath(libsbml.parseL3Formula(f"S*{_deep_parens(depth)}"))
    assert t.getMath() is not None and ea.getMath() is not None
    path = tmp_path / f"deep_{case}_{depth}.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    return str(path)


def _s_at_1(path: str, k: float | None = None) -> float:
    model = Model.from_sbml(path)
    if k is not None:
        model.set_param("k", k)
    with contextlib.redirect_stderr(io.StringIO()):
        res = Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    return float(np.asarray(res.species)[-1, 0])


def _dS_dk(path: str) -> float:
    with contextlib.redirect_stderr(io.StringIO()):
        res = Simulator(Model.from_sbml(path), method="ode", sensitivity_params=["k"]).run(
            t_span=(0, 1), n_points=2
        )
    return float(np.asarray(res.sensitivities).reshape(2, -1)[-1, 0])


#: Depths below and past the old collector's limit. Case a failed from 197
#: levels, case b (in the form the SBML loader writes it) from about 80.
CASES = [("a", 10), ("a", 197), ("a", 250), ("b", 10), ("b", 80), ("b", 100)]


@pytest.mark.parametrize("case, depth", CASES, ids=[f"{c}-{d}" for c, d in CASES])
def test_sensitivity_matches_closed_form(tmp_path, case, depth):
    """Before the fix: -2.257698 for deep (a) (dt*/dp dropped) and -3.678795
    for deep (b) (jump sensitivity zeroed), with no warning."""
    assert _dS_dk(_sbml(tmp_path, case, depth)) == pytest.approx(EXPECTED, abs=1e-4)


@pytest.mark.parametrize("case, depth", [("a", 250), ("b", 100)], ids=["a-250", "b-100"])
def test_sensitivity_matches_finite_difference(tmp_path, case, depth):
    """The closed form, checked independently of the code under test."""
    path = _sbml(tmp_path, case, depth)
    h = 1e-5
    fd = (_s_at_1(path, 1 + h) - _s_at_1(path, 1 - h)) / (2 * h)
    assert fd == pytest.approx(EXPECTED, abs=1e-4)
    assert _dS_dk(path) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("case, depth", [("a", 250), ("b", 100)], ids=["a-250", "b-100"])
def test_trajectory_was_never_the_problem(tmp_path, case, depth):
    """S(1) = 20 e^-1 at every depth; only the dependency set was lost."""
    assert _s_at_1(_sbml(tmp_path, case, depth)) == pytest.approx(20 * math.exp(-1), rel=1e-5)
