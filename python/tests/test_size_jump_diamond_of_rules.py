"""A size's jump read through a diamond of rules loads in linear time
(issue #1124, item 6).

#1054 writes a piecewise size's jump from the size through its rules: the
condition both ways, and the size on either side. #1116 kept the common
shapes short. Two were still written out once per path through a diamond of
rules ``r_i := sin(r_{i-1}) + cos(r_{i-2})``, exponential in its depth:

* **A condition over a diamond.** ``cv := piecewise(1, r_{n-1} > 0.5, 2)``:
  1.76 s to load at 22 rules, the condition printed in full, and sympy's
  Piecewise read it through once more for each rule put in.
* **A piecewise at the bottom of one.** ``r0 := piecewise(S, S > 3, 2*S)``
  under the diamond, ``cv := 1 + 1e-6*r_{n-1}``: no rule above r0 holds the
  same value on both sides, so none could be read by name.
* **Two relations over one,** ``piecewise(1, r21 > 1.4, 3, r21 < 1.1, 2)`` or
  ``piecewise(1, r21 > 1.2 && S > 3, 2)``: their differences were cancelled
  against each other, to see whether they share a root, and the And was put
  together by its constructor, each a walk of the tree.

The condition now reads its rules by name, and what the sides share is read
through functions of their own.

Oracles: closed forms, the rules evaluated in numpy. S = 10·e^(-t/2), and Sn
keeps its amount, so [Sn] = 5·V(0)/V(t).
"""

from __future__ import annotations

import time

import bngsim
import numpy as np
import pytest

_M = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _pw(a, c, b):
    return f"<piecewise><piece>{a}{c}</piece><otherwise>{b}</otherwise></piecewise>"


def _ap(op, *args):
    return f"<apply><{op}/>{''.join(args)}</apply>"


def _ci(x):
    return f"<ci>{x}</ci>"


def _cn(x):
    return f"<cn>{x}</cn>"


def _doc(n, shape):
    nonlinear = [
        (f"r{i}", _ap("plus", _ap("sin", _ci(f"r{i - 1}")), _ap("cos", _ci(f"r{i - 2}"))))
        for i in range(2, n)
    ]
    last = _ci(f"r{n - 1}")
    if shape in ("condition", "two", "pair", "and"):
        rules = [("r0", _ci("S")), ("r1", _ap("plus", _ci("S"), _cn(1))), *nonlinear]
        if shape == "condition":
            cv = _pw(_cn(1), _ap("gt", last, _cn(0.5)), _cn(2))
        elif shape in ("two", "pair"):
            # `pair`: the second relation shares the first one's root
            below = _ap("lt", last, _cn(1.1))
            if shape == "pair":
                below = _ap("lt", _ap("times", _cn(2), last), _cn(2.8))
            two = f"<piece>{_cn(3)}{below}</piece>"
            cv = _pw(_cn(1), _ap("gt", last, _cn(1.4)), _cn(2)).replace(
                "<otherwise>", two + "<otherwise>"
            )
        else:
            cv = _pw(
                _cn(1), _ap("and", _ap("gt", last, _cn(1.2)), _ap("gt", _ci("S"), _cn(3))), _cn(2)
            )
        rules.append(("cv", cv))
    else:
        bottom = _pw(_ci("S"), _ap("gt", _ci("S"), _cn(3)), _ap("times", _cn(2), _ci("S")))
        rules = [("r0", bottom), ("r1", _ap("plus", _ci("r0"), _cn(1))), *nonlinear]
        rules.append(("cv", _ap("plus", _cn(1), _ap("times", _cn(1e-6), _ci(f"r{n - 1}")))))
    return _sbml(rules)


def _sbml(rules):
    """S decays at k = 0.5 in c; Sn sits in cv, which the last rule sizes."""
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {_M}>{b}</math></assignmentRule>' for v, b in rules
    )
    params = "".join(f'<parameter id="{v}" constant="false"/>' for v, _ in rules if v != "cv")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
<listOfCompartments><compartment id="c" size="1" constant="true"/>
<compartment id="cv" size="1" constant="false"/></listOfCompartments>
<listOfSpecies>
<species id="S" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
<species id="Sn" compartment="cv" initialConcentration="5" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters><parameter id="k" value="0.5" constant="true"/>{params}</listOfParameters>
<listOfRules>{rule_xml}</listOfRules>
<listOfReactions><reaction id="degS" reversible="false"><listOfReactants>
<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
<kineticLaw><math {_M}><apply><times/><ci>k</ci><ci>S</ci><ci>c</ci></apply></math>
</kineticLaw></reaction></listOfReactions>
</model></sbml>"""


def _diamond(r0, r1, n):
    a, b = r0, r1
    for _ in range(2, n):
        a, b = b, np.sin(b) + np.cos(a)
    return b


def _size(t, n, shape):
    s = 10 * np.exp(-0.5 * t)
    if shape == "condition":
        return np.where(_diamond(s, s + 1, n) > 0.5, 1.0, 2.0)
    if shape == "two":
        r = _diamond(s, s + 1, n)
        return np.where(r > 1.4, 1.0, np.where(r < 1.1, 3.0, 2.0))
    if shape == "pair":
        r = _diamond(s, s + 1, n)
        return np.where(r > 1.4, 1.0, np.where(2 * r < 2.8, 3.0, 2.0))
    if shape == "and":
        return np.where((_diamond(s, s + 1, n) > 1.2) & (s > 3), 1.0, 2.0)
    r0 = np.where(s > 3, s, 2 * s)
    return 1 + 1e-6 * _diamond(r0, r0 + 1, n)


def _away(v):
    """Rows not within a step of a crossing, which would test where the root
    lands."""
    away = np.ones_like(v, dtype=bool)
    away[1:] &= v[1:] == v[:-1]
    away[:-1] &= v[:-1] == v[1:]
    return away


def _run(n, shape, times):
    model = bngsim.Model.from_sbml_string(_doc(n, shape))
    return model, _sn(model, times)


def _sn(model, times, seed=None):
    result = bngsim.Simulator(model, method="ode", jacobian="fd").run(
        sample_times=list(times), rtol=1e-10, atol=1e-12, seed=seed
    )
    return np.asarray(result.species)[:, list(result.species_names).index("Sn")]


def test_a_condition_over_a_diamond_reads_its_rules_by_name():
    """Main took 0.7 s to load this at 20 rules, ×2.6 for every two more; the
    jump reads ``r19`` by name."""
    n = 20
    times = np.linspace(0.0, 4.0, 41)
    v = _size(times, n, "condition")
    away = _away(v)
    model, sn = _run(n, "condition", times)
    assert model._core.event_trigger_sources() == ["(r19 > 0.5)", "(r19 <= 0.5)"]
    assert not [f for f in model.function_names if f.startswith("_szj_")]
    np.testing.assert_allclose(sn[away], (5 * v[0] / v)[away], rtol=1e-8)


def test_a_piecewise_under_a_diamond_reads_the_sides_through_functions():
    """S falls through 3 at t = 2·ln(10/3): r0 halves, and the size jumps."""
    n = 22
    times = np.linspace(0.0, 4.0, 41)
    model, sn = _run(n, "bottom", times)
    assert [f for f in model.function_names if f.startswith("_szj_")]
    v = _size(times, n, "bottom")
    np.testing.assert_allclose(sn, 5 * v[0] / v, rtol=1e-8)


def test_a_short_diamond_is_written_out_as_before():
    model, _ = _run(6, "bottom", [0.0, 4.0])
    assert not [f for f in model.function_names if f.startswith("_szj_")]


def test_a_condition_that_reads_a_piecewise_rule_is_read_as_its_relations():
    """``r := piecewise(S, S > 3, 2*S)`` under ``cv := piecewise(A, r > 5, B)``,
    with A and B piecewise on S > 3 too. The pair's own constructor reads
    ``r > 5`` as S > 5 and 2*S > 5, each continuous, and S > 3's root alone
    fires at S = 3; main does. Put together raw, as #1145 first had it, ``r >
    5`` jumped at S = 3 itself, and its root fired there beside S > 3's: two
    rescales at the instant, the one drawn last kept, and [Sn] at t = 2.5 was
    25/6 for 10/3 on half the seeds."""
    r = _pw(_ci("S"), _ap("gt", _ci("S"), _cn(3)), _ap("times", _cn(2), _ci("S")))
    a = _pw(_cn(1), _ap("gt", _ci("S"), _cn(3)), _cn(1.5))
    b = _pw(_cn(2), _ap("gt", _ci("S"), _cn(3)), _cn(2.5))
    rules = [("r", r), ("cv", _pw(a, _ap("gt", _ci("r"), _cn(5)), b))]
    times = np.array([0.0, 1.0, 2.0, 2.5, 3.0, 4.0])
    s = 10 * np.exp(-0.5 * times)
    v = np.where(
        np.where(s > 3, s, 2 * s) > 5, np.where(s > 3, 1.0, 1.5), np.where(s > 3, 2.0, 2.5)
    )
    for seed in range(8):
        model = bngsim.Model.from_sbml_string(_sbml(rules))
        assert not [t for t in model._core.event_trigger_sources() if "if(" in t]
        sn = _sn(model, times, seed=seed)
        np.testing.assert_allclose(sn, 5 * v[0] / v, rtol=1e-8, err_msg=f"seed {seed}")


@pytest.mark.parametrize(("shape", "roots"), [("two", 2), ("pair", 1), ("and", 2)])
def test_two_relations_over_a_diamond_load_in_linear_time(shape, roots):
    """``piecewise(1, r21 > 1.4, 3, r21 < 1.1, 2)``, the same with ``2*r21 <
    2.8``, which shares r21 > 1.4's root, and ``piecewise(1, r21 > 1.2 && S > 3,
    2)``. Whether two relations share a root was asked of ``cancel``, and the
    And was put together by its constructor, which sorts its arguments by their
    size as trees: 1.1 and 0.55 s to load at 18 rules, x2.6 for every two more.
    Two long differences are now compared term by term, then read at two
    points, and the And is put together as it is."""
    n = 22
    times = np.linspace(0.0, 4.0, 81)
    v = _size(times, n, shape)
    assert len(set(v)) > 1
    began = time.perf_counter()
    model = bngsim.Model.from_sbml_string(_doc(n, shape))
    assert time.perf_counter() - began < 2.0
    triggers = model._core.event_trigger_sources()
    assert f"(r{n - 1} > {1.2 if shape == 'and' else 1.4})" in triggers
    assert len(triggers) == 2 * roots
    away = _away(v)
    np.testing.assert_allclose(_sn(model, times)[away], (5 * v[0] / v)[away], rtol=1e-8)


def test_a_piecewise_whose_pieces_read_alike_has_no_root():
    """``r := piecewise(r0, S > 3, r1)`` with r0 and r1 both S is S, as the
    Piecewise constructor merges pieces in a row that read alike. Put together
    raw, its S > 3 was a root, with two events that rescaled by 1."""
    rules = [
        ("r0", _ci("S")),
        ("r1", _ci("S")),
        ("r", _pw(_ci("r0"), _ap("gt", _ci("S"), _cn(3)), _ci("r1"))),
        ("cv", _ap("plus", _cn(1), _ci("r"))),
    ]
    model = bngsim.Model.from_sbml_string(_sbml(rules))
    assert model._core.event_trigger_sources() == []
    times = np.linspace(0.0, 4.0, 9)
    v = 1 + 10 * np.exp(-0.5 * times)
    np.testing.assert_allclose(_sn(model, times), 5 * v[0] / v, rtol=1e-8)


def test_a_relation_that_cannot_be_read_at_a_point_is_left_to_cancel():
    """``max(sqrt(S - 0.9), r19) > 0.5`` beside ``S < 1``: the two are read at
    a point before ``cancel`` is asked whether they share a root, and at S =
    0.87 the square root is complex, which ``Max`` refuses by raising. The
    model failed to load; main loads it."""
    n = 20
    last = _ci(f"r{n - 1}")
    root = _ap("root", _cn(2), _ap("minus", _ci("S"), _cn(0.9)))
    below = f"<piece>{_cn(3)}{_ap('lt', _ci('S'), _cn(1))}</piece>"
    cv = _pw(_cn(1), _ap("gt", _ap("max", root, last), _cn(0.5)), _cn(2))
    nonlinear = [
        (f"r{i}", _ap("plus", _ap("sin", _ci(f"r{i - 1}")), _ap("cos", _ci(f"r{i - 2}"))))
        for i in range(2, n)
    ]
    rules = [("r0", _ci("S")), ("r1", _ap("plus", _ci("S"), _cn(1))), *nonlinear]
    rules.append(("cv", cv.replace("<otherwise>", below + "<otherwise>")))
    model = bngsim.Model.from_sbml_string(_sbml(rules))
    assert len(model._core.event_trigger_sources()) == 4
