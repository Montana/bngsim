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

The condition now reads its rules by name, and what the sides share is read
through functions of their own.

Oracles: closed forms, the rules evaluated in numpy. S = 10·e^(-t/2), and Sn
keeps its amount, so [Sn] = 5·V(0)/V(t).
"""

from __future__ import annotations

import bngsim
import numpy as np

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
    if shape == "condition":
        rules = [("r0", _ci("S")), ("r1", _ap("plus", _ci("S"), _cn(1))), *nonlinear]
        rules.append(("cv", _pw(_cn(1), _ap("gt", _ci(f"r{n - 1}"), _cn(0.5)), _cn(2))))
    else:
        bottom = _pw(_ci("S"), _ap("gt", _ci("S"), _cn(3)), _ap("times", _cn(2), _ci("S")))
        rules = [("r0", bottom), ("r1", _ap("plus", _ci("r0"), _cn(1))), *nonlinear]
        rules.append(("cv", _ap("plus", _cn(1), _ap("times", _cn(1e-6), _ci(f"r{n - 1}")))))
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {_M}>{b}</math></assignmentRule>' for v, b in rules
    )
    params = "".join(f'<parameter id="r{i}" constant="false"/>' for i in range(n))
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
    r0 = np.where(s > 3, s, 2 * s)
    return 1 + 1e-6 * _diamond(r0, r0 + 1, n)


def _run(n, shape, times):
    model = bngsim.Model.from_sbml_string(_doc(n, shape))
    result = bngsim.Simulator(model, method="ode", jacobian="fd").run(
        sample_times=list(times), rtol=1e-10, atol=1e-12
    )
    return model, np.asarray(result.species)[:, list(result.species_names).index("Sn")]


def test_a_condition_over_a_diamond_reads_its_rules_by_name():
    """Main took 0.7 s to load this at 20 rules, ×2.6 for every two more; the
    jump reads ``r19`` by name."""
    n = 20
    times = np.linspace(0.0, 4.0, 41)
    v = _size(times, n, "condition")
    # Rows within a step of a crossing would test where the root lands.
    away = np.ones_like(times, dtype=bool)
    away[1:] &= v[1:] == v[:-1]
    away[:-1] &= v[:-1] == v[1:]
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
