"""Issue #1055 — an event resizes an assignment-rule compartment it reaches
only through another rule, or through another compartment's species.

SBML keeps a species' amount across a change of its compartment's size, so an
event that changes the size rescales the concentration by V_old/V_new (#74).
That was done only when the size rule named the event's variable itself:

- ``cv := 1 + q``, ``q := 0.5*p``, event ``p := 4``: cv goes 1.5 -> 3 and
  [Sn] 5 -> 2.5, but Sn stayed at 5;
- ``cv := 1 + 0.1*Kw``, Kw in ``ce``, event ``ce := 2``: [Kw] goes 10 -> 5,
  cv 2 -> 1.5, and [Sn] 5 -> 6.667, but Sn stayed at 5.

The second also needs the engine to read cv's new size after Kw's rescale,
which it read before. RoadRunner 2.9.2 gives the values asserted here.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'


TRIGGER = f"<math {M}><apply><gt/>{T}<cn>1</cn></apply></math>"


def _doc(comps, species, params, rules, assignments, uvftt):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
 <listOfCompartments>{comps}</listOfCompartments>
 <listOfSpecies>{species}</listOfSpecies>
 <listOfParameters>{params}</listOfParameters>
 <listOfRules>{rules}</listOfRules>
 <listOfEvents><event id="e" useValuesFromTriggerTime="{uvftt}">
  <trigger initialValue="false" persistent="true">{TRIGGER}</trigger>
  <listOfEventAssignments>{assignments}</listOfEventAssignments>
 </event></listOfEvents>
</model></sbml>"""


def _comp(cid, size=1):
    return f'<compartment id="{cid}" size="{size}" constant="false"/>'


def _species(sid, comp, conc, constant=False):
    return (
        f'<species id="{sid}" compartment="{comp}" initialConcentration="{conc}" '
        f'hasOnlySubstanceUnits="false" boundaryCondition="false" '
        f'constant="{str(constant).lower()}"/>'
    )


def _rule(var, body):
    return f'<assignmentRule variable="{var}"><math {M}>{body}</math></assignmentRule>'


def _plus_tenth(name):
    return f"<apply><plus/><cn>1</cn><apply><times/><cn>0.1</cn><ci>{name}</ci></apply></apply>"


def _set(var, value):
    return f'<eventAssignment variable="{var}"><math {M}><cn>{value}</cn></math></eventAssignment>'


def _chain(uvftt):
    return _doc(
        _comp("cv"),
        _species("Sn", "cv", 5),
        '<parameter id="p" value="1" constant="false"/>'
        '<parameter id="q" value="0" constant="false"/>',
        _rule("cv", "<apply><plus/><cn>1</cn><ci>q</ci></apply>")
        + _rule("q", "<apply><times/><cn>0.5</cn><ci>p</ci></apply>"),
        _set("p", 4),
        uvftt,
    )


def _via_species(uvftt):
    return _doc(
        _comp("cv") + _comp("ce"),
        _species("Sn", "cv", 5) + _species("Kw", "ce", 10, constant=True),
        "",
        _rule("cv", _plus_tenth("Kw")),
        _set("ce", 2),
        uvftt,
    )


def _cascade(uvftt):
    """ce -> [Kw] -> cv -> [Sn] -> cw -> [Z], with cw's rule written first."""
    return _doc(
        _comp("cv") + _comp("ce") + _comp("cw"),
        _species("Sn", "cv", 5) + _species("Kw", "ce", 10, constant=True) + _species("Z", "cw", 3),
        "",
        _rule("cw", _plus_tenth("Sn")) + _rule("cv", _plus_tenth("Kw")),
        _set("ce", 2),
        uvftt,
    )


def _direct(uvftt):
    return _doc(
        _comp("cv"),
        _species("Sn", "cv", 5),
        '<parameter id="p" value="1" constant="false"/>',
        _rule("cv", "<ci>p</ci>"),
        _set("p", 2),
        uvftt,
    )


CASES = {
    # amount 7.5 in cv = 1.5, then cv = 3
    "through another rule": (_chain, {"Sn": 2.5}),
    # [Kw] 10 -> 5, cv 2 -> 1.5, amount 10 in cv
    "through a rescaled species": (_via_species, {"Sn": 10 / 1.5, "Kw": 5.0}),
    # and then cw 1.5 -> 1 + 0.1 * 20/3, amount 4.5 in cw
    "a cascade of three": (_cascade, {"Sn": 10 / 1.5, "Kw": 5.0, "Z": 4.5 / (1 + 2 / 3)}),
    # control: the size rule names the event's variable
    "direct (control)": (_direct, {"Sn": 2.5}),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("uvftt", ["true", "false"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_amount_is_kept_across_the_resize(case, uvftt, codegen):
    build, want = CASES[case]
    model = bngsim.Model.from_sbml_string(build(uvftt))
    result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
        t_span=(0.0, 4.0), n_points=9
    )
    names = list(result.species_names)
    species = np.asarray(result.species)
    for sid, value in want.items():
        column = species[:, names.index(sid)]
        # Before the event at t = 1 nothing moves; after it, the new value holds.
        np.testing.assert_allclose(column[-1], value, rtol=1e-6, err_msg=sid)
        np.testing.assert_allclose(column[3:], value, rtol=1e-6, err_msg=sid)
