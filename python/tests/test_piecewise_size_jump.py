"""Issue #1054 — a piecewise assignment-rule compartment size that jumps.

SBML keeps a species' amount across a change of its compartment's size, so a
concentration jumps by V_before/V_after where the size does. The §8c dilution
term carries the size's rate between jumps, but a jump is a delta no ODE term
carries, and the concentration was held there instead:

- ``cv := piecewise(1 + 0.1*S, S > 3, 1 + 0.2*S)``, S decaying from 10 at 0.5,
  jumps 1.30 -> 1.60 at t = 2.41, and [Sn] stayed high by 1.60/1.30 after;
- ``cv := piecewise(1 + t, t < 2, 5)`` jumps 3 -> 5 at t = 2, and [Sn] stayed
  at 1.667 where the amount, 5, over 5 is 1.

Each case is checked against amount/V(t) in closed form.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'


def _pw(*pieces, otherwise):
    body = "".join(f"<piece>{v}{c}</piece>" for v, c in pieces)
    return f"<piecewise>{body}<otherwise>{otherwise}</otherwise></piecewise>"


def _plus(a, b):
    return f"<apply><plus/>{a}{b}</apply>"


def _times(a, b):
    return f"<apply><times/>{a}{b}</apply>"


def _rel(op, a, b):
    return f"<apply><{op}/>{a}{b}</apply>"


def _cn(x):
    return f"<cn>{x}</cn>"


def _ci(x):
    return f"<ci>{x}</ci>"


def _doc(rules, *, functions="", params="", events=""):
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {M}>{b}</math></assignmentRule>' for v, b in rules
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
 <listOfFunctionDefinitions>{functions}</listOfFunctionDefinitions>
 <listOfCompartments><compartment id="c" size="1" constant="true"/>
  <compartment id="cv" size="1" constant="false"/></listOfCompartments>
 <listOfSpecies>
  <species id="S" compartment="c" initialAmount="10" hasOnlySubstanceUnits="false"
   boundaryCondition="false" constant="false"/>
  <species id="Sn" compartment="cv" initialConcentration="5"
   hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
 </listOfSpecies>
 <listOfParameters><parameter id="k" value="0.5" constant="true"/>{params}</listOfParameters>
 <listOfRules>{rule_xml}</listOfRules>
 <listOfReactions><reaction id="degS" reversible="false">
  <listOfReactants>
   <speciesReference species="S" stoichiometry="1" constant="true"/>
  </listOfReactants>
  <kineticLaw><math {M}>{_times(_times(_ci("k"), _ci("S")), _ci("c"))}</math></kineticLaw>
 </reaction></listOfReactions>
 <listOfEvents>{events}</listOfEvents>
</model></sbml>"""


S_SWITCH = _pw(
    (_plus(_cn(1), _times(_cn(0.1), _ci("S"))), _rel("gt", _ci("S"), _cn(3))),
    otherwise=_plus(_cn(1), _times(_cn(0.2), _ci("S"))),
)
T_STEP = _pw((_plus(_cn(1), T), _rel("lt", T, _cn(2))), otherwise=_cn(5))
# up at t = 1 and back down at t = 3
T_WINDOW = _pw(
    (_cn(2), f"<apply><and/>{_rel('gt', T, _cn(1))}{_rel('lt', T, _cn(3))}</apply>"),
    otherwise=_cn(1),
)
# continuous at its switch: 1 + t meets 3 at t = 2
T_CONTINUOUS = _pw((_plus(_cn(1), T), _rel("lt", T, _cn(2))), otherwise=_cn(3))
STEP_FN = (
    '<functionDefinition id="step"><math '
    + M
    + "><lambda><bvar><ci>x</ci></bvar>"
    + _pw((_plus(_cn(1), _ci("x")), _rel("lt", _ci("x"), _cn(2))), otherwise=_cn(5))
    + "</lambda></math></functionDefinition>"
)


def _s(t):
    return 10 * np.exp(-0.5 * t)


CASES = {
    # the size jumps 1.30 -> 1.60 as S falls through 3; V(0) = 2, amount 10
    "on a species": (
        _doc([("cv", S_SWITCH)]),
        lambda t: 10 / np.where(_s(t) > 3, 1 + 0.1 * _s(t), 1 + 0.2 * _s(t)),
    ),
    "on time": (_doc([("cv", T_STEP)]), lambda t: 5 / np.where(t < 2, 1 + t, 5)),
    "both ways": (
        _doc([("cv", T_WINDOW)]),
        lambda t: 5 / np.where((t > 1) & (t < 3), 2, 1),
    ),
    "through another rule": (
        _doc(
            [("cv", _plus(_cn(0), _ci("q"))), ("q", T_STEP)],
            params='<parameter id="q" value="1" constant="false"/>',
        ),
        lambda t: 5 / np.where(t < 2, 1 + t, 5),
    ),
    "through a function definition": (
        _doc([("cv", f"<apply><ci>step</ci>{T}</apply>")], functions=STEP_FN),
        lambda t: 5 / np.where(t < 2, 1 + t, 5),
    ),
    # control: no jump, so nothing to rescale
    "continuous (control)": (
        _doc([("cv", T_CONTINUOUS)]),
        lambda t: 5 / np.where(t < 2, 1 + t, 3),
    ),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_amount_is_kept_across_the_jump(case, codegen):
    xml, closed_form = CASES[case]
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", codegen=codegen
    ).run(t_span=(0.0, 4.0), n_points=17)
    t = np.asarray(result.time)
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    np.testing.assert_allclose(sn, closed_form(t), rtol=1e-5, atol=1e-8)


def test_a_condition_an_event_also_moves_is_refused():
    """``t > p`` flips with the clock and when the event sets p: the event's
    own rescale (#74) and the jump's would both apply, so it is refused."""
    size = _pw((_cn(2), _rel("gt", T, _ci("p"))), otherwise=_cn(1))
    event = (
        '<event id="e" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true"><math {M}>'
        f"{_rel('gt', T, _cn(0.5))}</math></trigger>"
        f'<listOfEventAssignments><eventAssignment variable="p"><math {M}>{_cn(0.25)}'
        "</math></eventAssignment></listOfEventAssignments></event>"
    )
    xml = _doc(
        [("cv", size)],
        params='<parameter id="p" value="3" constant="false"/>',
        events=event,
    )
    with pytest.raises(bngsim.ModelError, match=r"event target"):
        bngsim.Model.from_sbml_string(xml)
