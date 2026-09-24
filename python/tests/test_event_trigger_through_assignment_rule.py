"""An event trigger that reads state through an assignment rule is state-dependent
(issue #775).

``event_trigger_is_state_dependent`` looked only at the trigger's own variable
addresses and accepted a species, an observable or a rateOf slot. An SBML
assignment rule compiles to a parameter slot, so ``x < 1`` with ``x := 2*S``
named no state and was classed as time-only. The crossing was then applied at a
fixed time with dt*/dp = 0, so dS/dk and dS/dS0 came back wrong (the second an
exact 0), and ``and``/``or`` triggers that are refused when they read ``S``
directly got an answer instead. The classification now goes through
``expression_support()``, which follows assignment rules to the species behind
them.

Model: S(0) = S0 decays at k*S; when the trigger fires S is reset to 1. The
trigger crosses at t* = ln(2*S0)/k, so S(1) = 2*S0*exp(-k), and
dS(1)/dk = -2*exp(-1), dS(1)/dS0 = +2*exp(-1) at k = S0 = 1.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

_MML = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
)

_S_DECL = (
    '<species id="S" compartment="C" initialAmount="1" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>'
)
_X_SPECIES = (
    '<species id="x" compartment="C" initialAmount="0" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>'
)
_X_PARAM = '<parameter id="x" value="0" constant="false"/>'


def _rule(body: str) -> str:
    return (
        f'<listOfRules><assignmentRule variable="x"><math {_MML}>{body}</math>'
        "</assignmentRule></listOfRules>"
    )


_X_IS_2S = _rule("<apply><times/><cn>2</cn><ci>S</ci></apply>")
_X_IS_2S2 = _rule("<apply><times/><cn>2</cn><ci>S</ci><ci>S</ci></apply>")


def _lt(var: str, value: float) -> str:
    return f"<apply><lt/><ci>{var}</ci><cn>{value}</cn></apply>"


def _sbml(trigger: str, rule: str = "", x_species: bool = False) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments>
   <compartment id="C" size="1" constant="true" spatialDimensions="3"/>
  </listOfCompartments>
  <listOfSpecies>{_S_DECL}{_X_SPECIES if x_species else ""}</listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="1" constant="true"/>
   <parameter id="T" value="0.2" constant="true"/>
   {"" if x_species else _X_PARAM}
  </listOfParameters>
  {rule}
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfReactants>
     <speciesReference species="S" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <kineticLaw><math {_MML}><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
   </reaction>
  </listOfReactions>
  <listOfEvents>
   <event id="e1" useValuesFromTriggerTime="true">
    <trigger initialValue="false" persistent="true"><math {_MML}>{trigger}</math></trigger>
    <listOfEventAssignments>
     <eventAssignment variable="S"><math {_MML}><cn>1</cn></math></eventAssignment>
    </listOfEventAssignments>
   </event>
  </listOfEvents>
 </model>
</sbml>"""


_CASES = {
    "control S<0.5": _sbml(_lt("S", 0.5)),
    "param x:=2S, x<1": _sbml(_lt("x", 1), _X_IS_2S),
    "param x:=2S^2, x<0.5": _sbml(_lt("x", 0.5), _X_IS_2S2),
    "species x:=2S^2, x<0.5": _sbml(_lt("x", 0.5), _X_IS_2S2, x_species=True),
}


def _run(xml: str):
    m = bngsim.Model.from_sbml_string(xml)
    r = bngsim.Simulator(m, method="ode", sensitivity_params=["k"], sensitivity_ic=["S"]).run(
        t_span=(0.0, 1.0), n_points=2
    )
    i = list(r.species_names).index("S")
    return (
        m,
        float(np.asarray(r.species)[-1, i]),
        float(np.asarray(r.sensitivities)[-1, i, 0]),
        float(np.asarray(r.sensitivities_ic)[-1, i, 0]),
    )


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_crossing_term_is_carried_through_an_assignment_rule(case):
    """Before the fix every non-control case gave dS/dk = -0.2257697 (the
    fixed-t* value) and dS/dS0 = 0, and no event was on the runtime path."""
    m, s1, dk, ds0 = _run(_CASES[case])
    expected = 2.0 * math.exp(-1.0)
    assert s1 == pytest.approx(expected, rel=1e-6)
    assert dk == pytest.approx(-expected, rel=1e-5)
    assert ds0 == pytest.approx(expected, rel=1e-5)
    assert m._core.events_with_runtime_event_time_sens() == [0]


_AND = f"<apply><and/><apply><geq/>{_TIME}<ci>T</ci></apply>{{atom}}</apply>"
_OR = f"<apply><or/>{{atom}}<apply><geq/>{_TIME}<cn>5</cn></apply></apply>"


@pytest.mark.parametrize("shape", [_AND, _OR], ids=["and", "or"])
@pytest.mark.parametrize("through_rule", [False, True], ids=["direct-S", "through-x"])
def test_a_compound_trigger_is_refused_either_way(shape, through_rule):
    """The direct spelling was always refused; through x it silently answered."""
    atom = _lt("x", 1) if through_rule else _lt("S", 0.5)
    xml = _sbml(shape.format(atom=atom), _X_IS_2S if through_rule else "")
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        _run(xml)


# ─── A state-free trigger over a derived parameter ──────────────────────────

_DERIVED = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="d">
  <listOfCompartments>
   <compartment id="C" size="1" constant="true" spatialDimensions="3"/>
  </listOfCompartments>
  <listOfSpecies>{s}</listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="0.5" constant="true"/>
   <parameter id="d" constant="true"/>
  </listOfParameters>
  <listOfInitialAssignments>
   <initialAssignment symbol="d">
    <math {mml}><apply><times/><cn>2</cn><ci>k</ci></apply></math>
   </initialAssignment>
  </listOfInitialAssignments>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfReactants>
     <speciesReference species="S" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <kineticLaw><math {mml}><apply><times/><cn>0.1</cn><ci>S</ci></apply></math></kineticLaw>
   </reaction>
  </listOfReactions>
  <listOfEvents>
   <event id="e" useValuesFromTriggerTime="true">
    <trigger initialValue="false" persistent="true"><math {mml}>{trigger}</math></trigger>
    <listOfEventAssignments>
     <eventAssignment variable="S"><math {mml}><cn>5</cn></math></eventAssignment>
    </listOfEventAssignments>
   </event>
  </listOfEvents>
 </model>
</sbml>"""


def _derived(trigger: str) -> bngsim.Model:
    return bngsim.Model.from_sbml_string(_DERIVED.format(s=_S_DECL, mml=_MML, trigger=trigger))


def test_a_derived_threshold_the_detector_resolves_still_runs():
    """t >= d with d = 2k: t* = 1, S(3) = 5*exp(-0.1*(3 - 2k)), so
    dS(3)/dk = exp(-0.2). The resolved clock threshold is unaffected."""
    trig = f"<apply><geq/>{_TIME}<ci>d</ci></apply>"
    r = bngsim.Simulator(_derived(trig), method="ode", sensitivity_params=["k"]).run(
        t_span=(0.0, 3.0), n_points=2
    )
    assert float(np.asarray(r.sensitivities)[-1, 0, 0]) == pytest.approx(math.exp(-0.2), rel=1e-6)


def test_an_unresolvable_trigger_over_a_derived_parameter_is_refused():
    """sin(t) > d - 0.5 reaches k only through d. The refusal compared the
    trigger's own references with the requested parameters, so a primary behind
    a derived one slipped through; it now compares everything the trigger reaches."""
    trig = (
        f"<apply><gt/><apply><sin/>{_TIME}</apply>"
        "<apply><minus/><ci>d</ci><cn>0.5</cn></apply></apply>"
    )
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="'k'"):
        bngsim.Simulator(_derived(trig), method="ode", sensitivity_params=["k"]).run(
            t_span=(0.0, 3.0), n_points=2
        )


def test_the_core_refusal_reaches_a_primary_behind_a_derived_parameter():
    """The same check at the engine, without the Python detector in front of it:
    the trigger names only d, so comparing its own references with ['k'] found
    nothing and returned no reason. Its parameter support includes k."""
    trig = (
        f"<apply><gt/><apply><sin/>{_TIME}</apply>"
        "<apply><minus/><ci>d</ci><cn>0.5</cn></apply></apply>"
    )
    core = _derived(trig)._core
    for requested in ("k", "d"):
        reason = core.event_sensitivity_unsupported_reason([requested], [])
        assert reason is not None and f"'{requested}'" in reason
