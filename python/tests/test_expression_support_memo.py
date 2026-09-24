"""The expression_support() memo must follow a re-attached override (issue #773).

`NetworkModel::expression_support()` memoizes, per expression, the species and
parameters it reaches, following an expression-valued parameter into what its
expression reads. The event-jump difference in a forward-sensitivity run only
differences the parameters in that support, and writes dh/dp = 0 for the rest.

The memo was filled once and never cleared, on the argument that the only
post-build change, `set_param`'s detach, can only shrink a support. #188 made
an override reversible: writing back the value the expression gives re-attaches
it. A memo entry filled while detached then keeps the primaries out forever, so
on that Model the event assignment's dS/dk stayed exactly 0 while the trajectory
was right.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# S(0) = 0; at t = 0.5 an event sets S := d, with d = 2*k by initialAssignment.
# Nothing else moves S, so S(1) = d = 2k and dS(1)/dk = 2 while d is attached.
_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="memo">
  <listOfCompartments>
   <compartment id="C" size="1" constant="true" spatialDimensions="3"/>
  </listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialAmount="0" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
   <species id="P" compartment="C" initialAmount="1" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="1" constant="true"/>
   <parameter id="d" constant="true"/>
  </listOfParameters>
  <listOfInitialAssignments>
   <initialAssignment symbol="d"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><cn>2</cn><ci>k</ci></apply></math></initialAssignment>
  </listOfInitialAssignments>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfReactants>
     <speciesReference species="P" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><times/><ci>k</ci><ci>P</ci></apply></math></kineticLaw>
   </reaction>
  </listOfReactions>
  <listOfEvents>
   <event id="e" useValuesFromTriggerTime="true">
    <trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><geq/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
     <cn>0.5</cn></apply></math></trigger>
    <listOfEventAssignments>
     <eventAssignment variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>d</ci></math></eventAssignment>
    </listOfEventAssignments>
   </event>
  </listOfEvents>
 </model>
</sbml>"""


def _sim(m: bngsim.Model) -> bngsim.Simulator:
    return bngsim.Simulator(m, method="ode", sensitivity_params=["k"])


def _run(sim: bngsim.Simulator) -> tuple[float, float]:
    r = sim.run(t_span=(0, 1), n_points=3)
    i = list(r.species_names).index("S")
    return float(np.asarray(r.species)[-1, i]), float(np.asarray(r.sensitivities)[-1, i, 0])


def _s1_and_dsdk(m: bngsim.Model) -> tuple[float, float]:
    return _run(_sim(m))


@pytest.fixture
def model() -> bngsim.Model:
    return bngsim.Model.from_sbml_string(_SBML)


def test_a_fresh_model_differences_the_primary(model):
    """The control: d is attached from the start, so dS(1)/dk = d'(k) = 2."""
    s1, dk = _s1_and_dsdk(model)
    assert s1 == pytest.approx(2.0)
    assert dk == pytest.approx(2.0, rel=1e-8)


def test_an_override_drops_the_primary(model):
    """While d is overridden it is an independent input, so dS/dk = 0 is right."""
    model.set_param("d", 5.0)
    s1, dk = _s1_and_dsdk(model)
    assert s1 == pytest.approx(5.0)
    assert dk == 0.0


def test_a_reattached_override_differences_the_primary_again(model):
    """The reported defect: override, run with sensitivities (fills the memo
    with d's shrunken support), re-attach, run again. dS/dk was exactly 0."""
    model.set_param("d", 5.0)
    _s1_and_dsdk(model)
    model.set_param("d", 2.0)  # the value d = 2k gives, so #188 re-attaches it
    model.reset()
    s1, dk = _s1_and_dsdk(model)
    assert s1 == pytest.approx(2.0)
    assert dk == pytest.approx(2.0, rel=1e-8)

    # And it stays live: a new k moves both the trajectory and the gradient.
    model.set_param("k", 1.5)
    model.reset()
    s1, dk = _s1_and_dsdk(model)
    assert s1 == pytest.approx(3.0)
    assert dk == pytest.approx(2.0, rel=1e-8)


def test_a_reused_simulator_differences_the_primary_after_a_reattach(model):
    """The fitting-loop pattern: one Simulator, built before the override and
    kept across it. The memo lives on the Model, not the Simulator, so this
    path went stale the same way (dS/dk = 0.0 on main)."""
    sim = _sim(model)
    model.set_param("d", 5.0)
    model.reset()
    assert _run(sim)[1] == 0.0
    model.set_param("d", 2.0)
    model.reset()
    s1, dk = _run(sim)
    assert s1 == pytest.approx(2.0)
    assert dk == pytest.approx(2.0, rel=1e-8)
