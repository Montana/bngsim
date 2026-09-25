"""GH #706: the non-finite-RHS retry clamps only concentrations.

The GH #135 retry re-evaluates a non-finite RHS on a copy of the state with
negative components set to 0, which rescues a sqrt/fractional-power law at a
concentration the predictor pushed a hair below zero. It set EVERY negative
component to 0, so a quantity that is negative by design followed the wrong
ODE for the rest of the run, silently: an SBML rate-rule parameter V = -61
relaxing toward -70 ran away past -287, and a rate-rule X fed by a boundary
species B = -5 froze at -4.32. Now only slots that are concentrations by
construction are clamped.

In both models S is consumed at C*k*sqrt(S), reaches 0 at t = 2 and parks a
hair below it, which is what fires the retry. V and X do not read S:
V(t) = -70 + 9 exp(-0.1 t) and X(t) = -5 (1 - exp(-t)).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

M = "http://www.w3.org/1998/Math/MathML"
_SQRT_SINK = (
    '<reaction id="R1" reversible="false"><listOfReactants>'
    '<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>'
    f'<kineticLaw><math xmlns="{M}"><apply><times/><ci>C</ci><ci>k</ci>'
    "<apply><root/><ci>S</ci></apply></apply></math></kineticLaw></reaction>"
)
_S = (
    '<species id="S" compartment="C" initialConcentration="1" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>'
)


def _doc(species: str, params: str, rule: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
<model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>{_S}{species}</listOfSpecies>
<listOfParameters><parameter id="k" value="1" constant="true"/>{params}</listOfParameters>
<listOfRules>{rule}</listOfRules>
<listOfReactions>{_SQRT_SINK}</listOfReactions>
</model></sbml>"""


v1 = _doc(
    "",
    '<parameter id="V" value="-61" constant="false"/>'
    '<parameter id="g" value="0.1" constant="true"/>'
    '<parameter id="E" value="-70" constant="true"/>',
    f'<rateRule variable="V"><math xmlns="{M}"><apply><times/><ci>g</ci>'
    "<apply><minus/><ci>E</ci><ci>V</ci></apply></apply></math></rateRule>",
)
v2 = _doc(
    '<species id="B" compartment="C" initialConcentration="-5" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="true" constant="true"/>',
    '<parameter id="X" value="0" constant="false"/>',
    f'<rateRule variable="X"><math xmlns="{M}">'
    "<apply><minus/><ci>B</ci><ci>X</ci></apply></math></rateRule>",
)


def _col(r, name):
    if name in r.species_names:
        return np.asarray(r.species[:, list(r.species_names).index(name)])
    return np.asarray(r.observables[name])


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
def test_a_negative_rate_rule_parameter_follows_its_own_ode(codegen):
    r = bngsim.Simulator(bngsim.Model.from_sbml_string(v1), method="ode", codegen=codegen).run(
        t_span=(0, 40), n_points=41
    )
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "V"), -70.0 + 9.0 * np.exp(-0.1 * t), rtol=1e-4)
    assert _col(r, "S").min() < 0.0  # the retry did fire


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
def test_a_negative_boundary_species_keeps_its_value(codegen):
    r = bngsim.Simulator(bngsim.Model.from_sbml_string(v2), method="ode", codegen=codegen).run(
        t_span=(0, 40), n_points=41
    )
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "X"), -5.0 * (1.0 - np.exp(-t)), rtol=1e-4, atol=1e-6)
