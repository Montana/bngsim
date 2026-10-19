"""Issue #1128: a run that starts on a step of a rounding call over the state.

``X -> Y`` at ``0.01*ceil(0.4*X)*X`` from X(0) = 10 decays at 0.04 until X =
7.5, then at 0.03 until 5, then at 0.02. 0.4*10 = 4 sits on a step of the
``ceil``, which it does not take on the way down. The analytical Jacobian
declined ``ceil`` over the state, and CVODE's difference quotient perturbed X
upward across the step: a slope of about -7e5 for -0.04. Newton's updates
under that matrix are too small to fail, so the corrector kept the predictor
for as long as CVODE kept the matrix, and X came back 5 % off mid-run with no
error (1.9 % at t = 30), at rtol 1e-4 and 1e-6 alike.

The rounding calls now differentiate to 0, which they are between their steps,
and every step is rooted (#1068; SBML's by its GH #244 trigger), so the run is
on the analytical Jacobian.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 kx 1
end parameters
begin species
    1 X() {x0}
    2 Y() 0
end species
begin groups
    1 Xobs 1
end groups
begin functions
    1 f() kx*0.01*{call}(Xobs*0.4)
end functions
begin reactions
    1 1 2 f
end reactions
"""

SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>
   <species id="X" compartment="c" initialConcentration="{x0}"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
   <species id="Y" compartment="c" initialConcentration="0"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters><parameter id="kx" value="1" constant="true"/></listOfParameters>
  <listOfReactions>
   <reaction id="r2" reversible="false">
    <listOfReactants>
     <speciesReference species="X" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <listOfProducts>
     <speciesReference species="Y" stoichiometry="1" constant="true"/>
    </listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><times/><ci>c</ci><ci>kx</ci><ci>X</ci>
      <apply><ceiling/><apply><times/><ci>X</ci><cn>0.4</cn></apply></apply><cn>0.01</cn>
     </apply>
    </math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
"""


def _closed(t: float, x0: float, call: str = "ceil") -> float:
    """X(t): an exponential at 0.01*c between the steps of c = call(0.4*X)."""
    x, at = x0, 0.0
    while True:
        u = 0.4 * x
        # The level the run is on as X falls from x, and where it leaves it.
        c = math.ceil(u - 1e-12) if call == "ceil" else math.floor(u - 1e-12)
        leaves = (c - 1 if call == "ceil" else c) / 0.4
        rate = 0.01 * c
        if rate <= 0:
            return x
        jump = at + math.log(x / leaves) / rate if leaves > 0 else math.inf
        if t <= jump:
            return x * math.exp(-rate * (t - at))
        x, at = leaves, jump


def _model(tmp_path, form: str, x0: float, call: str = "ceil") -> bngsim.Model:
    if form == "net":
        path = tmp_path / "m.net"
        path.write_text(NET.format(x0=repr(x0), call=call))
        return bngsim.Model.from_net(str(path))
    path = tmp_path / "m.xml"
    path.write_text(SBML.format(x0=repr(x0)))
    return bngsim.Model.from_sbml(str(path))


def _worst(model: bngsim.Model, x0: float, rtol: float, call: str = "ceil") -> float:
    run = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 30.0), n_points=31, rtol=rtol, atol=rtol * 1e-3
    )
    x = np.asarray(run.species)[:, 0]  # X is the first species in both forms
    return max(abs(xi / _closed(ti, x0, call) - 1) for ti, xi in zip(run.time, x, strict=True))


@pytest.mark.parametrize("rtol", [1e-6, 1e-8])
@pytest.mark.parametrize("form", ["net", "sbml"])
def test_a_start_on_a_step_is_as_right_as_the_tolerance(tmp_path, form, rtol):
    assert _worst(_model(tmp_path, form, 10.0), 10.0, rtol) < 20 * rtol


@pytest.mark.parametrize("x0", [7.5, 5.0, 10.0 - 1e-9, 9.9])
def test_every_step_and_beside_it(tmp_path, x0):
    """On the other steps, within the difference quotient's reach of one (X
    perturbed by 1.5e-7), and clear of them all."""
    assert _worst(_model(tmp_path, "net", x0), x0, 1e-6) < 2e-5


def test_floor_too(tmp_path):
    assert _worst(_model(tmp_path, "net", 10.0, "floor"), 10.0, 1e-6, "floor") < 2e-5


@pytest.mark.parametrize("form", ["net", "sbml"])
def test_the_model_is_on_the_analytical_jacobian(tmp_path, form):
    model = _model(tmp_path, form, 10.0)
    assert model.prepare_analytical_jacobian() is True
    assert model.analytical_jacobian_status == "complete"
