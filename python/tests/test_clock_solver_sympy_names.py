"""The clock-threshold solvers read a model's names as the model's (issue #757).

The five sites that solve a time condition for its crossing called
``parse_expr`` with no ``local_dict``, so a bare name went through sympy's own
namespace first: a parameter ``E`` became Euler's number, ``S`` the ``S``
singleton, ``N`` ``evalf``, ``Q`` the assumptions object, ``O`` ``Order`` and
``beta``/``gamma``/``zeta`` special functions. Three things followed, none of
them loud:

* a periodic schedule over such a name was declined, so a plain run got no
  stops at its pulses and CVODE stepped over every one (X(240) = 0, not 10);
* ``time()-E*E >= 0`` solved to ``exp(2)``, so the crossing looked fixed and
  ``dX/dE`` came back an exact 0 instead of -12;
* a pulse over ``gamma``/``N``/``beta``/``S`` raised SensitivityUnsupportedError
  on a valid model.

Each case is checked against its closed form and against a twin whose only
difference is a name sympy does not know.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import (
    _clock_affine_threshold,
    _clock_monomial_threshold,
    _clock_periodic_schedule,
    _clock_quadratic_thresholds,
)

CLOCK = frozenset({"time()", "time"})
SYMPY_NAMES = ["E", "S", "N", "Q", "O", "beta", "gamma", "zeta"]

_T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'

# k while time mod P >= on, else 0: one hour on per 24 h day, 10 days -> X(240) = 10.
_SCHEDULE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="schedule">
  <listOfCompartments>
   <compartment id="C" size="1" constant="true" spatialDimensions="3"/>
  </listOfCompartments>
  <listOfSpecies>
   <species id="X" compartment="C" initialAmount="0" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="1" constant="true"/>
   <parameter id="P" value="24" constant="true"/>
   <parameter id="{on}" value="23" constant="true"/>
  </listOfParameters>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfProducts>
     <speciesReference species="X" stoichiometry="1" constant="true"/>
    </listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><piecewise>
     <piece><ci>k</ci><apply><geq/>
      <apply><minus/>{t}<apply><times/><ci>P</ci>
       <apply><floor/><apply><divide/>{t}<ci>P</ci></apply></apply></apply></apply>
      <ci>{on}</ci></apply></piece>
     <otherwise><cn>0</cn></otherwise>
    </piecewise></math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>"""

_AFFINE_NET = """
begin parameters
    1 {E} 2
    2 k 3
end parameters
begin species
    1 Z() 1
    2 X() 0
end species
begin reactions
    1 1 1,2 rate_X
end reactions
begin functions
    1 rate_X() if(time()-{E}*{E}>=0,k,0)
end functions
"""

_PULSE_NET = """
begin parameters
    1 {g} 2
    2 k 1
end parameters
begin species
    1 Z() 1
    2 X() 0
end species
begin reactions
    1 1 1,2 rate_X
end reactions
begin functions
    1 rate_X() if(time()-{g}>=0 && time()-{g}-1<0,k,0)
end functions
"""


def _net(tmp_path: Path, template: str, **names: str) -> bngsim.Model:
    path = tmp_path / ("m_" + "_".join(names.values()) + ".net")
    path.write_text(textwrap.dedent(template.format(**names)).strip() + "\n")
    return bngsim.Model.from_net(str(path))


def _x_sens(model: bngsim.Model, param: str, t_end: float, n: int) -> np.ndarray:
    r = bngsim.Simulator(model, method="ode", sensitivity_params=[param]).run(
        t_span=(0.0, t_end), n_points=n
    )
    return np.asarray(r.sensitivities)[:, 1, 0]  # X() is the second species


# ─── Plain run: a periodic schedule keeps its stops ─────────────────────────


@pytest.mark.parametrize("name", ["Dd", *SYMPY_NAMES])
def test_a_periodic_schedule_keeps_every_pulse(name):
    """Before the fix every sympy name gave X(240) = 0 with no warning."""
    m = bngsim.Model.from_sbml_string(_SCHEDULE_SBML.format(on=name, t=_T))
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, 240.0), n_points=11)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(10.0, rel=1e-6)


# ─── Sensitivity: an affine threshold keeps its parameter ───────────────────


@pytest.mark.parametrize("name", ["Ep", "E"])
def test_an_affine_threshold_over_E_is_differentiated(tmp_path, name):
    """X = k(t - E^2) past t* = E^2 = 4, so dX/dE = -2kE = -12 there. With `E`
    read as Euler's number the threshold was `exp(2)` and dX/dE was 0."""
    sens = _x_sens(_net(tmp_path, _AFFINE_NET, E=name), name, 8.0, 5)
    np.testing.assert_allclose(sens, [0.0, 0.0, -12.0, -12.0, -12.0], atol=1e-6)


@pytest.mark.parametrize("name", ["gamma", "N", "beta", "S"])
def test_a_pulse_over_a_sympy_name_matches_its_twin(tmp_path, name):
    """This used to raise SensitivityUnsupportedError; the Tq twin never did."""
    twin = _x_sens(_net(tmp_path, _PULSE_NET, g="Tq"), "Tq", 4.0, 5)
    np.testing.assert_allclose(twin, [0.0, 0.0, -1.0, 0.0, 0.0], atol=1e-6)
    sens = _x_sens(_net(tmp_path, _PULSE_NET, g=name), name, 4.0, 5)
    np.testing.assert_allclose(sens, twin, atol=1e-9)


# ─── The recognizers themselves ─────────────────────────────────────────────


@pytest.mark.parametrize("name", SYMPY_NAMES)
def test_the_affine_solver_returns_the_parameter(name):
    # Squared, not doubled: `2*E` prints the same whether E is the parameter or
    # Euler's number, so only `E*E` (which folded to `exp(2)`) tells them apart.
    got = _clock_affine_threshold(f"(time()-{name}*{name})>=0", CLOCK)
    assert got == ("time()", f"{name}^2")


def test_the_monomial_and_quadratic_solvers_return_the_parameters():
    assert _clock_monomial_threshold("time()*time()>=E", CLOCK) == ("time()", "sqrt(E)")
    clock, roots = _clock_quadratic_thresholds("(time()-beta)*(time()-beta)>=zeta", CLOCK)
    assert clock == "time()"
    assert sorted(roots) == ["beta + sqrt(zeta)", "beta - sqrt(zeta)"]


def test_the_periodic_recognizer_returns_the_parameters():
    s = _clock_periodic_schedule("time()-P*floor(time()/P)>=gamma", CLOCK)
    assert (s.period, s.offset, s.duty) == ("P", "0", "gamma")


def test_a_python_keyword_name_is_aliased_and_given_back():
    """`lambda` cannot be tokenized as-is; it is aliased to parse and the solved
    threshold is printed with the model's own name again."""
    assert _clock_affine_threshold("time()-lambda*2>=0", CLOCK) == ("time()", "2*lambda")
