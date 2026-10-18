"""Issue #745: a mass-action law in an assignment-rule compartment whose power does not cancel.

In ``C := 2 + time``, a bare law over concentrations (``k*A``, a zero-order
``ks``, ``k*A*B``, a reversible ``kf*A - kr*B``) is in amount per time, and SBML's
d[A]/dt is that over the live size, less the dilution. The classifier's
variable-volume guard knew rate-rule and event compartments only, so these laws
stayed Elementary with the load-time size baked into the rate: ``k*A`` ran 36%
off at t = 4 and the zero-order synthesis 100%, on both right-hand sides, with
nothing said. A ``C*k*A`` law, whose power cancels, was right and still is.

Each is checked against the SBML amount equations, amount' = law([S] = amount/V(t)),
solved in closed form where they have one.

A cross-compartment law (BIOMD0000000342's re9) is emitted per species, and the
engine divided a species in the assignment-rule compartment by its static size,
having no state column to read the live one from. The difference to the live
size is now added as its own reaction on that species, as the dilution is; the
published 342 matches its rate-rule spelling where it was 37% off.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

MATHML = 'xmlns="http://www.w3.org/1998/Math/MathML"'
TIME = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
T = np.linspace(0.0, 4.0, 5)
V = 2.0 + T


def _ap(op: str, *args: str) -> str:
    return f"<apply><{op}/>{''.join(args)}</apply>"


def _ci(x: str) -> str:
    return f"<ci>{x}</ci>"


def _species(sid: str, amount: float, comp: str = "C") -> str:
    return (
        f'<species id="{sid}" compartment="{comp}" initialAmount="{amount}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _ref(sid: str) -> str:
    return f'<speciesReference species="{sid}" stoichiometry="1" constant="true"/>'


def _reaction(reactants, products, law, reversible=False) -> str:
    r = f"<listOfReactants>{''.join(map(_ref, reactants))}</listOfReactants>" if reactants else ""
    p = f"<listOfProducts>{''.join(map(_ref, products))}</listOfProducts>" if products else ""
    return (
        f'<reaction id="J" reversible="{str(reversible).lower()}">{r}{p}'
        f"<kineticLaw><math {MATHML}>{law}</math></kineticLaw></reaction>"
    )


def _doc(species: str, reaction: str, params: str, size: str | None = None) -> str:
    size = size or _ap("plus", "<cn>2</cn>", TIME)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="m">
  <listOfCompartments><compartment id="C" size="2" constant="false"/>
   <compartment id="D" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>{species}</listOfSpecies>
  <listOfParameters>{params}</listOfParameters>
  <listOfRules><assignmentRule variable="C">
   <math {MATHML}>{size}</math></assignmentRule></listOfRules>
  <listOfReactions>{reaction}</listOfReactions>
 </model>
</sbml>"""


def _param(pid: str, value: float) -> str:
    return f'<parameter id="{pid}" value="{value}" constant="true"/>'


def _run(xml: str, codegen: bool) -> dict[str, np.ndarray]:
    model = bngsim.Model.from_sbml_string(xml)
    run = bngsim.Simulator(model, method="ode", codegen=codegen).run(
        sample_times=list(T), rtol=1e-10, atol=1e-12
    )
    sp = np.asarray(run.species)
    return {n: sp[:, i] for i, n in enumerate(run.species_names)}


def _decay():
    # amount' = -k*amount/V: amount = 4 (2/V)^k
    xml = _doc(
        _species("A", 4), _reaction(["A"], [], _ap("times", _ci("k"), _ci("A"))), _param("k", 0.5)
    )
    return xml, {"A": 4.0 * (2.0 / V) ** 0.5 / V}


def _synthesis():
    # amount' = ks
    xml = _doc(_species("A", 0), _reaction([], ["A"], _ci("ks")), _param("ks", 3))
    return xml, {"A": 3.0 * T / V}


def _bimolecular():
    law = _ap("times", _ci("k"), _ci("A"), _ci("B"))
    xml = _doc(
        _species("A", 4) + _species("B", 2) + _species("P", 0),
        _reaction(["A", "B"], ["P"], law),
        _param("k", 0.5),
    )

    def rhs(t, y):
        flux = 0.5 * (y[0] / (2 + t)) * (y[1] / (2 + t))
        return [-flux, -flux, flux]

    a = solve_ivp(rhs, (0, 4), [4, 2, 0], t_eval=T, rtol=1e-12, atol=1e-14).y
    return xml, {"A": a[0] / V, "B": a[1] / V, "P": a[2] / V}


def _reversible():
    # a' = M a / V(t), so a(t) = expm(M ln(V/2)) a(0)
    law = _ap("minus", _ap("times", _ci("kf"), _ci("A")), _ap("times", _ci("kr"), _ci("B")))
    xml = _doc(
        _species("A", 4) + _species("B", 0),
        _reaction(["A"], ["B"], law, reversible=True),
        _param("kf", 0.5) + _param("kr", 0.2),
    )
    m = np.array([[-0.5, 0.2], [0.5, -0.2]])
    a = np.array([expm(m * np.log(v / 2.0)) @ [4.0, 0.0] for v in V])
    return xml, {"A": a[:, 0] / V, "B": a[:, 1] / V}


def _cross_compartment():
    # A in C (2 + t), B and P in D (size 1): the flux is k [A][B] in amount per time
    law = _ap("times", _ci("k"), _ci("A"), _ci("B"))
    xml = _doc(
        _species("A", 4) + _species("B", 2, "D") + _species("P", 0, "D"),
        _reaction(["A", "B"], ["P"], law),
        _param("k", 0.5),
    )

    def rhs(t, y):
        flux = 0.5 * (y[0] / (2 + t)) * y[1]
        return [-flux, -flux, flux]

    a = solve_ivp(rhs, (0, 4), [4, 2, 0], t_eval=T, rtol=1e-12, atol=1e-14).y
    return xml, {"A": a[0] / V, "B": a[1], "P": a[2]}


CASES = {
    "decay k*A": _decay,
    "zero-order ks": _synthesis,
    "k*A*B": _bimolecular,
    "reversible kf*A - kr*B": _reversible,
    "cross-compartment k*A*B": _cross_compartment,
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(CASES))
def test_a_mass_action_law_reads_the_live_size(case, codegen):
    xml, want = CASES[case]()
    got = _run(xml, codegen)
    for sid, conc in want.items():
        np.testing.assert_allclose(got[sid], conc, rtol=1e-7, atol=1e-12, err_msg=sid)


def test_a_law_whose_power_cancels_is_unchanged():
    """``C*k*A``: amount' = -k*amount, whatever C does."""
    law = _ap("times", _ci("C"), _ci("k"), _ci("A"))
    xml = _doc(_species("A", 4), _reaction(["A"], [], law), _param("k", 0.5))
    got = _run(xml, codegen=False)
    np.testing.assert_allclose(got["A"], 4.0 * np.exp(-0.5 * T) / V, rtol=1e-7)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
def test_a_cross_compartment_law_where_the_size_reads_a_species(codegen):
    """C := 2 + P, with P made in D by the law: the size moves with the state,
    and the live divide follows it through C's rule."""
    law = _ap("times", _ci("k"), _ci("A"), _ci("B"))
    xml = _doc(
        _species("A", 4) + _species("B", 2, "D") + _species("P", 0, "D"),
        _reaction(["A", "B"], ["P"], law),
        _param("k", 0.5),
        size=_ap("plus", "<cn>2</cn>", _ci("P")),
    )

    def rhs(t, y):
        flux = 0.5 * (y[0] / (2 + y[2])) * y[1]
        return [-flux, -flux, flux]

    a = solve_ivp(rhs, (0, 4), [4, 2, 0], t_eval=T, rtol=1e-12, atol=1e-14).y
    got = _run(xml, codegen)
    np.testing.assert_allclose(got["A"], a[0] / (2 + a[2]), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(got["P"], a[2], rtol=1e-7, atol=1e-12)
