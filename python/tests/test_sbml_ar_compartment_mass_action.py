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

Off the Elementary path, a law whose rows need different divides took the
one-reaction emission and its single divide by one compartment: an amount
beside a concentration (`H => A; k*H` in `C := 3 + 0.5*time` put H's row over
the live size, [H](5) 0.527 for 0.401), two rule compartments of one load-time
size, a concentration in C beside an amount in a rate-rule compartment of C's
size. They take the per-species emission and its live-size correction too, and
with differing conversion factors, which that emission refuses, the one written
species by species. A rule over constants (``C := 2``) holds its size and keeps
the Elementary rate, so its reversible split still runs under the SSA.
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


def _species(sid: str, amount: float, comp: str = "C", hosu: bool = False, cf: str = "") -> str:
    factor = f' conversionFactor="{cf}"' if cf else ""
    return (
        f'<species id="{sid}" compartment="{comp}" initialAmount="{amount}" '
        f'hasOnlySubstanceUnits="{str(hosu).lower()}" boundaryCondition="false" '
        f'constant="false"{factor}/>'
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


def _doc(
    species: str,
    reaction: str,
    params: str,
    size: str | None = None,
    d_size: float = 1,
    d_rule: str = "",
) -> str:
    """C (size 2) is sized by ``size``, ``2 + time`` by default; D is static
    unless ``d_rule`` (an assignment or rate rule on D) moves it."""
    size = size or _ap("plus", "<cn>2</cn>", TIME)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="m">
  <listOfCompartments><compartment id="C" size="2" constant="false"/>
   <compartment id="D" size="{d_size}" constant="{str(not d_rule).lower()}"/>
  </listOfCompartments>
  <listOfSpecies>{species}</listOfSpecies>
  <listOfParameters>{params}</listOfParameters>
  <listOfRules><assignmentRule variable="C">
   <math {MATHML}>{size}</math></assignmentRule>{d_rule}</listOfRules>
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


def _amount_beside_a_concentration():
    # H an amount, A a concentration, both in C: H is stored over the load-time
    # size and A at its live concentration, so their rows are the law over each
    # of those. amount_H = 6 e^(-k t)
    xml = _doc(
        _species("H", 6, hosu=True) + _species("A", 2),
        _reaction(["H"], ["A"], _ap("times", _ci("k"), _ci("H"))),
        _param("k", 0.2),
    )
    h = 6.0 * np.exp(-0.2 * T)
    return xml, {"H": h / V, "A": (8.0 - h) / V}


def _two_rule_compartments_of_one_size():
    # D := 2 + 2t beside C := 2 + t, both 2 at load: amount_A = 4 (2/V)^k
    two_t = _ap("times", "<cn>2</cn>", TIME)
    xml = _doc(
        _species("A", 4) + _species("B", 0, "D"),
        _reaction(["A"], ["B"], _ap("times", _ci("k"), _ci("A"))),
        _param("k", 0.5),
        d_size=2,
        d_rule=f'<assignmentRule variable="D"><math {MATHML}>'
        f"{_ap('plus', '<cn>2</cn>', two_t)}</math></assignmentRule>",
    )
    a = 4.0 * (2.0 / V) ** 0.5
    return xml, {"A": a / V, "B": (4.0 - a) / (2.0 + 2.0 * T)}


def _amount_in_a_rate_rule_compartment():
    # B an amount in D, D' = 0.5 from 2, C's size at load: amount_A = 4 (2/V)^k
    xml = _doc(
        _species("A", 4) + _species("B", 0, "D", hosu=True),
        _reaction(["A"], ["B"], _ap("times", _ci("k"), _ci("A"))),
        _param("k", 0.5),
        d_size=2,
        d_rule=f'<rateRule variable="D"><math {MATHML}><cn>0.5</cn></math></rateRule>',
    )
    a = 4.0 * (2.0 / V) ** 0.5
    return xml, {"A": a / V, "B": (4.0 - a) / (2.0 + 0.5 * T)}


def _catalyst_written_first():
    # E (amount 1) in D, D' = 0.5 from 2, catalyses A => B in C: amount_A' =
    # -k a/V * 2/(4 + t), so amount_A = 4 (2 V/(4 + t))^(-k)
    xml = _doc(
        _species("E", 1, "D") + _species("A", 4) + _species("B", 0),
        _reaction(["E", "A"], ["E", "B"], _ap("times", _ci("k"), _ci("A"), _ci("E"))),
        _param("k", 0.5),
        d_size=2,
        d_rule=f'<rateRule variable="D"><math {MATHML}><cn>0.5</cn></math></rateRule>',
    )
    a = 4.0 * (2.0 * V / (4.0 + T)) ** -0.5
    return xml, {"A": a / V, "B": (4.0 - a) / V}


def _conversion_factors_differ():
    # amount_A' = -fa k [A] and amount_B' = fb k [A]: amount_A = 4 (2/V)^(fa k)
    xml = _doc(
        _species("A", 4, cf="fa") + _species("B", 0, "D", cf="fb"),
        _reaction(["A"], ["B"], _ap("times", _ci("k"), _ci("A"))),
        _param("k", 0.5) + _param("fa", 2) + _param("fb", 3),
    )
    a = 4.0 * (2.0 / V) ** 1.0
    return xml, {"A": a / V, "B": 1.5 * (4.0 - a)}


CASES = {
    "decay k*A": _decay,
    "zero-order ks": _synthesis,
    "k*A*B": _bimolecular,
    "reversible kf*A - kr*B": _reversible,
    "cross-compartment k*A*B": _cross_compartment,
    "an amount beside a concentration": _amount_beside_a_concentration,
    "two rule compartments of one size": _two_rule_compartments_of_one_size,
    "an amount in a rate-rule compartment": _amount_in_a_rate_rule_compartment,
    "conversion factors differ": _conversion_factors_differ,
    "a catalyst written first, in a rate-rule compartment": _catalyst_written_first,
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(CASES))
def test_a_mass_action_law_reads_the_live_size(case, codegen):
    xml, want = CASES[case]()
    got = _run(xml, codegen)
    for sid, conc in want.items():
        np.testing.assert_allclose(got[sid], conc, rtol=1e-7, atol=1e-12, err_msg=sid)


def test_a_size_over_constants_keeps_its_elementary_split():
    """``C := 2`` holds its load-time size (a write to what such a rule reads is
    refused), so ``kf*A - kr*B`` there still splits into two Elementary channels,
    which the SSA runs. Taken for a moving size, it was refused under the SSA
    as a reversible difference."""
    law = _ap("minus", _ap("times", _ci("kf"), _ci("A")), _ap("times", _ci("kr"), _ci("B")))
    xml = _doc(
        _species("A", 30) + _species("B", 0),
        _reaction(["A"], ["B"], law, reversible=True),
        _param("kf", 0.5) + _param("kr", 0.2),
        size="<cn>2</cn>",
    )
    model = bngsim.Model.from_sbml_string(xml)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    # a' = M a / 2
    m = np.array([[-0.5, 0.2], [0.5, -0.2]])
    a = np.array([expm(m * t / 2.0) @ [30.0, 0.0] for t in T])
    got = _run(xml, codegen=False)
    np.testing.assert_allclose(got["A"], a[:, 0] / 2.0, rtol=1e-7)
    np.testing.assert_allclose(got["B"], a[:, 1] / 2.0, rtol=1e-7)


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


# #787, the same code's rate-rule and event member: A an amount in C1, B a
# concentration in C2, both 2 at load, C2 resized. One Elementary rate divided
# B's row by the load-time 2 ([B](5) 0.2839 for RoadRunner's 0.1957 under the
# rate rule), and the SSA ran it, its propensity k·n_A reading an amount. The
# ODE took the per-species divide and the SSA refused the reaction; it is
# certified for the SSA now, as a monomial over concentrations is.
_C787 = (
    "compartment C1 = 2, C2 = 2; {resize} substanceOnly species A in C1; species B in C2; "
    "A = 200; B = 0; k = 0.3; J1: A {arrow} B; k*A;"
)
_RESIZE = {
    "rate rule": ("C2' = 1;", lambda t: 2.0 + t),
    "event": ("E1: at time >= 3: C2 = 4;", lambda t: np.where(t < 3.0, 2.0, 4.0)),
}


@pytest.mark.parametrize("arrow", ["=>", "->"])
@pytest.mark.parametrize("resize", list(_RESIZE))
def test_an_amount_into_a_resized_compartment_runs_under_the_ssa(resize, arrow):
    """The mean of this first-order SSA is the amount ODE's solution,
    n_A = 200 e^(-k t); the ODE's [B] is (200 - n_A) over C2's live size."""
    pytest.importorskip("antimony")
    rule, size = _RESIZE[resize]
    text = _C787.format(resize=rule, arrow=arrow)
    t = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0])
    n_a = 200.0 * np.exp(-0.3 * t)
    ode = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode").run(
        sample_times=list(t), rtol=1e-10, atol=1e-12
    )
    b = np.asarray(ode.species)[:, list(ode.species_names).index("B")]
    np.testing.assert_allclose(b, (200.0 - n_a) / size(t), rtol=1e-7, atol=1e-10)

    model = bngsim.Model.from_antimony_string(text)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    reps = 1000
    runs = bngsim.Simulator(model, method="ssa").run_batch(
        t_span=(0.0, 6.0), n_points=7, params=[{} for _ in range(reps)], seed=11
    )
    col = list(runs[0].species_names).index("A")
    # A is reported over C1's size, 2
    got = 2.0 * np.stack([np.asarray(r.species)[:, col] for r in runs])
    want = 200.0 * np.exp(-0.3 * np.linspace(0.0, 6.0, 7))
    se = got.std(0, ddof=1)[1:] / np.sqrt(reps)
    z = np.abs(got.mean(0)[1:] - want[1:]) / se
    assert z.max() < 5.0, z
