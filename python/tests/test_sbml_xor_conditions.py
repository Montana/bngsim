"""An SBML ``xor``, or ``eq`` / ``neq`` between two conditions, is read at each
of its comparisons.

The loader wrote ``xor(a, b)`` as ``((a)!=0) != ((b)!=0)``. The engine
evaluates that alike, but everything that reads a condition for where it flips
took it for one comparison of two conditions:

* a compartment sized by ``piecewise(1, xor(S3 < 1.5, S3 > 1), 2)`` failed to
  load, ``unsupported operand type(s) for -: 'StrictLessThan' and
  'StrictGreaterThan'`` (the jump pass subtracts a relation's sides);
* a rate law that branches on one was refused forward sensitivities, as no
  single comparison bngsim can root on, where the same condition written with
  ``and`` / ``or`` / ``not`` gets them exactly.

Both are now written with the connectives. S3 decays from 2 at 0.1·S3, so
S3 < 1.5 from t = 10·ln(4/3) = 2.877, S3 < 1.2 from 10·ln(5/3) = 5.108, and
S3 < 1 from 10·ln 2 = 6.931; every expected value below is exact.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._jacobian import _exprtk_to_sympy
from bngsim._sbml_loader import _xor_text

_M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
)
T = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
T15, T12, T10 = 10 * np.log(4 / 3), 10 * np.log(5 / 3), 10 * np.log(2)


def a(op, *x):
    return f"<apply><{op}/>{''.join(x)}</apply>"


def ci(x):
    return f"<ci>{x}</ci>"


def cn(x):
    return f"<cn>{x}</cn>"


def call(f, x):
    return f"<apply>{ci(f)}{ci(x)}</apply>"


def piecewise(value, cond, otherwise):
    return f"<piecewise><piece>{value}{cond}</piece><otherwise>{otherwise}</otherwise></piecewise>"


LT15 = a("lt", ci("S3"), cn(1.5))
GT1 = a("gt", ci("S3"), cn(1))
LT12 = a("lt", ci("S3"), cn(1.2))
# lt15(x) = x < 1.5, gt1(x) = x > 1: the same conditions through functionDefinitions
_FUNCS = "".join(
    f'<functionDefinition id="{f}"><math {_M}><lambda><bvar><ci>x</ci></bvar>{body}</lambda>'
    "</math></functionDefinition>"
    for f, body in (("lt15", a("lt", ci("x"), cn(1.5))), ("gt1", a("gt", ci("x"), cn(1))))
)


def _species(sid, comp):
    conc = {"S3": 2, "X": 4}[sid]
    return (
        f'<species id="{sid}" compartment="{comp}" initialConcentration="{conc}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _decay(rid, sid, law):
    return (
        f'<reaction id="{rid}" reversible="false"><listOfReactants>'
        f'<speciesReference species="{sid}" stoichiometry="1" constant="true"/>'
        f"</listOfReactants><kineticLaw><math {_M}>{law}</math></kineticLaw></reaction>"
    )


def _doc(x_comp, x_law, rules=""):
    """S3 in c, decaying at kd·S3 (kd = 0.1); X in x_comp, at x_law."""
    cv = '<compartment id="cv" size="1" constant="false"/>' if x_comp == "cv" else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        f'<model id="m"><listOfFunctionDefinitions>{_FUNCS}</listOfFunctionDefinitions>'
        f'<listOfCompartments><compartment id="c" size="1" constant="true"/>{cv}'
        f"</listOfCompartments><listOfSpecies>{_species('S3', 'c')}{_species('X', x_comp)}"
        '</listOfSpecies><listOfParameters><parameter id="k" value="0.3" constant="true"/>'
        '<parameter id="kd" value="0.1" constant="true"/>'
        f"</listOfParameters><listOfRules>{rules}</listOfRules><listOfReactions>"
        f"{_decay('d3', 'S3', a('times', ci('kd'), ci('S3'), ci('c')))}{_decay('dX', 'X', x_law)}"
        "</listOfReactions></model></sbml>"
    )


def _x(model, sens=False):
    params = {"sensitivity_params": ["k", "kd"]} if sens else {}
    result = bngsim.Simulator(model, method="ode", **params).run(sample_times=list(T))
    i = list(result.species_names).index("X")
    x = np.asarray(result.species)[:, i]
    return (x, np.asarray(result.sensitivities)[:, i, :]) if sens else x


# xor(S3 < 1.5, S3 > 1): true before 2.877 and after 6.931
XOR_TRUE = (T < T15) | (T > T10)


@pytest.mark.parametrize(
    "cond, true",
    [
        (a("xor", LT15, GT1), XOR_TRUE),
        (a("neq", LT15, GT1), XOR_TRUE),
        (a("eq", LT15, GT1), ~XOR_TRUE),
        (a("xor", call("lt15", "S3"), call("gt1", "S3")), XOR_TRUE),
        (a("neq", call("lt15", "S3"), call("gt1", "S3")), XOR_TRUE),
    ],
    ids=["xor", "neq", "eq", "xor-of-calls", "neq-of-calls"],
)
def test_a_size_that_jumps_on_two_conditions_compared_loads_and_rescales(cond, true):
    """cv := piecewise(1, cond, 2) steps between 1 and 2 at each flip, and X's
    amount, 4·cv(0)·e^(-t/4), is kept across each: [X] = amount / cv(t)."""
    size = piecewise(cn(1), cond, cn(2))
    rule = f'<assignmentRule variable="cv"><math {_M}>{size}</math></assignmentRule>'
    model = bngsim.Model.from_sbml_string(
        _doc("cv", a("times", cn(0.25), ci("X"), ci("cv")), rule)
    )
    cv = np.where(true, 1, 2)
    np.testing.assert_allclose(_x(model), 4 * cv[0] * np.exp(-T / 4) / cv, rtol=1e-5)


def _f(true, t):
    return 1.0 if true(t) else 0.2


def _integral(true, edges):
    """F(t) = ∫f over [0, t], f = 1 where the condition holds and 0.2
    elsewhere, exact for a condition that flips only at ``edges``."""
    grid = np.unique(np.concatenate([T, edges]))
    f = [_f(true, m) for m in 0.5 * (grid[:-1] + grid[1:])]
    return np.interp(T, grid, np.concatenate([[0.0], np.cumsum(f * np.diff(grid))]))


def _integral_dkd(true, edges):
    """dF/dkd. A flip of S3's comparison sits at c/kd, so kd moves it by
    -edge/kd, and F gains f(edge⁻) - f(edge⁺) per unit it moves: the jump
    term that a mis-signed or doubled root would get wrong."""
    return sum(
        (
            np.where(e < T, (_f(true, e - 1e-9) - _f(true, e + 1e-9)) * (-e / 0.1), 0.0)
            for e in edges
        ),
        np.zeros_like(T),
    )


@pytest.mark.parametrize(
    "cond, true, edges",
    [
        (a("xor", LT15, GT1), lambda t: (t > T15) != (t < T10), (T15, T10)),
        (a("neq", LT15, GT1), lambda t: (t > T15) != (t < T10), (T15, T10)),
        (a("eq", LT15, GT1), lambda t: (t > T15) == (t < T10), (T15, T10)),
        (
            a("xor", call("lt15", "S3"), call("gt1", "S3")),
            lambda t: (t > T15) != (t < T10),
            (T15, T10),
        ),
        (
            a("xor", LT15, GT1, LT12),
            lambda t: (int(t > T15) + int(t < T10) + int(t > T12)) % 2 == 1,
            (T15, T12, T10),
        ),
        (
            a("xor", a("lt", _TIME, cn(2)), a("gt", _TIME, cn(5))),
            lambda t: (t < 2) != (t > 5),
            (),  # the clock's flips, at 2 and 5, which kd does not move
        ),
    ],
    ids=["state", "neq", "eq", "of-calls", "three-way", "clock"],
)
def test_a_rate_law_that_branches_on_xor_gets_exact_sensitivities(cond, true, edges):
    """X' = -k·X·f, f = 1 or 0.2 as the xor reads: X = 4·e^(-k·F(t)) with F =
    ∫f, so dX/dk = -F·X, and dX/dkd = -k·X·dF/dkd, where kd moves each flip
    of S3's comparisons. Refused before, as no single comparison bngsim can
    root on."""
    law = a("times", ci("k"), ci("X"), piecewise(cn(1), cond, cn(0.2)))
    F = _integral(true, edges or (2.0, 5.0))
    x, dx = _x(bngsim.Model.from_sbml_string(_doc("c", law)), sens=True)
    np.testing.assert_allclose(x, 4 * np.exp(-0.3 * F), rtol=1e-5)
    np.testing.assert_allclose(dx[1:, 0], -F[1:] * x[1:], rtol=1e-5)
    np.testing.assert_allclose(
        dx[1:, 1], -0.3 * x[1:] * _integral_dkd(true, edges)[1:], rtol=1e-4, atol=1e-7
    )


def test_xor_text_grows_quadratically_not_exponentially():
    """Halved, not folded: a left fold writes the first operand 2^(n-1) times."""
    text = _xor_text([f"(c{i})" for i in range(16)])
    assert text.count("(c0)") <= 16, text.count("(c0)")


def test_the_parse_writes_a_comparison_of_two_conditions_with_connectives():
    """Whatever writes `(a) != (b)` over two conditions, the parse reads the
    xor (and `==` the xnor), not one relation between two relations; a
    comparison of two numbers, symbols included, is left alone."""
    from sympy import Symbol
    from sympy.core.relational import Relational

    for text, xor in (("((S3<1.5)!=(S3>1))", True), ("((S3<1.5)==(S3>1))", False)):
        cond = _exprtk_to_sympy(f"if({text}, 1, 2)").args[0].cond
        assert not any(isinstance(r.lhs, Relational) for r in cond.atoms(Relational)), cond
        for s3 in (0.5, 1.2, 2.0):
            want = (s3 < 1.5) != (s3 > 1) if xor else (s3 < 1.5) == (s3 > 1)
            assert bool(cond.subs(Symbol("S3"), s3)) is want, (text, s3)
    numeric = _exprtk_to_sympy("if(k != S3, 1, 2)").args[0].cond
    assert isinstance(numeric, Relational) and {str(s) for s in numeric.args} == {"k", "S3"}
