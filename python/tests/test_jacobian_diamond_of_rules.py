"""A rate law read through a diamond of rules keeps a short inlined form
(issue #1124, item 5).

``r0 := S``, ``r1 := S + 1``, ``r_i := r_{i-1} + r_{i-2}``, and a law
``k·r_{n-1}·c``. Written in as text, function by function, each rule was
copied once per path to it, so the law the analytic Jacobian, codegen and the
native saturable path read was exponential in n: the first ODE run took 1.2 s
at 26 rules and a codegen run 2.8 s, and each two more rules multiplied that
by 2.6. Past a length, the rules are now put in through sympy, each parsed
once, and sympy collects the sum: ``k·(121393·S + 75025)`` at 26.

Oracle: r_{n-1} = a·S + b·(S + 1), with a and b the Fibonacci numbers the
recurrence makes, so dS/dt = -k((a + b)S + b) and
S(t) = (S0 + b/(a + b))·e^(-k(a + b)t) - b/(a + b).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import _jacobian as J

_M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_K = 1e-6


def _coefficients(n):
    coef = [(1, 0), (0, 1)]  # r0 = S, r1 = (S + 1)
    for _ in range(2, n):
        coef.append((coef[-1][0] + coef[-2][0], coef[-1][1] + coef[-2][1]))
    return coef[n - 1]


def _sbml(n, nonlinear=False):
    def comb(a, b):
        if nonlinear:
            return (
                f"<apply><plus/><apply><sin/><ci>{a}</ci></apply>"
                f"<apply><cos/><ci>{b}</ci></apply></apply>"
            )
        return f"<apply><plus/><ci>{a}</ci><ci>{b}</ci></apply>"

    rules = [("r0", "<ci>S</ci>"), ("r1", "<apply><plus/><ci>S</ci><cn>1</cn></apply>")]
    rules += [(f"r{i}", comb(f"r{i - 1}", f"r{i - 2}")) for i in range(2, n)]
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {_M}>{b}</math></assignmentRule>' for v, b in rules
    )
    params = "".join(f'<parameter id="r{i}" constant="false"/>' for i in range(n))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
<listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
<listOfSpecies><species id="S" compartment="c" initialConcentration="5"
 hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters><parameter id="k" value="{_K}" constant="true"/>{params}</listOfParameters>
<listOfRules>{rule_xml}</listOfRules>
<listOfReactions><reaction id="J" reversible="false"><listOfReactants>
<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
<kineticLaw><math {_M}><apply><times/><ci>k</ci><ci>r{n - 1}</ci><ci>c</ci></apply></math>
</kineticLaw></reaction></listOfReactions>
</model></sbml>"""


def _context(n, nonlinear=False):
    ctx = bngsim.Model.from_sbml_string(_sbml(n, nonlinear))._core.functional_jacobian_context()
    (rxn,) = ctx["functional_reactions"]
    return rxn["rate_expr"], dict(ctx["function_map"])


def test_the_inlined_law_is_the_collected_sum():
    law, funcs = _context(26)
    assert J._inlined_length(law, funcs) > 1_000_000
    assert J._inline_functions(law, funcs) == "k*(121393*S + 75025.0)"


def test_a_short_law_is_written_out_as_before():
    """Below the length nothing changes, character for character."""
    law, funcs = _context(6)
    assert J._compact_inlined(law, funcs) is None
    assert J._inline_functions(law, funcs) == (
        "((k*((((((((((S+1.0))+(S)))+((S+1.0))))+((((S+1.0))+(S)))))"
        "+((((((S+1.0))+(S)))+((S+1.0))))))*c))/c"
    )


def test_a_diamond_that_does_not_collect_is_written_out_as_before():
    law, funcs = _context(18, nonlinear=True)
    assert J._inlined_length(law, funcs) > J._INLINE_TEXT_LIMIT
    assert J._compact_inlined(law, funcs) is None


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("n", [26, 40])
def test_the_run_has_its_analytic_jacobian(n, codegen):
    model = bngsim.Model.from_sbml_string(_sbml(n))
    result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
        t_span=(0.0, 1.0), n_points=5, rtol=1e-10, atol=1e-12
    )
    assert model.analytical_jacobian_status == "complete"
    a, b = _coefficients(n)
    rate = _K * (a + b)
    t = np.asarray(result.time)
    want = (5 + b / (a + b)) * np.exp(-rate * t) - b / (a + b)
    np.testing.assert_allclose(np.asarray(result.species)[:, 0], want, rtol=1e-7, atol=1e-9)
