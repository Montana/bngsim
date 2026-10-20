"""A rate law whose ``if()`` condition holds other ``if()`` chains (issue #540).

sympy will not keep a ``Piecewise`` in a condition: it folds the condition and
rewrites it as an ``ITE`` chain, computing ``as_set()`` of every inner
condition on the way. That was about 2 s per rate law of
``mt_music_sequencer``, which declined on its derivation budget after 8 of 14
reactions. The parse now builds the chain itself. These tests cover:

- the parse does not reach sympy's set-theory rewrite;
- what it builds equals what sympy built, on a grid over every threshold;
- a model with the construct attaches its analytical Jacobian, which matches
  finite differences on every branch.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
import sympy as sp
from bngsim import _jacobian as J

CHORD = [
    "if(T < 4, 1, if(T >= 16, if(T < 20, 1, 0), 0))",
    "if(T >= 4, if(T < 8, 1, if(T >= 20, if(T < 24, 1, 0), 0)), 0)",
    "if(T >= 8, if(T < 12, 1, if(T >= 24, if(T < 28, 1, 0), 0)), 0)",
]
FREQ = f"(({CHORD[0]})*fA + ({CHORD[1]})*fF + ({CHORD[2]})*fC)"
DS = f"({FREQ}*(V - offset))"
LAWS = [
    f"if({DS} > 0, {DS}, 0)",
    f"if({DS} < 0, -{DS}/max(V, 0.01), 0)",
    f"if(({CHORD[0]}) > 0 && ({CHORD[1]}) < 1, T, 2)",
]


@pytest.mark.parametrize("law", LAWS)
def test_the_parse_takes_no_set_theory(law, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise AssertionError("sympy's ITE rewrite ran")

    monkeypatch.setattr(sp.Piecewise, "_eval_rewrite_as_ITE", refuse)
    assert J._exprtk_to_sympy(law) is not None


@pytest.mark.parametrize("law", LAWS)
def test_the_parse_equals_sympys_own(law, monkeypatch):
    ours = J._exprtk_to_sympy(law)
    monkeypatch.setitem(J._piecewise_of_if_cache, "built", sp.Piecewise)
    theirs = J._exprtk_to_sympy(law)
    syms = {str(s): s for s in ours.free_symbols | theirs.free_symbols}
    values = dict(offset=0.5, fA=1.1, fF=1.3, fC=1.7)
    for T in np.arange(-1.0, 30.0, 0.5):
        for V in (-3.0, 0.0, 0.5, 2.0):
            point = {**values, "T": float(T), "V": V}
            sub = {syms[k]: v for k, v in point.items() if k in syms}
            assert sp.N(ours.xreplace(sub)) == sp.N(theirs.xreplace(sub)), point


NET = """begin parameters
    1 k 0.5
    2 g 0.2
end parameters
begin species
    1 A() 1.5
    2 B() 1.0
    3 C() 0.0
end species
begin functions
    1 sw() if((A_tot<1),1,if((A_tot>=2),2,0.5))
    2 d() sw()*B_tot-0.3
    3 rate() if((d()>0),k*d(),0)
end functions
begin reactions
    1 2 3 rate
    2 0 1 g
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
    3 C_tot 3
end groups
"""


def _central(m, y, h=1e-6):
    n = y.size
    D = np.zeros((n, n))
    for j in range(n):
        s = h * max(abs(y[j]), 1.0)
        yp, ym = y.copy(), y.copy()
        yp[j] += s
        ym[j] -= s
        D[:, j] = (np.asarray(m.rhs(yp)) - np.asarray(m.rhs(ym))) / (2.0 * s)
    return D


def test_a_model_with_the_construct_attaches_and_matches_fd(tmp_path, monkeypatch):
    for var in ("BNGSIM_JAC_SELFCHECK_DENSE_MAX", "BNGSIM_JAC_SELFCHECK_SAMPLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("BNGSIM_JAC_NO_SELFCHECK", raising=False)
    path = tmp_path / "if_in_condition.net"
    path.write_text(NET)
    m = bngsim.Model.from_net(str(path))
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    # One state on each branch of sw(), with d() on either side of 0, none on
    # a switching surface.
    for A in (0.5, 1.5, 2.5):
        for B in (0.1, 1.0):
            y = np.array([A, B, 0.2])
            Jy = m.jacobian(y)
            assert Jy.source == "analytical"
            np.testing.assert_allclose(np.asarray(Jy), _central(m, y), rtol=1e-6, atol=1e-9)
