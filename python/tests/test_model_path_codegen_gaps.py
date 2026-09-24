"""The two model-path codegen gaps #803's parity sweep found, closed.

1. A reaction whose statistical factor is an integer at or above 2**64 compiled
   to an integer literal C cannot represent (``str(int(1e24))``), so every
   ``codegen=True`` and every forward-sensitivity run of such a model failed to
   compile -- an Antimony/SBML kinetic law ``1e24*k*A`` folds its literal into
   that factor, and BNG2.pl writes one for a small cBNGL compartment.
2. Issue #734: ExprTk reads a lone ``=`` (and ``<>``) as equality, ``--x`` as
   ``-(-x)`` and a relational chain on one left-associative level
   (``a==b<1`` is ``(a==b)<1``). The C translators and both sympy parsers
   copied operators through as text, so C read an assignment, a decrement and
   ``a==(b<1)``, and sympy a Python chained comparison.

Every oracle is a closed form or the ExprTk interpreter, whose reading of the
#734 spellings BNG2.pl's own parenthesisation confirms (``if(((a==b)<1),..)``).
"""

from __future__ import annotations

import math

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _run(model, **kw):
    sim = bngsim.Simulator(model, method="ode", **kw)
    if kw.get("codegen"):
        # Compiled code on the path under test, not a quiet ExprTk run that would
        # match the interpreter trivially.
        assert sim.codegen_backend in ("cc", "mir")
    return sim.run(t_span=(0.0, 1.0), n_points=3, rtol=1e-10, atol=1e-12)


# ── 1. integer stat factors past 2**64 ───────────────────────────────────────


@pytest.mark.parametrize(
    ("x", "c"),
    [
        (2.0, "2"),
        (-3.0, "-3"),
        (0.5, "0.5"),
        (1.6605779e-09, "1.6605779e-09"),
        (2.0**53 - 1, "9007199254740991"),  # still an exact integer literal
        (2.0**53, "9007199254740992.0"),
        (1e24, "1e+24"),  # was 999999999999999983222784, which C rejects
    ],
)
def test_c_scalar_spells_every_double_as_a_valid_c_literal(x, c):
    assert cg._c_scalar(x) == c
    assert float(c) == x


BIG = """
model m
  compartment c = 1
  species A in c = 10
  k = 1e-24
  R1: A -> ; 1e24*k*A
end
"""


def test_large_literal_rate_factor_compiles_and_matches_the_closed_form():
    """A -> 0 at 1e24*k*A with k = 1e-24: A = 10 e^-t, dA/dk = -1e24 t A."""
    m = bngsim.Model.from_antimony_string(BIG)
    assert [r["stat_factor"] for r in m._core.codegen_data()["reactions"]] == [1e24]
    t = np.array([0.0, 0.5, 1.0])
    a_exact = 10 * np.exp(-t)
    r = _run(m, codegen=True)
    np.testing.assert_allclose(r.species[:, 0], a_exact, rtol=1e-7)
    r = _run(bngsim.Model.from_antimony_string(BIG), sensitivity_params=["k"])
    np.testing.assert_allclose(r.species[:, 0], a_exact, rtol=1e-7)
    np.testing.assert_allclose(r.sensitivities[:, 0, 0], -1e24 * t * a_exact, rtol=1e-6)


# ── 2. ExprTk spellings (issue #734) ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "normal"),
    [
        ("if(a==b<1,1,2)", "if((a==b)<1,1,2)"),
        ("if(k!=b>1,1,2)", "if((k!=b)>1,1,2)"),
        ("if(a<b<c,1,0)", "if((a<b)<c,1,0)"),  # Python would chain this
        ("f(a==b<1, c<d>e)", "f((a==b)<1, (c<d)>e)"),
        ("a==b<1 ? c<d<e : 0", "(a==b)<1 ? (c<d)<e : 0"),
        ("if(k=2,1,2)", "if(k==2,1,2)"),
        ("a<>b", "a!=b"),
        ("b*--k", "b*- -k"),
        ("a--b", "a- -b"),
        ("---x", "- - -x"),
    ],
)
def test_normalize_makes_exprtk_reading_explicit(text, normal):
    assert cg._normalize_exprtk_operators(text) == normal


@pytest.mark.parametrize(
    "text",
    [
        "k*A",
        "if(((t>=tau)&&(t<14)),r0,0)",  # how BNG2.pl writes every condition
        "if((a==b)<1,1,2)",
        "x<=1&&y>2",
        "(x<y)==(z>w)",
        "2.5E-3*k",
        "1e+24*k",
        "a:=b",
        "tfun('a=b<c.tfun', T)",  # a quoted file name is not an expression
        "if((a==b<1,1,2)",  # unbalanced: left for the parser to refuse
    ],
)
def test_normalize_leaves_unambiguous_text_byte_identical(text):
    assert cg._normalize_exprtk_operators(text) == text


@pytest.mark.parametrize(
    ("text", "normal"),
    [("k'", "k'"), ("a'b<c<d", "(a'b<c)<d"), ('x<1<2 + "', '(x<1)<2 + "')],
)
def test_normalize_reads_an_unpaired_quote_as_plain_text(text, normal):
    """A quote with no partner splits into no quoted piece. Recursing on the
    unchanged text never ended (RecursionError); it is plain text instead."""
    assert cg._normalize_exprtk_operators(text) == normal


OPS = """begin parameters
    1 k 3
    2 a 0.5
    3 b 2
end parameters
begin species
    1 $Src() 1
    2 S0() 0
    3 S1() 0
    4 S2() 0
    5 D() 0
end species
begin functions
    1 f0() F0
    2 f1() F1
    3 f2() F2
end functions
begin reactions
    1 1 1,2 f0
    2 1 1,3 f1
    3 1 1,4 f2
    4 1 1,5 k
end reactions
"""


@pytest.mark.parametrize(
    ("f0", "f1", "f2", "expected"),
    [
        # (a==b)<1 -> 1 and (k!=b)>1 -> 0; C read 2 and 1.
        ("if(a==b<1,1,2)", "if(k!=b>1,1,2)", "b*k", [1, 2, 6, 3]),
        # k==2 is false; C assigned p[k]=2, so D grew at 2 and S2 at 4.
        ("if(k=2,1,2)", "1", "b*k", [2, 1, 6, 3]),
        # -(-k) = 3; C decremented p[k] on every evaluation.
        ("1", "1", "b*--k", [1, 1, 6, 3]),
    ],
    ids=["relational-chain", "lone-equals", "double-negation"],
)
@pytest.mark.parametrize("path", ["net", "model"])
@pytest.mark.parametrize("sens", [False, True], ids=["codegen", "sensitivity"])
def test_exprtk_spellings_compile_to_what_the_interpreter_computes(
    tmp_path, f0, f1, f2, expected, path, sens
):
    """Constant-rate synthesis from a fixed source over t in [0, 1], so each
    species at t = 1 is its rate: (S0, S1, S2, D) = (f0, f1, f2, k)."""
    net = tmp_path / "ops.net"
    net.write_text(OPS.replace("F0", f0).replace("F1", f1).replace("F2", f2))
    ref = _run(bngsim.Model.from_net(net), codegen=False).species[-1, 1:]
    np.testing.assert_allclose(ref, expected, rtol=1e-9)
    m = bngsim.Model.from_net(net)
    if path == "model":
        m._net_path = ""  # the switch Simulator reads (#803)
    r = _run(m, **({"sensitivity_params": ["b"]} if sens else {"codegen": True}))
    np.testing.assert_allclose(r.species[-1, 1:], expected, rtol=1e-9)
    assert m.get_param("k") == 3.0
    if sens:  # d(S2)/db = k t for f2 = b*k (or b*(-(-k)))
        np.testing.assert_allclose(r.sensitivities[-1, 3, 0], 3.0, rtol=1e-7)
    assert math.isfinite(float(r.species[-1, 4]))


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("if(a==b<1,1,2)", 1.0),  # (0.5==2)<1
        ("if(k!=b>1,1,2)", 2.0),  # (3!=2)>1 is false
        ("if(k=2,1,2)", 2.0),  # k==2 is false; was invalid Python
        ("if(a<>b,1,2)", 1.0),  # was invalid Python
        ("b*--k", 6.0),
    ],
)
def test_jax_translator_reads_exprtk_spellings_as_the_interpreter_does(text, value):
    """The third translator (``jacobian="jax"``, diffrax) applies the same
    normaliser. numpy stands in for ``jnp``, so no JAX install is needed."""
    from bngsim._jax_rhs import _translate_expr_jax

    py = _translate_expr_jax(text, {"k": 0, "a": 1, "b": 2}, {}, set(), [])
    params = np.array([3.0, 0.5, 2.0])
    assert float(eval(py, {"jnp": np, "params": params})) == value
