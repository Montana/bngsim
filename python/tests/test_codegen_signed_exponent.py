"""GH #573 — every unary sign an exponent may carry must survive the rewrite.

``_extract_exp_right`` handled exactly one spelling: a single ``-`` immediately
followed by a bare token. Every other sign the engine accepts fell through to
the unsigned scan, which stops on a sign — ``+`` and ``-`` are not in its
character class — so the sign was left behind as loose text and the exponent
came back empty:

    x^+2     ->  pow(x, )+2
    x^--2    ->  pow(x, -)-2
    x^-(y+2) ->  pow(x, -)(y+2)

None of those is C a compiler will take, so a model carrying one lost codegen
and forward sensitivities while the interpreter evaluated it without complaint —
and ``A^+n`` is a spelling a modeller is entitled to write.

ExprTk folds a run of unary signs the ordinary way (``--n`` is ``n``, ``+-n`` is
``-n``) and allows spaces between them. The scan now collects that run, reduces
it to one effective sign, and takes the operand after it — a parenthesised group
or a bare token, the same two forms the unsigned path already handled.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _expr_to_c

NAMES = ["x", "y", "z"]


def _c(expr: str) -> str:
    return _expr_to_c(expr, NAMES, [], [], [])


# ── The rewrite ──────────────────────────────────────────────────────────────


def test_the_issue_expression():
    assert _c("x^+2") == "pow(p[0], 2.0)"


@pytest.mark.parametrize(
    "expr, want",
    [
        # a single sign, either way round
        ("x^+2", "pow(p[0], 2.0)"),
        ("x^-2", "pow(p[0], -2.0)"),
        ("x^+y", "pow(p[0], p[1])"),
        # spaces around and between the signs
        ("x^ + 2", "pow(p[0], 2.0)"),
        ("x^ - 2", "pow(p[0], -2.0)"),
        # a run of them, folded the way ExprTk folds it
        ("x^--2", "pow(p[0], 2.0)"),
        ("x^+-2", "pow(p[0], -2.0)"),
        ("x^-+2", "pow(p[0], -2.0)"),
        ("x^---2", "pow(p[0], -2.0)"),
        ("x^- - 2", "pow(p[0], 2.0)"),
        # a sign in front of a parenthesised exponent, which used to lose both
        ("x^-(y+2)", "pow(p[0], -(p[1]+2.0))"),
        ("x^+(y+2)", "pow(p[0], (p[1]+2.0))"),
        # and in front of a scientific-notation literal, whose own sign is not
        # part of the run (GH #240)
        ("x^+1e-3", "pow(p[0], 1e-3)"),
        ("x^-1e-3", "pow(p[0], -1e-3)"),
        ("x^--1e-3", "pow(p[0], 1e-3)"),
    ],
)
def test_each_sign_form(expr, want):
    assert _c(expr) == want


@pytest.mark.parametrize(
    "expr, want",
    [
        # unsigned exponents are untouched
        ("x^2", "pow(p[0], 2.0)"),
        ("x^y", "pow(p[0], p[1])"),
        ("x^(y+2)", "pow(p[0], (p[1]+2.0))"),
        # a binary operator after a complete exponent still belongs outside it
        ("x^y-3", "pow(p[0], p[1])-3.0"),
        ("x^2+3", "pow(p[0], 2.0)+3.0"),
        ("x+3^2", "p[0]+pow(3.0, 2.0)"),
        # the GH #555 chain, with and without a sign on a link
        ("x^y^z", "pow(p[0], pow(p[1], p[2]))"),
        ("x^-y^z", "pow(p[0], -pow(p[1], p[2]))"),
    ],
)
def test_what_must_not_change(expr, want):
    assert _c(expr) == want


def test_no_empty_exponent_is_emitted():
    """The signature of the bug, as a property of the output."""
    for expr in ("x^+2", "x^--2", "x^+-y", "x^-(y+2)", "x^ + 1e-3"):
        assert "pow(p[0], )" not in _c(expr)


# ── The models build, and compute what the interpreter computes ──────────────

NET = """begin parameters
    1 k       0.1  # Constant
    2 n       2.0  # Constant
end parameters
begin functions
    1 law() {body}
end functions
begin species
    1 A() 3.0
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 A                    1
end groups
"""


def _run(tmp_path, body: str, codegen: bool):
    net = tmp_path / f"m_{int(codegen)}.net"
    net.write_text(NET.format(body=body))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(net))
        result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
            t_span=(0.0, 2.0), n_points=5
        )
    return np.asarray(result.species)[:, 0]


@pytest.mark.parametrize(
    "body, same_as",
    [
        # each signed spelling against the plain one it is equivalent to
        ("k*A^+n", "k*A^n"),
        ("k*A^ + n", "k*A^n"),
        ("k*A^--n", "k*A^n"),
        ("k*A^+(n)", "k*A^n"),
        ("k*A^+-n", "k*A^-n"),
        ("k*A^-+n", "k*A^-n"),
        ("k*A^-(n)", "k*A^-n"),
    ],
)
def test_the_compiled_path_matches_the_interpreter(tmp_path, body, same_as):
    interpreted = _run(tmp_path, body, codegen=False)
    compiled = _run(tmp_path, body, codegen=True)
    # Same arithmetic in the same order, so this is an equality, not a tolerance.
    assert compiled == pytest.approx(interpreted, rel=0, abs=0)
    # ...and the sign folded the way ExprTk folds it, not merely consistently.
    assert interpreted == pytest.approx(_run(tmp_path, same_as, codegen=False), rel=0, abs=0)
    # ...and the model went somewhere, so none of this is two flat lines.
    assert interpreted[-1] != interpreted[0]
