"""GH #574 — differentiating a state-in-both power must not synthesize log(state).

``_takes_log_of_state`` (GH #336) reads the rate law as written, and a power is
written without a logarithm: ``k*A^B`` mentions none, so the guard passed it
through to the native path. The logarithm appears during differentiation.
``_diff_pow``'s constant-base branch emits ``base^exp · ln(base) · d(exp)``
whenever the base does not move with the target — and "does not move with *this*
target" is not "is a constant". Differentiating ``A^B`` with respect to ``B``
takes that branch with the state variable ``A`` as the base and writes
``ln(A)``.

That is the singularity #336 defers for. At ``A = 0`` the emitted
``pow(A, B) * log(A)`` is ``0 · -inf`` — NaN, where the derivative's limit is a
finite ``0`` — and ``A = 0`` is an ordinary initial condition, not a corner. Two
ways it hurt:

* seeded at ``A = 0``, the #183 probe caught the NaN and refused the analytic
  Jacobian, so the model silently ran on finite differences;
* seeded anywhere else the probe saw a finite value, the analytic Jacobian
  attached, and it returned NaN as soon as the trajectory reached ``A = 0``.

The law is now deferred to the SymPy emitters, whose zero-base guard
(#310/#317) emits ``((A == 0.0) && (B > 0.0)) ? 0.0 : pow(A, B)*log(A)`` and is
finite there. A constant base (``2^B``) keeps its native ``ln(base)``: that is a
number, with no singularity the state can reach.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._jacobian import attach_functional_jacobian
from bngsim._saturable_jacobian import differentiate_rate_law_native, emit_c

CONSTANTS = {"k", "n", "KM", "h"}


def _native(expr: str, states: set[str]):
    return differentiate_rate_law_native(expr, {}, states, CONSTANTS)


# ── What the native path must hand over ──────────────────────────────────────


def test_the_issue_expression_defers():
    """It used to return `(k * (pow(A, B) * log(A)))`, NaN at A=0."""
    assert _native("k*A^B", {"A", "B"}) is None


@pytest.mark.parametrize(
    "expr, states",
    [
        ("k*A^B", {"A", "B"}),
        # the power need not be the whole law, nor at the top of it
        ("k*A^B + n*A", {"A", "B"}),
        ("k/(1 + A^B)", {"A", "B"}),
        ("exp(A^B)", {"A", "B"}),
        ("-(A^B)", {"A", "B"}),
        # a compound base or exponent that still mentions state on both sides
        ("k*(A+1)^(B+1)", {"A", "B"}),
    ],
)
def test_a_state_in_both_base_and_exponent_defers(expr, states):
    assert _native(expr, states) is None


# ── What must stay native ────────────────────────────────────────────────────


def test_a_constant_base_keeps_its_logarithm():
    """`ln(2)` is a number; nothing the state does can make it singular."""
    out = _native("k*2^B", {"B"})
    assert out is not None
    assert emit_c(out["B"], lambda name: name) == "(k * (pow(2.0, B) * log(2.0)))"


def test_a_constant_exponent_is_unchanged():
    """The other branch — `n·A^(n-1)` — synthesizes no logarithm at all."""
    out = _native("k*A^n", {"A"})
    assert out is not None
    assert emit_c(out["A"], lambda name: name) == "(k * (n * pow(A, (n - 1.0))))"


def test_a_power_of_constants_is_still_a_zero_column():
    assert _native("k*KM^h", set()) == {}


def test_the_same_variable_in_both_positions_still_defers():
    """`A^A` was already deferred, by _diff_pow's general-power branch."""
    assert _native("k*A^A", {"A"}) is None


# ── The Jacobian a model ends up with ────────────────────────────────────────

NET = """begin parameters
    1 k       0.5
end parameters
begin functions
    1 law() k*A^B
end functions
begin species
    1 A() {a0}
    2 B() 2.0
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 A                    1
    2 B                    2
end groups
"""

# d/dB of the reaction rate k·A^B·A is k·A^(B+1)·ln(A); A is consumed, so the
# entry is its negative. At A=1.5, B=2: -0.5·1.5³·ln(1.5).
J_AB_AT_1_5 = -0.5 * 1.5**3 * np.log(1.5)


def _attached_core(tmp_path, a0: float):
    net = tmp_path / f"m_{a0}.net"
    net.write_text(NET.format(a0=a0))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(net))
        core = model._core
        attached = attach_functional_jacobian(core)
    return core, attached


@pytest.mark.parametrize("a0", [0.0, 1.5])
def test_the_analytic_jacobian_attaches_from_either_seed(tmp_path, a0):
    """Seeded at A=0 the NaN probe used to refuse it outright, dropping the
    model onto finite differences without the caller asking."""
    _, attached = _attached_core(tmp_path, a0)
    assert attached


@pytest.mark.parametrize("a0", [0.0, 1.5])
def test_the_jacobian_is_finite_at_a_zero_base(tmp_path, a0):
    """Seeded away from zero the probe passed and the NaN waited for the
    trajectory to arrive at A=0. It is a finite 0 now, from either seed."""
    core, _ = _attached_core(tmp_path, a0)
    jac = np.asarray(core.fill_dense_analytical_jacobian(0.0, [0.0, 2.0]))
    assert np.isfinite(jac).all()
    assert jac[0][1] == 0.0


@pytest.mark.parametrize("a0", [0.0, 1.5])
def test_the_jacobian_is_right_away_from_zero(tmp_path, a0):
    """The guard must not cost the value everywhere else: compared against the
    closed form, not merely against the other seed."""
    core, _ = _attached_core(tmp_path, a0)
    jac = np.asarray(core.fill_dense_analytical_jacobian(0.0, [1.5, 2.0]))
    assert jac[0][1] == pytest.approx(J_AB_AT_1_5, rel=1e-12)
    assert jac[0][0] == pytest.approx(-0.5 * 3.0 * 1.5**2, rel=1e-12)
