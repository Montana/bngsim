"""A function's steady-state sensitivity is not differenced (issue #1002).

``steady_state(sensitivity_params=...)`` assembles d(func)/dp from the compiled
output-sensitivity chain rule (GH #198), the one a time course uses. Where that
has no derivative for a function (an ``if()``, an ``abs()``, a ``max()``, and
the others its reason names), the row was differenced: each species stepped by
1.5e-8 of the state's largest concentration, each parameter by √eps of itself.
#1100 and #1294 refused the solve where two or three steps disagreed, and a
quotient can be wrong at every step a check takes. The rows are not
differenced now: they are NaN, and ``output_sensitivities`` refuses each by
name, with the reason, as after a run. dY_ss/dp and every function the chain
rule answers are returned as before.

Every expected value is a closed form.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

K1 = 0.21
SQRT_EPS = 1.4901161193847656e-8
# The floored step of a species at or near 0 beside G at 1e8.
H = SQRT_EPS * 1e8

# 0 -> X at s, X -> 0 at k1: X* = s/k1 and dX*/ds = 1/k1. G is made at sg and
# removed at 1, so G* = sg and dG*/ds = 0. The functions are the slot.
NET = """begin parameters
    1 s   {s!r}
    2 k1  0.21
    3 sg  1e8
    4 dg  1.0
    5 q   0.0
end parameters
begin species
    1 X() 0
    2 G() 1e8
end species
begin groups
    1 Xo 1
    2 Go 2
end groups
begin functions
{functions}end functions
begin reactions
    1 0 1 s
    2 1 0 k1
    3 0 2 sg
    4 2 0 dg
end reactions
"""


def _steady(tmp_path, functions, s=1.3, params=("s",)):
    path = tmp_path / "m.net"
    path.write_text(NET.format(s=s, functions=functions))
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode")
    return sim.steady_state(sensitivity_params=list(params))


def _refused(out, name, construct):
    with pytest.raises(ValueError, match=rf"'expression:{name}' has no steady-state") as err:
        out.output_sensitivities([name])
    assert f"uses unsupported construct: {construct}" in str(err.value)
    assert "issue #1002" in str(err.value)


def test_a_secant_the_same_at_both_steps_is_refused(tmp_path):
    """``hx = abs(X)·abs(X)/(K² + X²) + G`` at X* = 0 with K = h/√2, where
    h = 1.49 is the floored step: the secants over h and over h/2 are both
    2/(3h), so the two steps agreed, and main returned dhx/ds = 2/(3h·k1) =
    2.13 at the default settings with no error. The slope at X = 0 is 0 and
    G does not move, so dhx/ds = 0."""
    k = H / math.sqrt(2.0)
    out = _steady(tmp_path, f"    1 hx() abs(Xo)*abs(Xo)/({k!r}^2+Xo^2) + Go\n", s=0.0)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [1 / K1, 0.0], rtol=1e-9)
    assert out.sens_output_source == "none"
    assert np.isnan(out.sensitivities_expressions).all()
    _refused(out, "hx", "abs()")


@pytest.mark.parametrize("frac", [1e-5, 1e-6])
def test_a_kink_within_the_step_is_refused(tmp_path, frac):
    """``kx = if(X > c, X - c, 0) + G`` with c a hair beyond X* = s/k1: the
    slope at X* is 0, so dkx/ds = 0. The floored step crossed the kink, and
    the half step too, so main returned dkx/ds = 4.762 (1/k1) with no error
    where the kink was 1e-5·h or nearer."""
    c = 1.3 / K1 + frac * H
    out = _steady(tmp_path, f"    1 kx() if(Xo > {c!r}, Xo - {c!r}, 0) + Go\n")
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [1 / K1, 0.0], rtol=1e-9)
    _refused(out, "kx", "if() conditional")


def test_a_condition_on_a_parameter_at_0_is_refused(tmp_path):
    """``fq = if(q > 0, 10, 0) + X`` at q = 0 jumps there, and has no
    derivative in q. Its explicit term was differenced at a step of √eps in
    q, across the jump: main returned dfq/dq = 10/√eps = 6.7e8 with no error.
    The corpus has the same: Chitnis2008's ``sigma_v = if(Baseline_dummy_variable
    == 0, 0.33, 0.5)``, a scenario switch at 0, came back 1.1e7."""
    out = _steady(tmp_path, "    1 fq() if(q > 0, 10, 0) + Xo\n", params=("s", "q"))
    np.testing.assert_allclose(np.asarray(out.sensitivity)[0], [1 / K1, 0.0], rtol=1e-9)
    _refused(out, "fq", "if() conditional")


def test_a_compiled_function_beside_a_declined_one_is_returned(tmp_path):
    """``lin = 3·X + G`` has a compiled derivative, d(lin)/ds = 3/k1, and is
    returned, as on main, beside ``kx``, which main returned 4.762 for 0 and is
    refused now; the observable block is the linear projection it was."""
    c = 1.3 / K1 + 1e-5 * H
    out = _steady(
        tmp_path,
        f"    1 kx() if(Xo > {c!r}, Xo - {c!r}, 0) + Go\n    2 lin() 3*Xo + Go\n",
    )
    assert out.sens_output_source == "partial"
    np.testing.assert_allclose(out.output_sensitivities(["lin"]), [[3 / K1]], rtol=1e-9)
    np.testing.assert_allclose(out.output_sensitivities(["observable:Xo"]), [[1 / K1]], rtol=1e-9)
    _refused(out, "kx", "if() conditional")


def test_every_declined_function_is_named_at_once(tmp_path):
    """Two declined functions in one selection are refused in one message,
    each with its own construct; the compiled one between them is not named.
    Both were right on main here, at X* = 6.19, where neither the kink of
    ``abs`` nor that of ``max`` is within a step: what Step 2 refuses that
    main returned right is its cost."""
    out = _steady(
        tmp_path,
        "    1 ab() abs(Xo)\n    2 lin() 3*Xo\n    3 mx() max(Xo, 2)\n",
    )
    with pytest.raises(ValueError) as err:
        out.output_sensitivities(["ab", "lin", "mx"])
    message = str(err.value)
    assert "'expression:ab'" in message and "abs()" in message
    assert "'expression:mx'" in message and "max()" in message
    assert "'expression:lin'" not in message
