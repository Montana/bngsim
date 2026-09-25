"""`time()` is the clock even when the model declares a scalar named `time` (issue #776).

BNG2.pl rejects a *parameter* named `time` but accepts an *observable* named
`time`, and a hand-written .net can declare either. The engine registers such a
scalar under the mangled key `r_time`, and then two preprocessing steps turned
the clock into it:

* the interpreter's ``strip_empty_parens`` rewrote ``time()`` to a bare
  ``time`` because ``time`` was now a registered scalar name, and the bare word
  then remapped to ``r_time``; the ambiguity guard that should have caught it
  runs after the parens are gone;
* both codegen identifier tables (one since #803 retired the ``.net``
  emitter's) let a model name override the built-in ``time`` entry, so
  ``time()`` compiled to the scalar's slot.

The sensitivity layer (sympy) always read ``time()`` as the clock, so
sensitivities were computed for a model the trajectory was not: on the model
below the trajectory said B(2) = 10 while dB/dk said 2.

The rule now, in every path: the call form ``time()`` is the clock; the bare
word ``time`` is the model's scalar. The model is 0 -> B at rate ``k*time()``,
so B(t) = k t^2 / 2 whatever the scalar holds.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from bngsim import Model, Simulator
from bngsim._codegen import (
    _build_ident_lookup_model,
    _translate_expr_to_c,
)

T_OUT = np.array([0.0, 1.0, 2.0])
CLOCK_B = T_OUT**2 / 2  # k = 1

_BASE = """
begin parameters
  1 k 1.0
PARAMS
end parameters
begin species
  1 B() 0
  2 T() 5
end species
begin functions
  1 f() BODY
end functions
begin reactions
  1 0 1 f
end reactions
begin groups
  1 Btot 1
GROUPS
end groups
"""

#: Ways a model can own the name `time`, each holding the value 5.
SCALARS = {
    "observable": ("", "  2 time 2"),
    "parameter": ("  2 time 5", ""),
}


def _net(tmp_path: Path, scalar: str, body: str = "k*time()") -> str:
    params, groups = SCALARS[scalar]
    text = _BASE.replace("PARAMS\n", params + "\n" if params else "")
    text = text.replace("GROUPS\n", groups + "\n" if groups else "")
    text = text.replace("BODY", body)
    path = tmp_path / f"time_{scalar}_{abs(hash(body))}.net"
    path.write_text(textwrap.dedent(text).strip() + "\n")
    return str(path)


def _ode(net: str, codegen: bool) -> np.ndarray:
    res = Simulator(Model.from_net(net), method="ode", codegen=codegen).run(
        t_span=(0, 2), n_points=3
    )
    return np.asarray(res.observables["Btot"])


@pytest.mark.parametrize("scalar", sorted(SCALARS))
@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
def test_time_call_is_the_clock(tmp_path, scalar, codegen):
    """B = t^2/2 = [0, 0.5, 2]. Before the fix: [0, 5, 10] (time() read as 5)."""
    np.testing.assert_allclose(_ode(_net(tmp_path, scalar), codegen), CLOCK_B, atol=1e-6)


@pytest.mark.parametrize("scalar", sorted(SCALARS))
@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
def test_bare_time_is_the_scalar(tmp_path, scalar, codegen):
    """The other half of the rule: bare `time` still reads the model's scalar, so
    a rate `k*time` is the constant 5 and B = 5t."""
    got = _ode(_net(tmp_path, scalar, body="k*time"), codegen)
    np.testing.assert_allclose(got, 5.0 * T_OUT, atol=1e-6)


@pytest.mark.parametrize("scalar", sorted(SCALARS))
@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
def test_both_in_one_expression(tmp_path, scalar, codegen):
    """`time()*time` is clock times scalar: rate 5t, so B = 5t^2/2."""
    got = _ode(_net(tmp_path, scalar, body="k*time()*time"), codegen)
    np.testing.assert_allclose(got, 5.0 * CLOCK_B, atol=1e-6)


def _ssa_finals(net: str, seeds: range) -> list[float]:
    model = Model.from_net(net)
    return [
        float(
            Simulator(model, method="ssa")
            .run(t_span=(0, 2), n_points=3, seed=s)
            .observables["Btot"][-1]
        )
        for s in seeds
    ]


@pytest.mark.parametrize("scalar", sorted(SCALARS))
def test_ssa_rate_follows_the_clock(tmp_path, scalar):
    """SSA shares the interpreter's evaluator. The control is the same model with
    the scalar renamed, where `time()` can only be the clock: seed for seed, the
    two must produce the same trajectory. Before the fix the `time` model read
    the scalar (5) and fired about five times as often."""
    seeds = range(40)
    control = Path(_net(tmp_path, scalar)).read_text()
    control = control.replace("  2 time ", "  2 tscalar ")
    control_net = tmp_path / f"control_{scalar}.net"
    control_net.write_text(control)
    assert "time 5" not in control and "  2 time 2" not in control

    assert _ssa_finals(_net(tmp_path, scalar), seeds) == _ssa_finals(str(control_net), seeds)


@pytest.mark.parametrize("scalar", sorted(SCALARS))
def test_sensitivity_agrees_with_its_trajectory(tmp_path, scalar):
    """dB/dk = t^2/2, and a central difference of the trajectory must agree.
    Before the fix the sensitivity was t^2/2 but the difference quotient was 5t:
    the two layers disagreed on what `time()` meant."""
    net = _net(tmp_path, scalar)
    res = Simulator(Model.from_net(net), method="ode", sensitivity_params=["k"]).run(
        t_span=(0, 2), n_points=3
    )
    sens = np.asarray(res.sensitivities).reshape(3, -1)[:, 0]
    np.testing.assert_allclose(sens, CLOCK_B, atol=1e-6)

    h = 1e-4

    def b_at(k: float) -> np.ndarray:
        m = Model.from_net(net)
        m.set_param("k", k)
        res = Simulator(m, method="ode").run(t_span=(0, 2), n_points=3)
        return np.asarray(res.observables["Btot"])

    fd = (b_at(1 + h) - b_at(1 - h)) / (2 * h)
    np.testing.assert_allclose(sens, fd, atol=1e-5)


def test_model_codegen_translation_keeps_the_clock():
    """The C, independent of a compiler: with an observable `time` in the table,
    `time()` is the clock `t` and bare `time` is the observable's slot. The
    translator (``_translate_expr_to_c``, every model's since #803) needed the fix
    as much as the retired ``.net`` one; before it, it compiled all three `time`
    tokens below to the observable's slot."""
    lookup = _build_ident_lookup_model({"k": "p[0]"}, {}, {"time": "obs[0]"}, {})
    assert _translate_expr_to_c("k*time()", lookup) == "p[0]*t"
    assert _translate_expr_to_c("k*time", lookup) == "p[0]*obs[0]"
    assert _translate_expr_to_c("k*time( )*time", lookup) == "p[0]*t*obs[0]"


def test_no_scalar_named_time_is_unchanged(tmp_path):
    """The common case — no model scalar `time` — reads the clock as it always did."""
    text = Path(_net(tmp_path, "parameter")).read_text().replace("  2 time 5\n", "")
    assert "time 5" not in text
    net = tmp_path / "plain.net"
    net.write_text(text)
    for codegen in (False, True):
        np.testing.assert_allclose(_ode(str(net), codegen), CLOCK_B, atol=1e-6)
