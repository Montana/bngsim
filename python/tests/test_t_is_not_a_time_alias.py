"""Issue #659 — `t` is an ordinary model identifier, not a spelling of the clock.

The evaluator binds exactly one clock symbol. `t` is deliberately left free so
a model may name a parameter or observable `t` — the BNGL counter idiom
`Molecules t counter()` is the reason, and `src/expression.cpp` says so at both
the registration site and the file header::

    // time() — pointer set later via set_time_ptr().
    // We do NOT register `t` here so that `t` is free for use as a model
    // identifier (matches BNG2.pl convention).

Four Python layers translated ExprTk expression strings on the belief that
`t()` is an alias for `time()`. It is not: `t()` is how BNG2.pl writes a
reference to a scalar named `t`, and the engine reads it that way
(`strip_empty_parens`). Reading it as the clock erased the observable, which is
issue #28's defect surviving for exactly one name.

Two of the consequences were silent wrong numbers, not just a missed
optimisation:

* the derived Jacobian entry for a rate law reading `t()` came back empty, so
  the **forward sensitivity** of such a model was wrong by a factor of 3.3 on
  the model below — no warning, no error (the codegen sensitivity RHS has no
  self-check);
* the interpreted attach *was* caught, by the C++ finite-difference self-check,
  which declined the whole analytical Jacobian — so that model silently ran on
  finite differences for a reason that was a preprocessing bug, not a real
  non-differentiability.

The JAX translator carried the inverse of the same confusion: it rewrote both
`time()` and `t()` to a bare `t`, which the observable pass then rewrote, so in
a model with an observable named `t` the **clock** came out as that observable's
population.

`time()` remains the only way to read the clock everywhere. One grammar is
exempt and must stay that way: the *index name* of a table function, where
`is_time_index()` accepts `time`, `T`, `Time()` and `t()` alike — a different
namespace from an expression token. `test_table_function_index_t_still_means_time`
pins that down so a later cleanup does not "fix" it.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from bngsim import Model, Simulator
from bngsim._bngsim_core import ModelBuilder
from bngsim._jacobian import _preprocess_exprtk, differentiate_rate_law
from bngsim._jax_rhs import _translate_expr_jax

# An observable named `t`, and a rate law that references it the way BNG2.pl
# emits a scalar — as a zero-argument call. `law = k*t()` means `k*A`, so the
# reaction rate is `k*A*A` and every derivative below is non-trivial in A.
NET = """begin parameters
    1 k 0.1  # Constant
end parameters
begin functions
    1 law() k*t()
end functions
begin species
    1 A() 10.0
    2 B() 0.0
end species
begin reactions
    1 1 2 law #_R1
end reactions
begin groups
    1 t                    1
    2 Btot                 2
end groups
"""


@pytest.fixture
def t_obs_net(tmp_path):
    p = tmp_path / "t_obs.net"
    p.write_text(NET)
    return str(p)


def _model(path):
    with contextlib.redirect_stderr(io.StringIO()):
        return Model.from_net(path)


# ─── The rewrite itself ──────────────────────────────────────────────────────


def test_preprocess_leaves_t_call_alone_and_takes_time_call():
    """`time()` becomes the clock placeholder; `t()` becomes the scalar `t`."""
    assert _preprocess_exprtk("k*time()") == "k*_bngsim_time_csymbol"
    assert _preprocess_exprtk("k*t()") == "k*t"
    # The zero-arg strip that handles `t()` is issue #28's, unchanged for
    # every other name.
    assert _preprocess_exprtk("k*Atot()") == "k*Atot"


def test_partial_wrt_an_observable_named_t():
    """d(k*t())/dt is k, not absent.

    Returned empty before the fix — the same silent collapse issue #28
    documents for `divide()`, surviving for the single name `t`.
    """
    assert differentiate_rate_law("k*t()", {}, {"t"}, {"k"}) == differentiate_rate_law(
        "k*t", {}, {"t"}, {"k"}
    )
    assert set(differentiate_rate_law("k*t()", {}, {"t"}, {"k"})) == {"t"}
    # The clock really is constant w.r.t. every observable — the property the
    # old rewrite was reaching for, still true for the spelling that means it.
    assert differentiate_rate_law("k*time()", {}, {"t"}, {"k"}) == {}


# ─── What it cost: the analytical Jacobian ───────────────────────────────────


def test_analytical_jacobian_attaches_and_matches_finite_differences(t_obs_net):
    """The model keeps its analytical Jacobian, and the entries are right.

    Before the fix the C++ FD self-check declined the attach outright
    ("analytical -1, finite-difference -2" at the seed state), so this model
    ran on finite differences.
    """
    model = _model(t_obs_net)
    with contextlib.redirect_stderr(io.StringIO()):
        attached = model.prepare_analytical_jacobian()
    assert attached, f"attach declined: {model.analytical_jacobian_status}"
    assert model.analytical_jacobian_status == "complete"

    y = np.array([4.0, 6.0])
    with contextlib.redirect_stderr(io.StringIO()):
        analytic = np.asarray(model.jacobian(y, 1.0))
        f0 = np.asarray(model.rhs(y, 1.0))
        numeric = np.empty((2, 2))
        for j in range(2):
            yy = y.copy()
            h = 1e-6 * max(1.0, abs(y[j]))
            yy[j] += h
            numeric[:, j] = (np.asarray(model.rhs(yy, 1.0)) - f0) / h
    np.testing.assert_allclose(analytic, numeric, atol=1e-4)


# ─── What it cost: forward sensitivities ─────────────────────────────────────


def test_forward_sensitivity_matches_finite_differences(t_obs_net):
    """The load-bearing case: wrong numbers, no error, before the fix.

    d[A]/dk at t=2 is -22.22 by central differences on the engine's own solve.
    The codegen sensitivity RHS reported -6.67 — 3.3x off — because the
    Jacobian term for the observable `t` was missing. Nothing checks that RHS
    against finite differences, so it was reported as an ordinary result.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        result = Simulator(_model(t_obs_net), method="ode").compute_all_sensitivities(
            t_span=(0.0, 2.0), n_points=3, params=["k"]
        )
        sens = np.asarray(result.sensitivities)[:, :, 0]

        k0, rel = 0.1, 1e-5
        legs = []
        for k in (k0 * (1 - rel), k0 * (1 + rel)):
            m = _model(t_obs_net)
            m.set_param("k", k)
            legs.append(
                np.asarray(Simulator(m, method="ode").run(t_span=(0.0, 2.0), n_points=3).species)
            )
        fd = (legs[1] - legs[0]) / (k0 * 2 * rel)

    np.testing.assert_allclose(sens, fd, rtol=5e-3, atol=1e-6)


def test_codegen_trajectory_still_agrees_with_the_interpreter(t_obs_net):
    """The RHS path was already right (a model name outranks the built-in
    table), and stays right — the fix must not move it."""
    with contextlib.redirect_stderr(io.StringIO()):
        runs = [
            np.asarray(
                Simulator(_model(t_obs_net), method="ode", codegen=cg)
                .run(t_span=(0.0, 2.0), n_points=3)
                .species
            )
            for cg in (False, True)
        ]
    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-9)


# ─── The JAX translator ──────────────────────────────────────────────────────


def test_jax_translator_keeps_the_clock_and_the_observable_apart():
    """`time()` is the RHS's time argument; `t()` and `t` are the observable.

    Before the fix `time()` translated to `obs[3]` in this model — the clock
    replaced by a population, silently.
    """
    tr = lambda e: _translate_expr_jax(e, {"k": 0}, {"t": 3, "Atot": 1}, set(), [])  # noqa: E731
    assert tr("k*time()") == "params[0]*t"
    assert tr("k*t()") == "params[0]*obs[3]"
    assert tr("k*t") == "params[0]*obs[3]"
    # Both in one expression, each resolved on its own terms.
    assert tr("k*time()*t") == "params[0]*t*obs[3]"
    # Issue #28 in this path, which the clock fix required: a scalar written as
    # a call was emitting `obs[1]()`, a call on a JAX array.
    assert tr("k*Atot()") == "params[0]*obs[1]"


# ─── The one place `t` IS the clock ──────────────────────────────────────────


def test_table_function_index_t_still_means_time():
    """A tfun's *index name* is a different grammar: `t()` selects time there.

    `is_time_index()` (include/bngsim/table_function.hpp) accepts `time`, `T`,
    `Time()` and `t()`, case-insensitively and tolerating a trailing `()`. That
    is correct and documented; a cleanup that rewrote every `t()` in the tree
    would break it, so it is pinned here.
    """
    for index in ("time", "t", "Time", "T", "t()"):
        b = ModelBuilder()
        a = b.add_species("A", 5.0)
        b.add_species("B", 0.0)
        b.add_function("kf", "kf")
        b.add_inline_table_function_spec("kf", [0.0, 5.0, 10.0], [1.0, 2.0, 3.0], index, "linear")
        b.add_reaction([a], [], "functional", "kf")
        assert b.build().functions_use_time, index


# ─── The spellings the engine refuses ────────────────────────────────────────


def test_a_model_with_no_scalar_named_t_refuses_both_spellings():
    """`t` does not fall back to the clock: with nothing named `t`, it is an
    unknown identifier and the model does not build. This is what the docs used
    to promise worked."""
    for expr in ("0.1*t()", "0.1*t"):
        b = ModelBuilder()
        a = b.add_species("A", 5.0)
        b.add_species("B", 0.0)
        b.add_function("kf", expr)
        b.add_reaction([a], [], "functional", "kf")
        with pytest.raises(Exception, match="compil"):
            b.build()
    # `time()` in the same position builds.
    b = ModelBuilder()
    a = b.add_species("A", 5.0)
    b.add_species("B", 0.0)
    b.add_function("kf", "0.1*time()")
    b.add_reaction([a], [], "functional", "kf")
    assert b.build().functions_use_time
