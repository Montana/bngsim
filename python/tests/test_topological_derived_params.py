"""Regression test for topological derived-parameter order (issue #568).

A *derived* parameter takes its value from an expression over other parameters
(``k = 2*kbase``), and the engine re-evaluates the whole set whenever anything
underneath one of them moves — at load, on every ``set_param``, and on every
finite-difference sensitivity probe. That re-evaluation used to be a single pass
in *declaration* order, which silently assumes declaration order is dependency
order. Nothing enforced that: ``ModelBuilder.add_parameter`` appends in call
order, and the ``.net`` reader hands rows over in file order.

So a chain declared bottom-last moved exactly one link per pass:

    bb = a*3        declared first — reads `a` BEFORE `a` is re-derived
    a  = base*2     declared second

``set_param("base", 5)`` left ``bb`` at its load-time value, and a second,
value-identical ``set_param`` moved it one link further — a solve rather than a
single relaxation step is what the write needs. Because a derived parameter is
routinely a rate constant, the failure mode is a plausible wrong number: the
model integrates at the pre-write rate and reports success.

This is GH #76's defect one field over (see test_topological_function_eval.py,
which pins the same fix for *functions*), and it has the same fix:
``ModelBuilder`` sorts the derived parameters into dependency order once at
build, and every re-evaluation pass — ``NetworkModel::set_param``, the CVODES
sensitivity RHS syncs, the steady-state FD probe — walks that one order, so one
pass converges a chain of any depth.

These tests pin that with models declared in REVERSE dependency order, against
the identical model declared bottom-up: the two must agree on everything.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

NET = "derived_param_reverse_order.net"


def _build(rows, rate_law="bb"):
    """A one-reaction model whose rate constant is `rate_law`, with `rows`
    ((name, seed, expression), expression "" for a primary) declared in order."""
    b = ModelBuilder()
    for name, seed, expr in rows:
        b.add_parameter(name, seed, expr, bool(expr))
    b.add_species("S", 1.0, False, 1.0)
    b.add_reaction([0], [], "elementary", rate_law, 1.0, True)
    return b.build()


# The issue's own repro: bb = a*3 = (base*2)*3, declared in both orders.
_REVERSED = [("base", 1.0, ""), ("bb", 6.0, "a*3"), ("a", 2.0, "base*2")]
_ORDERED = [("base", 1.0, ""), ("a", 2.0, "base*2"), ("bb", 6.0, "a*3")]


@pytest.mark.parametrize("rows", [_REVERSED, _ORDERED], ids=["reverse", "dependency"])
def test_set_param_propagates_the_whole_chain_in_one_call(rows):
    """One write, fully propagated, whichever order the rows were declared in."""
    core = _build(rows)
    core.set_param("base", 5.0)
    assert core.get_param("a") == 10.0
    assert core.get_param("bb") == 30.0


@pytest.mark.parametrize("rows", [_REVERSED, _ORDERED], ids=["reverse", "dependency"])
def test_repeating_an_identical_write_changes_nothing(rows):
    """A second, value-identical write must be a no-op. It moving the model is
    the signature of an unconverged propagation — one relaxation step per call
    rather than a solve."""
    core = _build(rows)
    core.set_param("base", 5.0)
    once = [core.get_param(n) for n in ("base", "a", "bb")]
    core.set_param("base", 5.0)
    assert [core.get_param(n) for n in ("base", "a", "bb")] == once


def test_load_time_values_are_dependency_ordered():
    """Not only writes: build() evaluates each expression once, so a forward
    reference used to bake the front end's *seed* into the model. With seeds of
    zero the reverse-ordered chain loaded with bb = 0 — a zero rate constant."""
    core = _build([("base", 1.0, ""), ("bb", 0.0, "a*3"), ("a", 0.0, "base*2")])
    assert core.get_param("a") == 2.0
    assert core.get_param("bb") == 6.0


def test_chain_of_five_converges_in_one_write():
    """Depth is not bounded by the number of passes: one write, five links."""
    rows = [(f"p{i}", 0.0, f"p{i - 1}*2") for i in (5, 4, 3, 2, 1)]
    rows.append(("p0", 1.0, ""))
    core = _build(rows, rate_law="p5")
    assert [core.get_param(f"p{i}") for i in range(6)] == [1, 2, 4, 8, 16, 32]
    core.set_param("p0", 3.0)
    assert [core.get_param(f"p{i}") for i in range(6)] == [3, 6, 12, 24, 48, 96]


def test_diamond_dependency_converges():
    """A DAG, not just a chain: d reads b and c, which both read a."""
    core = _build(
        [("d", 0.0, "b+c"), ("b", 0.0, "a*2"), ("c", 0.0, "a*3"), ("a", 1.0, "")],
        rate_law="d",
    )
    assert core.get_param("d") == 5.0
    core.set_param("a", 2.0)
    assert core.get_param("d") == 10.0


def test_reference_cycle_is_refused():
    """A cycle is input no order satisfies, and it is now refused by name
    (issue #617). This test used to assert the opposite — that such a model
    still built — which is what let the value be manufactured from the seeds and
    drift on every later write. See test_parameter_reference_cycle.py for the
    full rule."""
    with pytest.raises(RuntimeError, match=r"reference cycle"):
        _build([("x", 1.0, "y+1"), ("y", 1.0, "x+1"), ("z", 1.0, "")], rate_law="z")


def test_written_derived_parameter_still_overrides_its_dependents():
    """Issue #188's write-side rule is unchanged by the ordering: writing a
    derived parameter overrides its own expression, and the parameters below it
    follow the written value rather than the one the expression would give."""
    core = _build(_REVERSED)
    core.set_param("a", 4.0)
    assert core.get_param("a") == 4.0
    assert core.get_param("bb") == 12.0  # a*3 at the WRITTEN a, not base*2*3


# ─── The .net front end, end to end ──────────────────────────────────────────


def _net_model(data_dir):
    return bngsim.Model.from_net(str(data_dir / NET))


def test_net_chain_integrates_the_closed_form(data_dir):
    """k = k2*2 = (k3*1.0)*2 = ((kbase*3)*1.0)*2 = 0.6, declared top-down in the
    file. The whole chain used to load one link deep, leaving k = 0.0: S never
    decayed and the run reported success."""
    m = _net_model(data_dir)
    assert m.get_param("k") == pytest.approx(0.6)
    r = bngsim.Simulator(m, method="ode").run(
        t_span=(0.0, 4.0), n_points=9, rtol=1e-10, atol=1e-14
    )
    t = np.asarray(r.time)
    s = np.asarray(r.species)[:, list(r.species_names).index("S()")]
    assert s == pytest.approx(10.0 * np.exp(-0.6 * t), rel=1e-6)


def test_net_chain_carries_the_sensitivity_chain_rule(data_dir):
    """The CVODES finite-difference probe re-derives the parameters after every
    perturbation, so the same ordering decides whether ∂S/∂kbase carries through
    the chain. Before the fix this column came back roughly 4x too large and the
    wrong shape entirely (-30/-60/-90 against -22.2/-32.9/-36.6), because the
    perturbation reached only the first link."""
    m = _net_model(data_dir)
    r = bngsim.Simulator(m, method="ode", sensitivity_params=["kbase"]).run(
        t_span=(0.0, 4.0), n_points=9, rtol=1e-10, atol=1e-14
    )
    t = np.asarray(r.time)
    got = np.asarray(r.sensitivities)[:, list(r.species_names).index("S()"), 0]
    # d/dkbase of 10·exp(-6·kbase·t) at kbase = 0.1.
    assert got == pytest.approx(-6.0 * t * 10.0 * np.exp(-0.6 * t), rel=1e-5, abs=1e-6)


def test_net_chain_set_param_moves_the_rate_constant(data_dir):
    """A dose/rate scan over the primary: one write must reach the rate law."""
    m = _net_model(data_dir)
    m.set_param("kbase", 0.2)
    assert m.get_param("k") == pytest.approx(1.2)
    r = bngsim.Simulator(m, method="ode").run(
        t_span=(0.0, 4.0), n_points=9, rtol=1e-10, atol=1e-14
    )
    t = np.asarray(r.time)
    s = np.asarray(r.species)[:, list(r.species_names).index("S()")]
    assert s == pytest.approx(10.0 * np.exp(-1.2 * t), rel=1e-6)
