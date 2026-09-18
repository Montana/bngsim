"""Regression test for the unsatisfiable-parameter refusal (issue #617).

A derived parameter defined through a reference cycle — `x = y+1` with
`y = x+1` — denotes no value, and one defined in terms of itself (`s = s*2`)
denotes none either. bngsim used to build both.

What it produced was worse than an error. `build()` has no order to evaluate a
cycle in, so the fallback order ran and the parameters came out of the front
end's *seeds*: the same model loaded `x = 2.0`, `1.0` or `101.0` on three
different seed pairs. Each later re-derivation relaxed the cycle one more step,
so writing an **unrelated** parameter moved `x` by a fixed amount every time —
the model was a function of how many writes had happened, not of its parameters.
And it reached the solver: with `x` as a rate constant the run completed and
returned an ordinary-looking trajectory. A self-reference was quieter still,
being demoted to a plain `Constant` holding `seed*2` with the expression
dropped.

This is the residue issue #568 left. That fix sorts the derived parameters
topologically so an acyclic chain converges in one pass; a cycle has no
topological order, so the drift #568 removed everywhere else survived there
alone. Both reference implementations refuse these models by name — BNG2.pl with
"ABORT: Parameter y has a dependency cycle y->x->y" and "ABORT: Parameter s is
defined recursively", run_network with "Could not find parameter y. Exiting." —
so refusing aligns bngsim with both rather than trading one disagreement for
another.

Note the deliberate asymmetry with *functions*, pinned at the bottom of this
file: a cyclic function graph still builds, because a real vendored model
(MODEL1006230117) has mutually recursive assignment rules that are linear in
their two unknowns and therefore a solvable simultaneous system, not nonsense.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim._bngsim_core import ModelBuilder


def _build(rows, rate_law="z"):
    """A one-reaction model whose rate constant is `rate_law`, with `rows`
    ((name, seed, expression), expression "" for a primary) declared in order."""
    b = ModelBuilder()
    for name, seed, expr in rows:
        b.add_parameter(name, seed, expr, bool(expr))
    b.add_species("S", 1.0, False, 1.0)
    b.add_reaction([0], [], "elementary", rate_law, 1.0, True)
    return b.build()


# ─── The refusal ─────────────────────────────────────────────────────────────


def test_two_parameter_cycle_is_refused():
    with pytest.raises(RuntimeError) as exc:
        _build([("x", 1.0, "y+1"), ("y", 1.0, "x+1"), ("z", 1.0, "")])
    assert "reference cycle" in str(exc.value)


def test_the_message_names_the_cycle_in_reading_order():
    """Naming the parameters is not enough to act on — the message has to say
    which reads which, the way BNG2.pl's `y->x->y` does."""
    with pytest.raises(RuntimeError) as exc:
        # a reads c, c reads b, b reads a.
        _build([("a", 1.0, "c+1"), ("b", 1.0, "a+1"), ("c", 1.0, "b+1"), ("z", 1.0, "")])
    assert "a -> c -> b -> a" in str(exc.value)


def test_self_referential_parameter_is_refused():
    """`s = s*2` used to be demoted to a Constant holding seed*2, with the
    expression dropped — no error, and nothing left to show the number came from
    a definition that defines nothing."""
    with pytest.raises(RuntimeError) as exc:
        _build([("s", 2.0, "s*2"), ("z", 1.0, "")])
    assert "defined in terms of itself" in str(exc.value)
    assert "s = s*2" in str(exc.value)


def test_a_solvable_cycle_is_refused_too():
    """`a = 10-b` with `b = a` HAS a solution (a = 5). It is still refused:
    finding it needs a simultaneous solve, which the single evaluation pass is
    not, and BNG2.pl refuses every parameter dependency cycle regardless of
    whether one exists. Pinned so that 'it has a solution' is never mistaken for
    'the engine will find it'."""
    with pytest.raises(RuntimeError, match=r"reference cycle"):
        _build([("a", 1.0, "10-b"), ("b", 1.0, "a"), ("z", 1.0, "")])


def test_the_message_names_the_cycle_not_a_parameter_downstream_of_it():
    """Kahn leaves everything downstream of a cycle unplaced too, so the naive
    report would blame `d` — which is perfectly well defined and would send the
    reader to the wrong row."""
    with pytest.raises(RuntimeError) as exc:
        _build([("d", 1.0, "x*2"), ("x", 1.0, "y+1"), ("y", 1.0, "x+1"), ("z", 1.0, "")])
    msg = str(exc.value)
    assert "x -> y -> x" in msg
    assert "'d'" not in msg


def test_a_cyclic_net_file_is_refused_at_load(tmp_path: Path):
    """The refusal has to reach the front end a user actually loads."""
    net = tmp_path / "cyc.net"
    net.write_text(
        "begin parameters\n"
        "    1 x  y+1\n"
        "    2 y  x+1\n"
        "end parameters\n"
        "begin species\n"
        "    1 S() 1.0\n"
        "end species\n"
        "begin reactions\n"
        "    1 1 0 x\n"
        "end reactions\n"
        "begin groups\n"
        "    1 S_tot 1\n"
        "end groups\n"
    )
    with pytest.raises(Exception, match=r"reference cycle"):
        bngsim.Model.from_net(str(net))


# ─── What must still build ───────────────────────────────────────────────────


def test_an_acyclic_chain_out_of_declaration_order_still_builds():
    """The #568 guarantee is untouched: a forward reference is not a cycle. It
    resolves at load and one write still propagates the whole chain."""
    core = _build([("bb", 6.0, "a*3"), ("a", 2.0, "base*2"), ("base", 1.0, ""), ("z", 1.0, "")])
    assert core.get_param("a") == 2.0  # base*2
    assert core.get_param("bb") == 6.0  # a*3
    core.set_param("base", 5.0)
    assert core.get_param("bb") == 30.0


def test_a_diamond_still_builds():
    """Two paths to the same parameter is a DAG, not a cycle — the check must
    key on reachability, not on a parameter being referenced twice."""
    core = _build(
        [("d", 0.0, "b+c"), ("b", 0.0, "a*2"), ("c", 0.0, "a*3"), ("a", 1.0, ""), ("z", 1.0, "")]
    )
    assert core.get_param("d") == 5.0


def test_a_referenceless_expression_still_builds():
    """`gamma = 1/7` names nothing, so it cannot be in a cycle; it stays the
    issue #227 demotion to a plain constant."""
    core = _build([("g", 0.0, "1/7"), ("z", 1.0, "")])
    assert core.get_param("g") == pytest.approx(1.0 / 7.0)


def test_a_function_cycle_still_builds():
    """The deliberate asymmetry (see the module docstring). A cyclic *function*
    graph keeps GH #76's fallback order, because the one real corpus instance is
    a solvable simultaneous system rather than nonsense, and refusing it would
    drop a model whose equations mean something."""
    b = ModelBuilder()
    b.add_parameter("k", 1.0, "", False)
    b.add_species("S", 1.0, False, 1.0)
    b.add_function("f", "g+1")
    b.add_function("g", "f+1")
    b.add_reaction([0], [], "functional", "f", 1.0, True)
    assert b.build().n_species == 1


def test_the_vendored_model_with_mutually_recursive_rules_still_loads():
    """MODEL1006230117 is the model that decided the function-side scope:
    `R = R_Total-LR-LRG-RG` and `LRG = (L_iso*R*Gs)/(K_H*K_C)` read each other.
    If a future change refuses function cycles, this is what it costs."""
    doc = Path("benchmarks/suites/biomodels/data/sbml_downloads/MODEL1006230117.xml")
    if not doc.is_file():
        pytest.skip("vendored BioModels SBML corpus not present")
    assert bngsim.Model.from_sbml(str(doc)).n_species > 0
