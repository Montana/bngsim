"""Issue #1362: where run_replicates and run_batch start.

``run()`` starts from the model's state: a ``Model.set_concentration`` made
first is its start. ``run_replicates`` reset every replicate to the seed, so
after ``set_concentration("A()", 500)`` each started at A = 0, with no warning.
It now starts every replicate from the model's state at the call, as ``run()``
and ``parameter_scan`` do. ``run_batch`` starts every point from the initial
conditions by design (``docs/user-guide/simulation.md``), and now warns where
the model is elsewhere.

A birth-death network, 0 -> A at k1 = 20, A -> 0 at k2 = 1, seeded A = 0.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

NET = (
    "begin parameters\n 1 k1 20\n 2 k2 1\nend parameters\n"
    "begin species\n 1 A() 0\nend species\n"
    "begin reactions\n 1 0 1 k1\n 2 1 0 k2\nend reactions\n"
)
KW = dict(t_span=(0, 0.01), n_points=2, seed=1)
BATCH_WARNING = r"run_batch starts every point from the model's initial conditions"


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "bd.net"
    path.write_text(NET)
    return str(path)


def _a0(result) -> float:
    return float(np.asarray(result.species)[0, 0])


def _written(net: str, method: str = "ssa") -> bngsim.Simulator:
    model = bngsim.Model.from_net(net)
    model.set_concentration("A()", 500)
    return bngsim.Simulator(model, method=method)


@pytest.mark.parametrize("processors", [None, 2], ids=["sequential", "threads"])
def test_every_replicate_starts_from_a_written_concentration(net, processors):
    """Before the fix, every replicate started at A = 0."""
    sim = _written(net)
    reps = sim.run_replicates(3, num_processors=processors, **KW)
    assert [_a0(r) for r in reps] == [500.0, 500.0, 500.0]
    np.testing.assert_array_equal(sim.model.get_state(), [500.0])


def test_replicates_after_a_run_start_where_it_ended(net):
    """As a second run() does: the model's state at the call is the start."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ssa")
    sim.run(t_span=(0, 1.0), n_points=2, seed=1)
    ended = float(sim.model.get_state()[0])
    assert ended > 0.0
    reps = sim.run_replicates(2, **KW)
    assert [_a0(r) for r in reps] == [ended, ended]


def test_replicates_after_a_reset_start_from_the_seed(net):
    """Control: reset() first is the way back to the initial conditions."""
    sim = _written(net)
    sim.model.reset()
    assert [_a0(r) for r in sim.run_replicates(2, **KW)] == [0.0, 0.0]


@pytest.mark.parametrize("method", ["ssa", "ode"])
def test_run_batch_warns_that_it_does_not_start_from_a_write(net, method):
    """Its points start from the initial conditions, as documented; it now
    says so where the model is elsewhere. Before, nothing was said."""
    sim = _written(net, method)
    with pytest.warns(UserWarning, match=BATCH_WARNING):
        rows = sim.run_batch(params=[{"k1": 20.0}, {"k2": 1.0}], **KW)
    assert [_a0(r) for r in rows] == [0.0, 0.0]


def test_run_batch_warns_after_a_run(net):
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    sim.run(t_span=(0, 1.0), n_points=2)
    with pytest.warns(UserWarning, match=BATCH_WARNING):
        sim.run_batch(params=[{"k1": 20.0}], **KW)


@pytest.mark.parametrize("how", ["fresh", "saved"])
def test_run_batch_at_its_start_says_nothing(net, how):
    """Control: a fresh model, or one whose write is saved as the state reset
    returns to, starts every point where it is, and no warning is raised."""
    if how == "fresh":
        sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
        start = 0.0
    else:
        sim = _written(net, "ode")
        sim.model.save_concentrations()
        start = 500.0
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=BATCH_WARNING)
        rows = sim.run_batch(params=[{"k1": 20.0}, {"k2": 1.0}], **KW)
    assert [_a0(r) for r in rows] == [start, start]


def test_run_batch_after_a_seed_parameter_write_says_nothing(tmp_path):
    """Control: a parameter that seeds a species moves the model's state and
    its initial conditions together, so the points start where it is and no
    warning is raised."""
    path = tmp_path / "seeded.net"
    path.write_text(NET.replace(" 2 k2 1\n", " 2 k2 1\n 3 A0 7\n").replace("A() 0", "A() A0"))
    model = bngsim.Model.from_net(str(path))
    model.set_param("A0", 50.0)
    sim = bngsim.Simulator(model, method="ode")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=BATCH_WARNING)
        rows = sim.run_batch(params=[{"k1": 1.0}], **KW)
    assert _a0(rows[0]) == 50.0
