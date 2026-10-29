"""Issue #1369: run_batch and run_replicates meet the stop conditions run() does.

``add_stop_condition`` was honoured by ``run()`` alone: ``run_batch`` and
``run_replicates`` returned every row to t_end, past the condition, with no
word. Each row is now checked as ``run()`` checks it. A row that meets a
condition comes back cut at the first sample it holds at, with
``Result.stop_condition`` naming it, beside the rows that ran to the end, and
nothing is raised: one row's stop is not the batch's.

A() decays at k from 100, observed as A_tot; the condition is A_tot < 10, met
at t = ln(10)/k, about 23 at k = 0.1 and never by t = 100 at k = 0.001.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = (
    "begin parameters\n 1 k 0.1\nend parameters\n"
    "begin species\n 1 A() 100\nend species\n"
    "begin reactions\n 1 1 0 k\nend reactions\n"
    "begin groups\n 1 A_tot 1\nend groups\n"
)
KW = dict(t_span=(0, 100), n_points=101, seed=1)


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "decay.net"
    path.write_text(NET)
    return str(path)


def _sim(net: str, method: str, condition="A_tot < 10", label="low") -> bngsim.Simulator:
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method=method)
    sim.add_stop_condition(condition, label=label)
    return sim


def _met_at_its_end(result) -> bool:
    """The row holds A_tot >= 10 at every sample but its last."""
    a = np.asarray(result.observables["A_tot"])
    return bool(a[-1] < 10.0 and np.all(a[:-1] >= 10.0))


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_a_batch_row_that_meets_a_condition_is_cut_there(net, method):
    """Before the fix, both rows ran to t = 100 and stop_condition did not exist."""
    rows = _sim(net, method).run_batch(params=[{"k": 0.1}, {"k": 0.001}], **KW)
    met, ran = rows
    assert met.stop_condition == "low"
    assert met.n_times < 101 and _met_at_its_end(met)
    assert ran.stop_condition is None
    assert ran.n_times == 101 and float(ran.time[-1]) == 100.0


def test_an_ode_row_is_cut_where_run_is(net):
    """The row is run()'s StopConditionMet result, to the bit."""
    with pytest.raises(bngsim.StopConditionMet) as raised:
        _sim(net, "ode").run(**KW)
    row = _sim(net, "ode").run_batch(params=[{"k": 0.1}], **KW)[0]
    np.testing.assert_array_equal(row.time, raised.value.result.time)
    np.testing.assert_array_equal(row.species, raised.value.result.species)
    assert raised.value.result.stop_condition == row.stop_condition == "low"


@pytest.mark.parametrize("processors", [None, 2], ids=["sequential", "threads"])
def test_every_replicate_is_cut_where_it_meets_the_condition(net, processors):
    reps = _sim(net, "ssa").run_replicates(3, num_processors=processors, **KW)
    assert [r.stop_condition for r in reps] == ["low", "low", "low"]
    assert all(_met_at_its_end(r) for r in reps)


@pytest.mark.parametrize("call", ["batch", "replicates"])
def test_squeeze_refuses_a_cut_row(net, call):
    sim = _sim(net, "ssa")
    with pytest.raises(ValueError, match=r"Cannot squeeze results a stop condition cut short"):
        if call == "batch":
            sim.run_batch(params=[{"k": 0.1}, {"k": 0.001}], squeeze=True, **KW)
        else:
            sim.run_replicates(2, squeeze=True, **KW)


def test_squeeze_stacks_rows_no_condition_cut(net):
    """Control: where no row meets the condition, the stack is as before."""
    batch = _sim(net, "ode").run_batch(params=[{"k": 0.001}, {"k": 0.002}], squeeze=True, **KW)
    assert batch.species.shape == (2, 101, 1)
    assert batch.stop_condition is None


def test_without_conditions_rows_run_to_the_end(net):
    """Control: nothing registered, nothing cut."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    rows = sim.run_batch(params=[{"k": 0.1}], **KW)
    assert rows[0].n_times == 101 and rows[0].stop_condition is None
    assert sim.run(**KW).stop_condition is None


def test_the_first_condition_added_that_a_row_meets_names_it(net):
    """Conditions are checked in the order they were added; an unlabelled one
    reads as its expression; a callable is called with the row."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    sim.add_stop_condition("A_tot < 50")
    sim.add_stop_condition(lambda r: True, label="always")
    rows = sim.run_batch(params=[{"k": 0.1}, {"k": 0.001}], **KW)
    assert rows[0].stop_condition == "A_tot < 50"
    assert float(rows[0].observables["A_tot"][-1]) < 50.0
    assert rows[1].stop_condition == "always"
