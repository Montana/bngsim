"""Issue #772: ``Model.add_table_function`` under a name the model already has.

A second table under a ``.net`` tfun's name was accepted and appended: the
evaluator kept the first ``tfun_<name>`` (ExprTk refused the second registration
and nothing checked), so the dynamics read the old data, ``table_function_names``
listed the name twice, and codegen, resolving names last-wins, reported the new
table beside dynamics that used the old one. Under a plain function's name the
interpreter kept the function's body while compiled code read the table, so the
same model ran two ways by codegen. Both are refused now, before the model
changes, and every engine runs the model it had.

``A' = drive`` from ``A(0) = 0`` with ``drive`` constant on ``[0, 10]``, so
``A(1)`` is the value of ``drive`` the dynamics used.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ModelError

_NET = """begin parameters
    1 k 1
    2 a 0.5
end parameters
begin functions
{functions}
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 drive
end reactions
begin groups
    1 Aobs 1
end groups
"""

_TFUN = '    1 drive() tfun([0,10],[1,1], time, method=>"linear")'
_PLAIN = "    1 drive() k\n    2 g() drive()*a"


def _net(tmp_path: Path, functions: str) -> str:
    path = tmp_path / "m.net"
    path.write_text(_NET.format(functions=functions))
    return str(path)


def _run(model: bngsim.Model, codegen: bool) -> tuple[float, list[float]]:
    r = bngsim.Simulator(model, method="ode", codegen=codegen).run(t_span=(0, 1), n_points=3)
    drive = np.asarray(r.expressions)[:, list(r.expression_names).index("drive")]
    return float(np.asarray(r.species)[-1, 0]), drive.tolist()


@pytest.mark.parametrize("functions", [_TFUN, _PLAIN], ids=["tfun", "plain-function"])
@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
def test_a_table_under_a_functions_name_is_refused(tmp_path, functions, codegen) -> None:
    model = bngsim.Model.from_net(_net(tmp_path, functions))
    tables = model.table_function_names
    with pytest.raises(ModelError, match="already has a function named 'drive'"):
        model.add_table_function("drive", times=[0.0, 10.0], values=[7.0, 7.0])
    assert model.table_function_names == tables
    # Both engines run the model's own drive = 1: no table data reached either.
    assert _run(model, codegen) == (pytest.approx(1.0), [1.0, 1.0, 1.0])


def test_a_second_table_under_one_name_is_refused(tmp_path) -> None:
    model = bngsim.Model.from_net(_net(tmp_path, _TFUN))
    model.add_table_function("other", times=[0.0, 10.0], values=[5.0, 5.0])
    with pytest.raises(ModelError, match="table function 'other' already exists"):
        model.add_table_function("other", times=[0.0, 10.0], values=[7.0, 7.0])
    assert model.table_function_names == ["drive", "other"]
    assert model.clone().table_function_names == ["drive", "other"]


def test_a_table_added_after_a_run_under_a_functions_name_is_refused(tmp_path) -> None:
    # The face seen in the review of #1020: a Simulator made before the call kept
    # the output-sensitivity map and compiled artifact it had, so the same model
    # and table gave X(2) = 2 or 20 by the order of two calls.
    model = bngsim.Model.from_net(_net(tmp_path, _PLAIN))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    sim.run(t_span=(0, 1), n_points=3)
    with pytest.raises(ModelError, match="already has a function named 'drive'"):
        model.add_table_function("drive", times=[0.0, 10.0], values=[0.0, 100.0])
    assert model.table_function_names == []


def test_a_name_the_evaluator_refuses_is_reported(tmp_path) -> None:
    # ExprTk's symbol table refuses `tfun_a b`; the refusal used to be dropped and
    # the table appended under a name no expression could call.
    model = bngsim.Model.from_net(_net(tmp_path, _TFUN))
    with pytest.raises(ModelError, match="failed to register function 'tfun_a b'"):
        model.add_table_function("a b", times=[0.0, 10.0], values=[7.0, 7.0])
    assert model.table_function_names == ["drive"]


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
def test_a_table_under_a_new_name_is_still_added(tmp_path, codegen) -> None:
    model = bngsim.Model.from_net(_net(tmp_path, _TFUN))
    model.add_table_function("other", times=[0.0, 10.0], values=[5.0, 5.0])
    assert model.table_function_names == ["drive", "other"]
    assert _run(model, codegen) == (pytest.approx(1.0), [1.0, 1.0, 1.0])
