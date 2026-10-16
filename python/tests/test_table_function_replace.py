"""``Model.add_table_function`` under a table's name replaces its data.

Since issue #772 a name the model already has as a function is refused, so a
table under a new name was the only one accepted, and nothing in a loaded model
can call a table it did not declare: every call that succeeded added a table
the run never read. Under the name of a table the model has, a ``.net`` tfun's
among them, the call now puts the new data in that table, in place. The
interpreter, the compiled code (which reads the table through the model when
it runs), the SSA and the steady-state solvers all read it, and so does a
``Simulator`` made before the call. What the table is read over is compiled
into the code, so a replacement keeps it.

A table under a new name is still added. Nothing reads it, so it is reported
with a warning, and it no longer marks the model as reading the time, which
refused ``steady_state()``, or puts its breakpoints among the run's stops.

``A' = drive`` from ``A(0) = 0``, so ``A(t)`` is the integral of the table the
dynamics read.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest
from bngsim import ModelError

_NET = """begin parameters
    1 k 1
    2 a 0.5
    3 kd 1
end parameters
begin functions
{functions}
end functions
begin species
    1 A() 0
end species
begin reactions
{reactions}
end reactions
begin groups
    1 Aobs 1
end groups
"""

_GROWS = "    1 0 1 drive"
_SETTLES = "    1 0 1 drive\n    2 1 0 kd"

_TFUN = '    1 drive() tfun([0,10],[1,1], time, method=>"linear")'
_TFUN_FILE = "    1 drive() tfun('drive.tfun', time)"
_TFUN_OVER_K = '    1 drive() tfun([0,10],[1,1], k, method=>"linear")'
_EMBEDDED = '    1 drive() 2*tfun([0,10],[1,1], time, method=>"linear")'
_EMBEDDED_FILE = "    1 drive() 2*tfun('drive.tfun', time)"
_PLAIN = "    1 drive() k"

SEVEN = dict(times=[0.0, 10.0], values=[7.0, 7.0])


def _model(tmp_path: Path, functions: str, reactions: str = _GROWS) -> bngsim.Model:
    (tmp_path / "drive.tfun").write_text("# time drive\n0 1\n10 1\n")
    path = tmp_path / "m.net"
    path.write_text(_NET.format(functions=functions, reactions=reactions))
    return bngsim.Model.from_net(str(path))


def _run(model, codegen=False, t_end=1.0, n_points=3, method="ode", **kw):
    model.reset()
    r = bngsim.Simulator(model, method=method, codegen=codegen).run(
        t_span=(0.0, t_end), n_points=n_points, **kw
    )
    drive = np.asarray(r.expressions)[:, list(r.expression_names).index("drive")]
    return float(np.asarray(r.species)[-1, 0]), drive.tolist()


def _quiet_add(model, name, **kw):
    """add_table_function, failing the test on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.add_table_function(name, **kw)


ENGINES = pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])


# ─── A table the model has takes the new data ──────────────────────────────


@ENGINES
@pytest.mark.parametrize("functions", [_TFUN, _TFUN_FILE], ids=["inline", "file"])
def test_a_tfun_given_new_data_runs_it(tmp_path, functions, codegen) -> None:
    model = _model(tmp_path, functions)
    assert _run(model, codegen) == (pytest.approx(1.0), [1.0, 1.0, 1.0])
    _quiet_add(model, "drive", **SEVEN)
    assert model.table_function_names == ["drive"]
    assert _run(model, codegen) == (pytest.approx(7.0), [7.0, 7.0, 7.0])


@ENGINES
def test_new_data_from_a_file(tmp_path, codegen) -> None:
    model = _model(tmp_path, _TFUN_FILE)
    (tmp_path / "seven.tfun").write_text("# time drive\n0 7\n10 7\n")
    _quiet_add(model, "drive", file=tmp_path / "seven.tfun")
    assert _run(model, codegen) == (pytest.approx(7.0), [7.0, 7.0, 7.0])


@ENGINES
@pytest.mark.parametrize("functions", [_EMBEDDED, _EMBEDDED_FILE], ids=["inline", "file"])
def test_a_table_inside_a_function_takes_new_data(tmp_path, functions, codegen) -> None:
    # `2*tfun(...)` registers its table as `drive__tfun0`; a file of the
    # model's own form, its value column headed `drive`, is accepted for it.
    model = _model(tmp_path, functions)
    assert model.table_function_names == ["drive__tfun0"]
    assert _run(model, codegen) == (pytest.approx(2.0), [2.0, 2.0, 2.0])
    (tmp_path / "seven.tfun").write_text("# time drive\n0 7\n10 7\n")
    _quiet_add(model, "drive__tfun0", file=tmp_path / "seven.tfun")
    assert _run(model, codegen) == (pytest.approx(14.0), [14.0, 14.0, 14.0])
    _quiet_add(model, "drive__tfun0", times=[0.0, 10.0], values=[3.0, 3.0])
    assert _run(model, codegen) == (pytest.approx(6.0), [6.0, 6.0, 6.0])


@ENGINES
def test_a_simulator_made_before_the_new_data_runs_it(tmp_path, codegen) -> None:
    model = _model(tmp_path, _TFUN)
    sim = bngsim.Simulator(model, method="ode", codegen=codegen)
    assert float(np.asarray(sim.run(t_span=(0.0, 1.0), n_points=3).species)[-1, 0]) == (
        pytest.approx(1.0)
    )
    _quiet_add(model, "drive", **SEVEN)
    model.reset()
    assert float(np.asarray(sim.run(t_span=(0.0, 1.0), n_points=3).species)[-1, 0]) == (
        pytest.approx(7.0)
    )


@ENGINES
def test_the_method_can_change(tmp_path, codegen) -> None:
    # 1 on [0, 0.5) and 3 after it as a step, A(1) = 2; the same knots read
    # linearly, A(1) = 2.5.
    model = _model(tmp_path, _TFUN)
    knots = dict(times=[0.0, 0.5, 10.0], values=[1.0, 3.0, 3.0])
    _quiet_add(model, "drive", method="step", **knots)
    assert _run(model, codegen)[0] == pytest.approx(2.0, rel=1e-6)
    _quiet_add(model, "drive", method="linear", **knots)
    assert _run(model, codegen)[0] == pytest.approx(2.5, rel=1e-6)


@ENGINES
def test_the_runs_stops_are_the_new_tables(tmp_path, codegen) -> None:
    # The model's table is flat, so it breaks nowhere. The new one is a pulse
    # of 10 on [5, 5.5): A(10) = 5 only if the run stops on its edges, which
    # an integrator stepping across a flat stretch does not otherwise see.
    model = _model(tmp_path, _TFUN)
    assert model.table_function_breaks() == []
    _quiet_add(
        model, "drive", times=[0.0, 5.0, 5.5, 10.0], values=[0.0, 10.0, 0.0, 0.0], method="step"
    )
    assert len(model.table_function_breaks()) == 1
    assert _run(model, codegen, t_end=10.0, n_points=2)[0] == pytest.approx(5.0, rel=1e-6)


def test_the_ssa_reads_the_new_data(tmp_path) -> None:
    # A Poisson count of mean 10 * drive by t = 10: 10 against 70.
    model = _model(tmp_path, _TFUN)
    _quiet_add(model, "drive", **SEVEN)
    counts = [_run(model, method="ssa", t_end=10.0, n_points=2, seed=s)[0] for s in range(5)]
    assert 50.0 < np.mean(counts) < 90.0


@ENGINES
def test_the_steady_state_reads_the_new_data(tmp_path, codegen) -> None:
    # Over the parameter k the table does not read the time, so the model has
    # a steady state: A* = drive / kd.
    model = _model(tmp_path, _TFUN_OVER_K, _SETTLES)
    _quiet_add(model, "drive", index="k", **SEVEN)
    ss = bngsim.Simulator(model, method="ode", codegen=codegen).steady_state()
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(7.0, rel=1e-6)


def test_a_clone_has_the_data_it_was_cloned_with(tmp_path) -> None:
    model = _model(tmp_path, _TFUN)
    before = model.clone()
    _quiet_add(model, "drive", **SEVEN)
    after = model.clone()
    _quiet_add(after, "drive", times=[0.0, 10.0], values=[3.0, 3.0])
    assert [_run(m)[0] for m in (model, before, after)] == pytest.approx([7.0, 1.0, 3.0])
    assert [m.table_function_names for m in (model, before, after)] == [["drive"]] * 3


def test_compiled_code_shared_with_another_model_reads_each_ones_data(
    tmp_path, monkeypatch
) -> None:
    # Issue #772: the compiled code of a .net is shared by every model of it,
    # and is keyed by the .net alone. The data is the model's, read when the
    # code runs, so the models sharing it each run their own.
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    swapped = _model(tmp_path, _TFUN)
    _quiet_add(swapped, "drive", **SEVEN)
    assert _run(swapped, codegen=True)[0] == pytest.approx(7.0)
    assert _run(_model(tmp_path, _TFUN), codegen=True)[0] == pytest.approx(1.0)
    assert _run(swapped, codegen=True)[0] == pytest.approx(7.0)


# ─── What a replacement may not change ─────────────────────────────────────


@pytest.mark.parametrize(
    "functions, index, read_over",
    [(_TFUN, "k", "time"), (_TFUN_OVER_K, "time", "k"), (_TFUN_OVER_K, "a", "k")],
    ids=["time-to-k", "k-to-time", "k-to-a"],
)
def test_new_data_over_another_index_is_refused(tmp_path, functions, index, read_over) -> None:
    model = _model(tmp_path, functions)
    with pytest.raises(ModelError, match=f"is read over '{read_over}'"):
        model.add_table_function("drive", index=index, **SEVEN)
    assert _run(model)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("index", ["Time", "t", "time()"])
def test_the_same_index_may_be_spelled_otherwise(tmp_path, index) -> None:
    model = _model(tmp_path, _TFUN)
    _quiet_add(model, "drive", index=index, **SEVEN)
    assert _run(model)[0] == pytest.approx(7.0)


def test_a_parameter_index_may_be_written_as_a_call(tmp_path) -> None:
    model = _model(tmp_path, _TFUN_OVER_K)
    _quiet_add(model, "drive", index="k()", **SEVEN)
    assert _run(model)[0] == pytest.approx(7.0)


@pytest.mark.parametrize(
    "data, match",
    [
        (dict(times=[0.0, 10.0, 5.0], values=[7.0, 7.0, 7.0]), "increasing"),
        (dict(times=[0.0], values=[7.0]), "at least 2"),
        (dict(times=[0.0, float("nan")], values=[7.0, 7.0]), "NaN"),
        (dict(times=[0.0, 10.0], values=[7.0]), "same size|length"),
        (dict(file="no-such.tfun"), "cannot open"),
        (dict(file="wrong-header.tfun"), "header"),
        (dict(times=[0.0, 10.0], values=[7.0, 7.0], method="cubic"), None),
    ],
    ids=["unsorted", "one-point", "nan", "ragged", "no-file", "wrong-header", "bad-method"],
)
def test_bad_new_data_leaves_the_table_as_it_was(tmp_path, data, match) -> None:
    model = _model(tmp_path, _TFUN)
    (tmp_path / "wrong-header.tfun").write_text("# time other\n0 7\n10 7\n")
    data = {k: (tmp_path / v if k == "file" else v) for k, v in data.items()}
    with pytest.raises((ModelError, ValueError), match=match):
        model.add_table_function("drive", **data)
    assert model.table_function_names == ["drive"]
    assert model._core.table_function_tables[0]["ys"] == [1.0, 1.0]
    assert _run(model, codegen=True) == (pytest.approx(1.0), [1.0, 1.0, 1.0])


@ENGINES
def test_a_plain_functions_name_is_still_refused(tmp_path, codegen) -> None:
    model = _model(tmp_path, _PLAIN)
    with pytest.raises(ModelError, match="already has a function named 'drive'"):
        model.add_table_function("drive", **SEVEN)
    assert model.table_function_names == []
    assert _run(model, codegen)[0] == pytest.approx(1.0)


# ─── A table nothing reads ─────────────────────────────────────────────────


def test_a_table_under_a_new_name_is_added_with_a_warning(tmp_path) -> None:
    model = _model(tmp_path, _PLAIN)
    with pytest.warns(UserWarning, match="nothing in the model reads table function 'signal'"):
        model.add_table_function("signal", **SEVEN)
    assert model.table_function_names == ["signal"]
    with pytest.warns(UserWarning, match="nothing in the model reads"):
        model.add_table_function("signal", times=[0.0, 10.0], values=[3.0, 3.0])


@pytest.mark.parametrize("index", ["time", "k"])
def test_a_table_nothing_reads_leaves_the_run_as_it_was(tmp_path, index) -> None:
    # Read over the clock it marked the model as reading the time, which
    # refused steady_state() and held the early stop, and its breakpoints were
    # among the run's stops.
    model = _model(tmp_path, _PLAIN, _SETTLES)
    want = _run(model, t_end=10.0, n_points=11)
    with pytest.warns(UserWarning):
        model.add_table_function(
            "pulse",
            index=index,
            times=[0.0, 2.0, 2.5, 10.0],
            values=[0.0, 9.0, 0.0, 0.0],
            method="step",
        )
    assert model._core.functions_use_time is False
    assert model.clone()._core.functions_use_time is False
    assert model.table_function_breaks() == []
    assert _run(model, t_end=10.0, n_points=11) == want
    ss = bngsim.Simulator(model, method="ode").steady_state()
    assert ss.converged and ss.concentrations[0] == pytest.approx(1.0, rel=1e-6)


def test_a_tfun_on_the_clock_still_marks_the_model_and_its_clones(tmp_path) -> None:
    model = _model(tmp_path, _TFUN)
    assert model._core.functions_use_time is True
    _quiet_add(model, "drive", **SEVEN)
    assert model._core.functions_use_time is True
    assert model.clone()._core.functions_use_time is True
    assert _model(tmp_path, _TFUN_OVER_K)._core.functions_use_time is False
