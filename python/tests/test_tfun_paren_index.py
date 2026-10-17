"""A ``.net`` tfun whose index is written as a call, ``k()``.

BioNetGen keeps a tfun's index argument as written and puts it in the ``.net``
as it is, so a BNGL ``tfun([0,10], [0,10], Atot())`` gives a ``.net`` line with
``Atot()`` in it. The loader read the index up to the first parenthesis, as
``Atot(``, and the load failed: "index variable 'Atot(' not found". The bare
name loaded. Both readers share the parse (``net_function_tables``), so both
failed.

The table is the identity on ``[0, 10]``, so ``drive`` is the value of its
index, and ``A' = drive`` from ``A(0) = 0`` makes ``A(1)`` that value: 4 for
``k`` and for ``Bobs``, 8 for ``h() = 2*k``, and 1/2 over the clock.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._net_reader import build_model_from_parsed, parse_net_file

_NET = """begin parameters
    1 k 4
end parameters
begin functions
    1 h() 2*k
{functions}
end functions
begin species
    1 A() 0
    2 B() 4
end species
begin reactions
    1 0 1 drive
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
end groups
"""

# index as the table reads it -> A(1), and the kind the core reports
INDEXES = {
    "k": (4.0, "parameter"),
    "Bobs": (4.0, "observable"),
    "h": (8.0, "function"),
    "time": (0.5, "time"),
}

FORMS = {
    "inline": "    2 drive() tfun([0,10],[0,10],{index})",
    "inline-method": '    2 drive() tfun([0,10], [0,10], {index}, method=>"linear")',
    "file": "    2 drive() tfun('d.tfun',{index})",
    "embedded": "    2 drive() 1*tfun([0,10],[0,10],{index})",
}


def _net(tmp_path: Path, form: str, index: str) -> Path:
    bare = index.removesuffix("()")
    (tmp_path / "d.tfun").write_text(f"# {bare} drive\n0 0\n10 10\n")
    path = tmp_path / "m.net"
    path.write_text(_NET.format(functions=FORMS[form].format(index=index)))
    return path


def _load(path: Path, loader: str) -> bngsim.Model:
    if loader == "net_file_loader":
        return bngsim.Model.from_net(str(path))
    return build_model_from_parsed(parse_net_file(path))


def _a_at_1(model: bngsim.Model, codegen: bool) -> float:
    r = bngsim.Simulator(model, method="ode", codegen=codegen).run(t_span=(0.0, 1.0), n_points=2)
    return float(np.asarray(r.species)[-1, 0])


@pytest.mark.parametrize("call", [False, True], ids=["bare", "call"])
@pytest.mark.parametrize("index", list(INDEXES))
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize(
    "loader, codegen",
    [("net_file_loader", False), ("net_file_loader", True), ("net_reader", False)],
    ids=["cpp-interpreted", "cpp-compiled", "python-reader"],
)
def test_an_index_written_as_a_call_reads_what_the_name_does(
    tmp_path, loader, codegen, form, index, call
) -> None:
    model = _load(_net(tmp_path, form, f"{index}()" if call else index), loader)
    want, kind = INDEXES[index]
    (table,) = model._core.table_function_tables
    assert table["index_kind"] == kind
    assert _a_at_1(model, codegen) == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize(
    "index, want", [("Time()", 0.5), ("t()", 0.5), ("k ()", 4.0), ("k( )", 4.0)]
)
def test_other_spellings_of_a_call(tmp_path, index, want) -> None:
    # `Time()` and `t()` are the clock, as `Time` and `t` are. The loader reads
    # the name before a space, as it always has, so `k ()` is `k`; a call with a
    # space inside its parentheses is a call of `k` all the same.
    path = _net(tmp_path, "inline", index)
    assert _a_at_1(bngsim.Model.from_net(str(path)), False) == pytest.approx(want, rel=1e-6)


def test_the_call_does_not_swallow_the_rest_of_the_line(tmp_path) -> None:
    # Two tables in one function, the first over a call: each call is its own,
    # and the method of the second is the second's.
    path = tmp_path / "m.net"
    path.write_text(
        _NET.format(
            functions="    2 drive() tfun([0,10],[0,10],k()) + "
            'tfun([0,1,10],[0,10,10],time(),method=>"step")'
        )
    )
    for loader in ("net_file_loader", "net_reader"):
        model = _load(path, loader)
        tables = model._core.table_function_tables
        assert [(t["index_kind"], t["method"]) for t in tables] == [
            ("parameter", "linear"),
            ("time", "step"),
        ]
        # 4 throughout, plus a step that is 0 on [0, 1): A(1) = 4.
        assert _a_at_1(model, False) == pytest.approx(4.0, rel=1e-6)
