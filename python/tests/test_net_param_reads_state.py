"""GH #844: a .net parameter that reads an observable or a function is refused.

A parameter is a constant. BNG2.pl refuses one whose expression names an
observable ("Parameter 'Atot' is referenced but not defined"). bngsim compiled
it, evaluated it once at build before any observable or function had a value,
and loaded it as 0.0, so a reaction whose rate it was never fired.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError

_NET = """begin parameters
    1 k   2.0
    2 k2  {expr}
end parameters
begin species
    1 A() 100
end species
begin functions
    1 f() Atot*k
end functions
begin reactions
    1 1 0 k2
end reactions
begin groups
    1 Atot 1
end groups
"""


def _write(tmp_path: Path, expr: str) -> str:
    path = tmp_path / "m.net"
    path.write_text(_NET.format(expr=expr), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    ("expr", "kind", "name"),
    [("Atot*k", "observable", "Atot"), ("f*2", "function", "f"), ("f()*2", "function", "f")],
)
def test_a_parameter_that_reads_the_state_is_refused(tmp_path, expr, kind, name):
    with pytest.raises(ModelError, match=rf"parameter 'k2' .* reads the {kind} '{name}'"):
        bngsim.Model.from_net(_write(tmp_path, expr))


@pytest.mark.parametrize(("expr", "value"), [("k*3", 6.0), ("2.5e-3*k", 0.005), ("1E2", 100.0)])
def test_a_constant_parameter_expression_still_loads(tmp_path, expr, value):
    m = bngsim.Model.from_net(_write(tmp_path, expr))
    assert m._core.get_param("k2") == pytest.approx(value)


def test_a_parameter_shadowed_by_a_same_named_function_still_loads():
    """The #266 shape (an SBML assignment rule after .net conversion): a
    parameter row and a function share a name. Not a parameter reading state."""
    data = Path(__file__).resolve().parents[2] / "tests" / "data" / "shadowed_function_param.net"
    bngsim.Model.from_net(str(data))
