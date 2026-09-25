"""``build_model_from_parsed(parse_net_file(f))`` reproduces ``Model.from_net(f)`` (issue #803).

The C++ loader reads a ``.net`` file in two phases: it parses the text into
records, then feeds them to ModelBuilder. ``parse_net_file`` returns the first
phase's records as a dict, and ``build_model_from_parsed`` makes the second
phase's calls from them in Python. So the two doors share one reading of the
format and differ only in who feeds the builder, and this file pins that the
Python feed builds the same model as the C++ one, over every committed ``.net``.

"The same model" is checked three ways, none of them through the dict itself:
``codegen_data()`` equal field for field (every reaction's type, rate parameter
and stat factor, every function's expression, every observable's entries), the
parameters and initial state equal, and the RHS equal at a positive state.

Before step 5 the dict came from a second parser, in Python; it was moved in step
with the loader by hand, and #554, #597, #600 and #606 are the times it was not.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import build_model_from_parsed, parse_net_file

_NET_TREE = Path(__file__).resolve().parents[2]
#: The roots holding committed ``.net`` fixtures, named rather than reached by
#: rglob-ing the repo root (issue #590: a whole-tree rglob sweeps up caches, and
#: ``Path.rglob`` does not descend into a symlinked directory below the root).
_NET_ROOTS = ("benchmarks", "parity_checks", "tests")
_NETS = sorted(
    p
    for root in _NET_ROOTS
    for p in (_NET_TREE / root).rglob("*.net")
    if "build" not in p.parts and ".pytest_cache" not in p.parts
)
#: The constructs where the two phases could part ways, each in a fixture the
#: sweep must reach, or a green run proves nothing about them: the Sat/Hill
#: rewrite, table functions, Michaelis-Menten, a clamped species (bare and inside
#: a compartment prefix), an expression-valued initial concentration, and a
#: stat-factor prefix.
_MUST_REACH = {
    "hill_rewrite.net",
    "sat_rewrite.net",
    "tfun_time_indexed.net",
    "tfun_param_indexed.net",
    "wrap_single.net",
    "mm_tqssa.net",
    "fixed_species.net",
    "sink_compart.net",
    "expr_param_species.net",
    "homodimer_ssa.net",
}
_ABSENT = f".net models not present under {_NET_TREE}"


def _differences(a, b, path="codegen_data"):
    """Where two ``codegen_data()`` trees differ, exactly (NaN equal to NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [d for k in sorted(a) for d in _differences(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _differences(x, y, f"{path}[{i}]")
        ]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


@pytest.mark.parametrize(
    "net",
    _NETS or [pytest.param(None, marks=pytest.mark.skip(reason=_ABSENT))],
    ids=lambda p: str(p.relative_to(_NET_TREE)) if isinstance(p, Path) else "none",
)
def test_the_dict_builds_the_model_from_net_loads(net: Path) -> None:
    with warnings.catch_warnings():
        # The Sat/Hill rewrite warns on both doors; that it does is pinned in
        # test_net_reader_legacy_and_tfun.py.
        warnings.simplefilter("ignore", UserWarning)
        loaded = bngsim.Model.from_net(str(net))
        built = build_model_from_parsed(parse_net_file(net))
    a, b = loaded._core, built._core

    assert not _differences(a.codegen_data(), b.codegen_data())
    assert list(a.param_names) == list(b.param_names)
    assert [a.get_param(n) for n in a.param_names] == [b.get_param(n) for n in b.param_names]
    assert list(a.species_ic_param_refs) == list(b.species_ic_param_refs)
    # Python-side state Model.from_net sets on top of the core: the output column
    # of a species an assignment rule defines (sbml_to_net's networks, #515).
    assert loaded._ar_report_map == built._ar_report_map
    y0 = np.asarray(a.get_initial_state())
    np.testing.assert_array_equal(y0, np.asarray(b.get_initial_state()))

    rng = np.random.default_rng(803)
    y = np.abs(y0) * (1.0 + rng.random(y0.size)) + max(float(np.abs(y0).max(initial=0.0)), 1.0) * (
        0.1 + rng.random(y0.size)
    )
    np.testing.assert_array_equal(loaded.rhs(y, 0.37), built.rhs(y, 0.37))


@pytest.mark.skipif(not _NETS, reason=_ABSENT)
def test_the_sweep_reaches_every_construct_that_could_diverge() -> None:
    missing = _MUST_REACH - {p.name for p in _NETS}
    assert not missing, f"the round-trip sweep cannot see {sorted(missing)}"
    assert len(_NETS) > 100, f"only {len(_NETS)} .net files found under {_NET_TREE}"


#: Two inline tables, whose data and interpolation method codegen_data() does not
#: carry, so the sweep above cannot see them: a step table that is a whole
#: function, and a linear one inside arithmetic (the synthetic-table shape).
_INLINE_TABLES = """begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 drive #_R1
    2 1 2 wrapped #_R2
end reactions
begin groups
    1 Atot 1
end groups
begin functions
    1 drive() tfun([0,1,2,3],[0,10,20,30],time, method=>"step")
    2 wrapped() k*(tfun([0,2],[1,3],time)+1)
end functions
"""


def test_inline_tables_keep_their_data_and_method(tmp_path: Path) -> None:
    """At t = 1.5 the step table holds 10 (linear would give 15) and the linear
    one reads 2.5, so dA/dt = -(10 + 0.5*(2.5+1)) * 10 = -117.5."""
    net = tmp_path / "inline.net"
    net.write_text(_INLINE_TABLES)
    loaded = bngsim.Model.from_net(str(net))
    built = build_model_from_parsed(parse_net_file(net))
    y = np.array([10.0, 0.0])
    for t in (0.25, 1.0, 1.5, 2.5):
        np.testing.assert_array_equal(loaded.rhs(y, t), built.rhs(y, t))
    np.testing.assert_allclose(built.rhs(y, 1.5), [-117.5, 117.5], rtol=1e-12)
