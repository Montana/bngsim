"""A table function indexed by a function is read at the function's value under codegen (#965).

``g() tfun([0,100],[0,100], h)`` with ``h() 2*Atot``: the table spec names the
index by the slot the function ``h`` owns, ``index_kind == "parameter"``, and the
compiled RHS read that slot, ``p[idx]``. Only the interpreter's
``evaluate_functions`` writes it, so compiled code saw ``h``'s load-time value and
the trajectory froze at its initial state, with nothing raised. That hit
``codegen=True`` and the default path at or above ``BNGSIM_CODEGEN_THRESHOLD``.
The compiled RHS now reads ``func[h]``, computed before the table call.

The table is the identity, so g = h = 2A, dA/dt = -2A^2 and A(t) = 10/(1 + 20t).
"""

from __future__ import annotations

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest

NET = """begin parameters
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 g
end reactions
begin groups
    1 Atot 1
end groups
begin functions
{functions}
end functions
"""

SPELLINGS = {
    "whole body": "    1 g() tfun([0,100],[0,100], h)\n    2 h() 2*Atot",
    "wrapped in arithmetic": "    1 g() tfun([0,100],[0,100], h) + 0\n    2 h() 2*Atot",
    "index declared first": "    1 h() 2*Atot\n    2 g() tfun([0,100],[0,100], h)",
}

TIMES = np.linspace(0.0, 1.0, 5)
CLOSED_FORM = 10.0 / (1.0 + 20.0 * TIMES)


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _model(tmp_path, functions: str) -> bngsim.Model:
    net = tmp_path / "tfun_function_index.net"
    net.write_text(NET.format(functions=functions))
    return bngsim.Model.from_net(str(net))


def _a(sim: bngsim.Simulator) -> np.ndarray:
    result = sim.run(t_span=(0.0, 1.0), n_points=len(TIMES), rtol=1e-10, atol=1e-12)
    return np.asarray(result.species)[:, 0]


@pytest.mark.parametrize("functions", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_codegen_true_follows_the_closed_form(tmp_path, functions):
    sim = bngsim.Simulator(_model(tmp_path, functions), method="ode", codegen=True)
    assert sim.codegen_backend in ("cc", "mir")
    np.testing.assert_allclose(_a(sim), CLOSED_FORM, rtol=1e-7)


@pytest.mark.parametrize("functions", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_default_path_above_the_threshold_follows_the_closed_form(
    tmp_path, monkeypatch, functions
):
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    sim = bngsim.Simulator(_model(tmp_path, functions), method="ode")
    assert sim.codegen_backend in ("cc", "mir")
    np.testing.assert_allclose(_a(sim), CLOSED_FORM, rtol=1e-7)


@pytest.mark.parametrize("functions", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_compiled_and_interpreted_agree(tmp_path, functions):
    # A model per run: a run leaves its end state in the model it was given.
    compiled = _a(bngsim.Simulator(_model(tmp_path, functions), method="ode", codegen=True))
    interpreted = _a(bngsim.Simulator(_model(tmp_path, functions), method="ode", codegen=False))
    np.testing.assert_allclose(compiled, interpreted, rtol=1e-7)


def test_the_index_reads_the_functions_value_and_orders_it_first(tmp_path):
    data = _model(tmp_path, SPELLINGS["whole body"])._core.codegen_data()
    calls = cg._tfun_call_map(data)
    h = [f["name"] for f in data["functions"]].index("h")
    assert calls == {"g": (0, f"func[{h}]")}
    order = cg._topological_function_order(data["functions"], calls)
    assert order.index(h) < order.index(1 - h)


def test_a_table_over_a_plain_parameter_still_reads_its_slot(tmp_path):
    net = NET.replace(
        "begin parameters\nend parameters", "begin parameters\n    1 x 5\nend parameters"
    ).format(functions="    1 g() tfun([0,100],[0,100], x)")
    path = tmp_path / "tfun_parameter_index.net"
    path.write_text(net)
    data = bngsim.Model.from_net(str(path))._core.codegen_data()
    x = [p["name"] for p in data["parameters"]].index("x")
    assert cg._tfun_call_map(data) == {"g": (0, f"p[{x}]")}


# ── Forward sensitivities: the table has no analytic sensitivity RHS ─────────
# A run takes its columns from CVODES' difference quotient, over the compiled
# right-hand side, so it reads the index live as the run does.


def test_a_smooth_index_gives_the_closed_form_sensitivity(tmp_path):
    # h() = c*Atot: A = 10/(1 + 10ct), so dA/dc = -100t/(1 + 10ct)^2.
    net = NET.replace(
        "begin parameters\nend parameters", "begin parameters\n    1 c 2\nend parameters"
    ).format(functions="    1 g() tfun([0,100],[0,100], h)\n    2 h() c*Atot")
    path = tmp_path / "tfun_smooth_index.net"
    path.write_text(net)
    run = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["c"]
    ).run(t_span=(0.0, 1.0), n_points=3, rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, 0, 0]
    assert got == pytest.approx(-100.0 / (1.0 + 20.0) ** 2, rel=1e-6)


def test_an_index_that_holds_a_state_crossing_is_refused(tmp_path):
    # g jumps where h's condition crosses, and the quotient reads across it
    # (issue #938). Read at its load-time value, h never crossed, and the run
    # returned columns that ignored the jump.
    net = NET.replace(
        "begin parameters\nend parameters", "begin parameters\n    1 thr 2\nend parameters"
    ).format(functions="    1 h() if(Atot<thr,1,0)\n    2 g() tfun([0,1],[0,3], h)")
    path = tmp_path / "tfun_crossing_index.net"
    path.write_text(net)
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["thr"]
    )
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"Atot<thr"):
        sim.run(t_span=(0.0, 1.0), n_points=3)
