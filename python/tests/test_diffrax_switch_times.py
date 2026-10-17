"""``run_diffrax`` at a switch on the clock (issue #865).

A rate law such as ``if(time() > 1.37, 5, 0.1)`` jumps at t = 1.37, and
diffrax's step-size controller stepped across it. At ``rtol=1e-6`` its error
around the switch was several times CVODE's, and at ``rtol=1e-8`` and tighter it
spent its whole step budget there and failed. The run is now split at the
switch times, the crossings CVODE stops on, and each piece reads the
right-hand side only on its own side of the jump. A switch on the model's state
has no such times, and is refused, as events are.

``A' = -f*A`` from ``A(0) = 100``. Its closed form is
``A = 100 exp(-0.1 t)`` up to ``t* = 1.37``, and ``A(t*) exp(-5 (t - t*))``
after. Other shapes are checked against ``Simulator(method="ode")`` at
``rtol=1e-12``. The error allowed is the size of diffrax's own on the same model
with no switch (``f() = 0.1``): 5.6e-5 at ``rtol=1e-8``.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

pytest.importorskip("diffrax")

from bngsim._diffrax_solver import run_diffrax  # noqa: E402

# The issue's model, with a parameter for the switch time.
_NET = """begin parameters
    1 k 1.0
    2 tsw 1.37
end parameters
begin species
    1 A() 100
end species
begin functions
    1 f() {f}
end functions
begin reactions
    1 1 0 f
end reactions
begin groups
    1 Aobs 1
end groups
"""

# The same with a counter, C' = 1 from C(0) = 0, so Cobs is the clock.
_NET_COUNTER = (
    _NET.replace("    1 A() 100\n", "    1 A() 100\n    2 C() 0\n")
    .replace("    1 1 0 f\n", "    1 1 0 f\n    2 0 2 k\n")
    .replace("    1 Aobs 1\n", "    1 Aobs 1\n    2 Cobs 2\n")
)

TS = np.linspace(0.0, 3.0, 31)


def _model(tmp_path: Path, f: str) -> bngsim.Model:
    path = tmp_path / "m.net"
    path.write_text((_NET_COUNTER if "Cobs" in f else _NET).format(f=f))
    return bngsim.Model.from_net(str(path))


def _diffrax(model, rtol=1e-8, **kw):
    r = run_diffrax(model, t_start=0.0, t_end=3.0, n_points=31, rtol=rtol, atol=rtol / 100, **kw)
    np.testing.assert_array_equal(r["time"], TS)
    return r["species"][:, 0], r["n_steps"]


def _ode(model) -> np.ndarray:
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 3.0), n_points=31, rtol=1e-12, atol=1e-14
    )
    return np.asarray(r.species)[:, 0]


def _closed_form(t, t_switch=1.37):
    return np.where(
        t <= t_switch,
        100.0 * np.exp(-0.1 * t),
        100.0 * np.exp(-0.1 * t_switch) * np.exp(-5.0 * (t - t_switch)),
    )


@pytest.mark.parametrize("rtol, allowed", [(1e-6, 2.5e-3), (1e-8, 2e-4), (1e-10, 1e-5)])
def test_the_issues_switch(tmp_path, rtol, allowed) -> None:
    # Before: 3.4e-3 in 114 steps at rtol=1e-6 (CVODE 7.5e-4), and "The
    # maximum number of solver steps was reached" at 1e-8 and 1e-10.
    a, n_steps = _diffrax(_model(tmp_path, "if(time()>1.37,5,0.1)"), rtol)
    assert np.abs(a - _closed_form(TS)).max() < allowed
    assert n_steps < 1000


SHAPES = {
    ">=": "if(time()>=1.37,5,0.1)",
    "two switches": "if(time()>1,5,0.1)+if(time()>2,-4,0)",
    "on an output time": "if(time()>1.5,5,0.1)",
    "exp": "if(exp(time())>20,5,0.1)",
    "square": "if(time()^2>2,5,0.1)",
    "periodic": "if(sin(3*time())>0,5,0.1)",
    "counter clock": "if(Cobs>1.37,5,0.1)",
    "no switch, moves with time": "0.1+0.1*sin(time())",
    "no switch": "0.1",
}


@pytest.mark.parametrize("f", list(SHAPES.values()), ids=list(SHAPES))
def test_each_shape_of_switch_matches_the_ode_path(tmp_path, f) -> None:
    a, _ = _diffrax(_model(tmp_path, f))
    assert np.abs(a - _ode(_model(tmp_path, f))).max() < 2e-4


def test_a_switch_time_moved_by_param_dict(tmp_path) -> None:
    # The switch times are the run's, on the clone param_dict is applied to, and
    # the caller's model keeps its own.
    model = _model(tmp_path, "if(time()>tsw,5,0.1)")
    a, _ = _diffrax(model, param_dict={"tsw": 2.0})
    assert np.abs(a - _closed_form(TS, t_switch=2.0)).max() < 2e-4
    assert model._core.get_param("tsw") == 1.37


def test_a_switch_on_the_state_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="switches on the model's state"):
        _diffrax(_model(tmp_path, "if(Aobs<50,5,0.1)"))


def test_a_condition_past_the_budget_of_stops_is_refused(tmp_path) -> None:
    # More crossings than a run may stop at have none placed; integrated across
    # they are the issue again.
    from bngsim._switch_sensitivity import _RUN_STOP_BUDGET

    omega = np.pi * (_RUN_STOP_BUDGET + 50) / 3.0
    with pytest.raises(ValueError, match="cannot stop at every switch"):
        _diffrax(_model(tmp_path, f"if(sin({omega:.6f}*time())>0,5,0.1)"))
