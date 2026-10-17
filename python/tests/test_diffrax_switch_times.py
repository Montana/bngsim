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
``rtol=1e-12``. The error allowed, 1e-4 at ``rtol=1e-8``, is about twice
diffrax's own on the model with no switch (``f() = 0.1``, 4.3e-5). Stepped
across, the counter clock's switch gave 1.6e-4.
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


@pytest.mark.parametrize("rtol, allowed", [(1e-6, 2.5e-3), (1e-8, 1e-4), (1e-10, 1e-7)])
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
    assert np.abs(a - _ode(_model(tmp_path, f))).max() < 1e-4


def test_a_switch_time_moved_by_param_dict(tmp_path) -> None:
    # The switch times are the run's, on the clone param_dict is applied to, and
    # the caller's model keeps its own.
    model = _model(tmp_path, "if(time()>tsw,5,0.1)")
    a, _ = _diffrax(model, param_dict={"tsw": 2.0})
    assert np.abs(a - _closed_form(TS, t_switch=2.0)).max() < 1e-4
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


@pytest.mark.parametrize(
    "f", ["if(time()>1.37,5,0.1)", "if(Cobs>1.37,5,0.1)"], ids=["time", "counter"]
)
def test_a_run_before_this_one_does_not_move_its_switches(tmp_path, f) -> None:
    # run_diffrax starts from the initial state, and a counter clock's crossing
    # was placed from the counter's present value: after an ODE run on the same
    # model it found none and stepped across the switch. (A switch on time()
    # reads no state, and is the control.)
    model = _model(tmp_path, f)
    bngsim.Simulator(model, method="ode").run(t_span=(0.0, 3.0), n_points=4)
    a, _ = _diffrax(model)
    np.testing.assert_array_equal(a, _diffrax(_model(tmp_path, f))[0])


def test_a_run_backward_with_no_switch(tmp_path) -> None:
    # A(3) = 100 run back to t = 0: A = 100 exp(-0.1 (t - 3)).
    r = run_diffrax(
        _model(tmp_path, "0.1"), t_start=3.0, t_end=0.0, n_points=7, rtol=1e-10, atol=1e-12
    )
    # It returned np.empty's contents at every point but t_end.
    np.testing.assert_allclose(
        r["species"][:, 0], 100.0 * np.exp(-0.1 * (r["time"] - 3.0)), rtol=1e-6
    )


def test_a_run_backward_over_a_switch_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="backward"):
        run_diffrax(_model(tmp_path, "if(time()>1.37,5,0.1)"), t_start=3.0, t_end=0.0)


def test_a_reporting_function_past_the_budget_is_not_refused(tmp_path) -> None:
    # A condition no rate law reads is no reason to refuse the run, as on the
    # ODE path; the species are those of the model without it.
    from bngsim._switch_sensitivity import _RUN_STOP_BUDGET

    omega = np.pi * (_RUN_STOP_BUDGET + 50) / 3.0
    path = tmp_path / "m.net"
    path.write_text(
        _NET.format(f="0.1").replace(
            "    1 f() 0.1\n", f"    1 f() 0.1\n    2 g() if(sin({omega:.6f}*time())>0,1,0)\n"
        )
    )
    a, _ = _diffrax(bngsim.Model.from_net(str(path)))
    np.testing.assert_array_equal(a, _diffrax(_model(tmp_path, "0.1"))[0])


def test_a_rate_law_that_moves_with_time_and_no_switch(tmp_path) -> None:
    # Reading the time clamped to the piece froze this rate in the run's last
    # step, where Kvaerno5 reads a stage past the step's end: 2.8e-8 at t = 3,
    # four times rtol, where main had 3.7e-10. A = 100 exp(-F) with
    # F = 0.1 t + 0.01 (1 - cos(10 t)).
    a, _ = _diffrax(_model(tmp_path, "0.1+0.1*sin(10*time())"), rtol=1e-10)
    exact = 100.0 * np.exp(-(0.1 * TS + 0.01 * (1.0 - np.cos(10.0 * TS))))
    assert abs(a[-1] - exact[-1]) < 1e-10 * exact[-1]


def test_a_switch_on_a_rate_law_that_moves_with_time(tmp_path) -> None:
    # f = r (1 + 0.5 sin(10 t)), r = 0.1 up to 1.37 and 5 after: the piece
    # before the switch keeps the rate's time dependence to its end (2.2e-7
    # with the time clamped, at rtol=1e-10).
    a, _ = _diffrax(_model(tmp_path, "if(time()>1.37,5,0.1)*(1+0.5*sin(10*time()))"), rtol=1e-10)

    def g(t):
        return t - 0.05 * np.cos(10.0 * t)

    s = 1.37
    f_int = np.where(s >= TS, 0.1 * (g(TS) - g(0.0)), 0.1 * (g(s) - g(0.0)) + 5.0 * (g(TS) - g(s)))
    assert np.abs(a - 100.0 * np.exp(-f_int)).max() < 1e-7
