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


@pytest.mark.parametrize("rtol, allowed", [(1e-6, 2.5e-3), (1e-8, 1e-4), (1e-10, 1e-5)])
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
    # (exp, periodic and the no-switch shapes main gets within 1e-4 too: they
    # guard that the split places them right, the others that it is needed.)
    a, _ = _diffrax(_model(tmp_path, f))
    assert np.abs(a - _ode(_model(tmp_path, f))).max() < 1e-4


def test_a_switch_time_moved_by_param_dict(tmp_path) -> None:
    # The switch times are the run's, on the clone param_dict is applied to, and
    # the caller's model keeps its own. (At rtol=1e-10 main ran out of steps.)
    model = _model(tmp_path, "if(time()>tsw,5,0.1)")
    a, _ = _diffrax(model, rtol=1e-10, param_dict={"tsw": 2.0})
    assert np.abs(a - _closed_form(TS, t_switch=2.0)).max() < 1e-5
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
    # A draft of the split returned np.empty's contents at every point but t_end.
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
    # step, where Kvaerno5 reads a stage past the step's end: off by 3.8e-10 of
    # A(3) at rtol=1e-10, where main is off by 5e-12 of it. A = 100 exp(-F)
    # with F = 0.1 t + 0.01 (1 - cos(10 t)).
    a, _ = _diffrax(_model(tmp_path, "0.1+0.1*sin(10*time())"), rtol=1e-10)
    exact = 100.0 * np.exp(-(0.1 * TS + 0.01 * (1.0 - np.cos(10.0 * TS))))
    assert abs(a[-1] - exact[-1]) < 1e-10 * exact[-1]


def test_a_switch_on_a_rate_law_that_moves_with_time(tmp_path) -> None:
    # f = r (1 + 0.5 sin(10 t)), r = 0.1 up to 1.37 and 5 after, against the
    # closed form: the pieces keep the rate's time dependence to their ends.
    # (Main gets this within 1e-6 as well, in 2731 steps: a guard.)
    a, _ = _diffrax(_model(tmp_path, "if(time()>1.37,5,0.1)*(1+0.5*sin(10*time()))"), rtol=1e-10)

    def g(t):
        return t - 0.05 * np.cos(10.0 * t)

    s = 1.37
    f_int = np.where(s >= TS, 0.1 * (g(TS) - g(0.0)), 0.1 * (g(s) - g(0.0)) + 5.0 * (g(TS) - g(s)))
    assert np.abs(a - 100.0 * np.exp(-f_int)).max() < 1e-6


NARROW = {
    "1e-6 after the start": "if(time()>1e-06,5,0.1)",
    "1e-12 before the end": "if(time()>2.999999999999,5,0.1)",
    "a pulse 2e-8 wide": "if(time()>1.37,5,0.1)+if(time()>1.37000002,-4.9,0)",
}


@pytest.mark.parametrize("f", list(NARROW.values()), ids=list(NARROW))
def test_a_narrow_piece(tmp_path, f) -> None:
    # Kvaerno3's Newton iteration read roundoff on so short a piece as
    # divergence, and turned its steps down until they ran out; main and the
    # ODE path complete. A stretch this short goes to Kvaerno3, and to Tsit5,
    # which has no Newton iteration, where Kvaerno3 fails (1e-6 after the
    # start); Kvaerno5 took 5642 steps over the pulse.
    a, n_steps = _diffrax(_model(tmp_path, f))
    assert np.abs(a - _ode(_model(tmp_path, f))).max() < 1e-4
    assert n_steps < 2000


NARROWER_THAN_THE_LEAST_STEP = {
    "1e-12 at the start of [0, 1000]": ("if(time()>1e-12,0.001,0.1)", 1000.0),
    "a pulse 1e-12 wide on [0, 1000]": (
        "if(time()>1.37,0.003,0.001)+if(time()>1.370000000001,-0.002,0)",
        1000.0,
    ),
    "1e-300 at the start of [0, 3]": ("if(time()>1e-300,5,0.1)", 3.0),
}


@pytest.mark.parametrize(
    "f, t_end", list(NARROWER_THAN_THE_LEAST_STEP.values()), ids=list(NARROWER_THAN_THE_LEAST_STEP)
)
def test_a_piece_narrower_than_the_least_step(tmp_path, f, t_end) -> None:
    # Such a piece is one step, accepted, but diffrax checks the least step on
    # the step it proposes next, and flagged the solve as failed. Every solver
    # failed it the same way, and the run was refused; main completes them.
    # diffrax's own error on [0, 1000] with no switch is 3.5e-5.
    r = run_diffrax(
        _model(tmp_path, f), t_start=0.0, t_end=t_end, n_points=11, rtol=1e-8, atol=1e-10
    )
    ref = bngsim.Simulator(_model(tmp_path, f), method="ode").run(
        t_span=(0.0, t_end), n_points=11, rtol=1e-12, atol=1e-14
    )
    assert np.abs(r["species"] - np.asarray(ref.species)).max() < 1e-4


def test_a_counter_clock_restarts_past_its_threshold(tmp_path) -> None:
    # C = t reaches 2.9 three or four floats short, and the next piece read the
    # branch that had ended: 9.5e-7 of A(3) off at rtol=1e-8, where literal
    # time is 5.8e-9 off.
    a, _ = _diffrax(_model(tmp_path, "if(Cobs>2.9,5,0.1)"))
    exact = _closed_form(TS, t_switch=2.9)
    assert abs(a[-1] / exact[-1] - 1.0) < 1e-7


def test_a_condition_only_an_output_reads_does_not_split_the_run(tmp_path) -> None:
    # Its split changed the answer of a model whose rate laws never switch, and
    # cost a fast one hundreds of steps.
    path = tmp_path / "m.net"
    path.write_text(
        _NET.format(f="0.1").replace(
            "    1 f() 0.1\n", "    1 f() 0.1\n    2 g() if(time()>1.37,1,0)\n"
        )
    )
    a, n_steps = _diffrax(bngsim.Model.from_net(str(path)))
    b, m_steps = _diffrax(_model(tmp_path, "0.1"))
    np.testing.assert_array_equal(a, b)
    assert n_steps == m_steps


def test_crossings_a_float_apart_are_one_switch(tmp_path) -> None:
    # Two crossings a float apart made a piece that ran backward, and two
    # floats apart one of no width; within the ODE path's one instant they are
    # one switch, the piece before ending short of the first and the one after
    # beginning past the second.
    from bngsim._diffrax_solver import _switch_times

    f = "if(time()>1.37,5,0.1)+if(time()>1.3700000000000003,-4,0)"
    # diffrax happened to integrate the backward piece; what it is handed is
    # pinned here, its answer below.
    assert _switch_times(_model(tmp_path, f), 0.0, 3.0) == [(1.37, 1.3700000000000003, ())]
    a, _ = _diffrax(_model(tmp_path, f))
    assert np.abs(a - _ode(_model(tmp_path, f))).max() < 1e-4


def test_a_condition_on_two_counters_is_refused(tmp_path) -> None:
    # No resolver places it, and the resolvers passed it over: the run stepped
    # across it, which is the issue again.
    net = (
        _NET_COUNTER.replace("    2 C() 0\n", "    2 C() 0\n    3 D() 0.5\n")
        .replace("    2 0 2 k\n", "    2 0 2 k\n    3 0 3 k\n")
        .replace("    2 Cobs 2\n", "    2 Cobs 2\n    3 Dobs 3\n")
    )
    path = tmp_path / "m.net"
    path.write_text(net.format(f="if(Cobs+Dobs>2.0,5,0.1)"))
    with pytest.raises(ValueError, match="cannot place the times these conditions switch at"):
        _diffrax(bngsim.Model.from_net(str(path)))


def test_fast_dynamics_before_a_switch_cost_what_they_did(tmp_path) -> None:
    # The issue's switch the other way round, 5 before and 0.1 after. Kvaerno3
    # over the whole piece before the switch took 1512 steps at rtol=1e-8, where
    # main took 102 at 1e-6 and ran out of steps at 1e-8. Kvaerno5 takes four
    # fifths of what is left of the piece, three times, and Kvaerno3 the rest.
    f = "if(time()>1.37,0.1,5)"
    a, n_steps = _diffrax(_model(tmp_path, f))
    assert np.abs(a - _ode(_model(tmp_path, f))).max() < 1e-4
    assert n_steps < 400


def test_a_stiff_run_far_from_t_zero(tmp_path) -> None:
    # Narrow measured against max(|t|) made every piece of a run on [1e6, 1e6+3]
    # narrow, and a narrow stretch is solved with an explicit method: on a stiff
    # model, thousands of steps where the same run on [0, 3] takes a hundred.
    # A <-> B at 1e4 each way, and A decays at f, which switches 1.37 into the
    # run. The model is the same at every time but for the switch, so the run on
    # [0, 3] is the reference (CVODE cannot reach rtol=1e-12 at t ~ 1e6).
    def model(t0):
        s = t0 + 1.37
        net = (
            _NET.replace("    1 A() 100\n", "    1 A() 100\n    2 B() 0\n")
            .replace("    2 tsw 1.37\n", "    2 tsw 1.37\n    3 kf 1e4\n")
            .replace("    1 1 0 f\n", "    1 1 0 f\n    2 1 2 kf\n    3 2 1 kf\n")
            .format(f=f"if(time()>{s!r},5,0.1)")
        )
        path = tmp_path / f"stiff{t0:g}.net"
        path.write_text(net)
        return bngsim.Model.from_net(str(path))

    r = run_diffrax(model(1e6), t_start=1e6, t_end=1e6 + 3.0, n_points=31, rtol=1e-8, atol=1e-10)
    ref = bngsim.Simulator(model(0.0), method="ode").run(
        t_span=(0.0, 3.0), n_points=31, rtol=1e-12, atol=1e-14
    )
    assert np.abs(r["species"] - np.asarray(ref.species)).max() < 1e-4
    assert r["n_steps"] < 1000


def test_a_transient_too_fast_for_the_time_scale_is_refused(tmp_path) -> None:
    # At t ~ 1e6 the least step is 3e-8, and a rate of 1e8 just past a switch
    # needs shorter. Made to take the least step anyway, the run returned
    # A = -1.2e-6 for 4.1e-3, with nothing said; CVODE fails on it too.
    s = 1e6 + 1.0
    with pytest.raises(ValueError, match="no step shorter than"):
        run_diffrax(
            _model(tmp_path, f"if(time()>{s!r},1e8,0.1)"),
            t_start=1e6,
            t_end=s + 1e-7,
            n_points=2,
            rtol=1e-8,
            atol=1e-10,
        )


def _stiff(tmp_path, f, kf):
    """The issue's model with A <-> B at *kf* each way."""
    net = (
        _NET.replace("    1 A() 100\n", "    1 A() 100\n    2 B() 0\n")
        .replace("    2 tsw 1.37\n", f"    2 tsw 1.37\n    3 kf {kf}\n")
        .replace("    1 1 0 f\n", "    1 1 0 f\n    2 1 2 kf\n    3 2 1 kf\n")
        .format(f=f)
    )
    path = tmp_path / "stiff.net"
    path.write_text(net)
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize(
    "f, kf, t_end, rtol",
    [
        ("if(time()>99.95,5,0.01)", 1e7, 100.0, 1e-6),
        ("if(time()>1,5,0.1)+if(time()>1.2,-4.9,0)", 1e8, 3.0, 1e-8),
    ],
    ids=["near the end of [0, 100]", "a pulse 0.2 wide"],
)
def test_a_narrow_stretch_of_a_stiff_model(tmp_path, f, kf, t_end, rtol) -> None:
    # A stretch is narrow against the run's span, so on [0, 100] one 0.05 long
    # is, and the last 0.8% of the pulse's piece is. Solved with Tsit5 alone,
    # whose step stiffness binds, both ran past max_steps and were refused (main
    # completes them, in 149 and 47740 steps); Kvaerno3 is tried first.
    r = run_diffrax(
        _stiff(tmp_path, f, kf), t_start=0.0, t_end=t_end, n_points=11, rtol=rtol, atol=rtol / 100
    )
    ref = bngsim.Simulator(_stiff(tmp_path, f, kf), method="ode").run(
        t_span=(0.0, t_end), n_points=11, rtol=1e-12, atol=1e-14
    )
    assert np.abs(r["species"] - np.asarray(ref.species)).max() < 1e-3
    assert r["n_steps"] < 1000


def test_a_run_backward_that_crosses_no_switch(tmp_path) -> None:
    # A clock condition outside the window is no switch of this run: main ran
    # it, and it was refused. A(1) = 100 run back to 0 at rate 0.1.
    r = run_diffrax(
        _model(tmp_path, "if(time()>1.37,5,0.1)"),
        t_start=1.0,
        t_end=0.0,
        n_points=11,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        r["species"][:, 0], 100.0 * np.exp(-0.1 * (r["time"] - 1.0)), rtol=1e-6
    )


def test_a_run_backward_on_a_counter_clock_is_refused(tmp_path) -> None:
    # A counter's crossings are placed from its value at t_start, forward.
    with pytest.raises(ValueError, match="counter clock"):
        run_diffrax(_model(tmp_path, "if(Cobs>1.37,5,0.1)"), t_start=3.0, t_end=0.0)
