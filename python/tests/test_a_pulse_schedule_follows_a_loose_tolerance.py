"""At a loose tolerance a stop sits on the last double before each change of a
rate law's schedule (issue #1226).

A step ends on a stop, and the implicit step reads the rate at its end. On the
first double of the next stretch, as a stop on the change is, the step that
reaches the stop is taken at the next stretch's rate. At a tight tolerance that
step fails on the jump, and the run walks the stretch before it. At a loose
one, where `rtol*|X|` has grown past the jump, it passes and spans the stretch:
`if(mod(time(), 24) < 1, 1, 0)` over 1,000 days came to 14,627 at rtol 1e-2
for 1,000, each day's 23 hours off taken at the rate of the hour on, and the
hour on at the rate off.

The oracle is the schedule's closed form: a pulse of width `w` every `P`.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import crossings_with_periodic

_NET = """begin parameters
    1 P {P!r}
    2 w {w!r}
end parameters
begin species
    1 X() {x0!r}
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
"""

_SPELLINGS = {
    "mod": "if(mod(time(), P) < w, 1, 0)",
    "remainder": "if(time() - P*floor(time()/P) < w, 1, 0)",
}
# (P, w): edges that are exact doubles and edges that are not.
_SCHEDULES = [(24.0, 1.0), (1.0, 0.1), (0.3, 0.1), (0.03, 0.01), (1.0, 0.5)]


def _model(tmp_path, dose, P, w, x0=0.0):
    path = tmp_path / "m.net"
    path.write_text(_NET.format(P=P, w=w, dose=dose, x0=x0), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _on_time(P, w, t_end):
    n = math.floor(t_end / P)
    return n * w + min(t_end - n * P, w)


@pytest.mark.parametrize("rtol", [1e-2, 3e-3, 1e-3])
@pytest.mark.parametrize("P, w", _SCHEDULES)
@pytest.mark.parametrize("dose", _SPELLINGS.values(), ids=_SPELLINGS.keys())
def test_a_pulse_schedule_at_a_loose_tolerance_is_exact(tmp_path, dose, P, w, rtol):
    """The issue's table, 1,000 periods: on main from 1363% over to 21% under,
    and exact to rounding here."""
    t_end = 1000 * P
    sim = bngsim.Simulator(_model(tmp_path, dose, P, w), method="ode")
    r = sim.run(t_span=(0.0, t_end), n_points=3, rtol=rtol, atol=rtol * 1e-2)
    x = float(np.asarray(r.species)[-1, 0])
    assert x == pytest.approx(_on_time(P, w, t_end), rel=1e-9)


def test_a_stretch_is_taken_at_its_own_rate(tmp_path):
    """Each day's hour on adds its hour, and its 23 hours off add nothing: on
    main, late in the run, 0.0004 and 22.9955."""
    edges = np.sort(np.concatenate([np.arange(0.0, 24001.0, 24.0), np.arange(1.0, 24000.0, 24.0)]))
    sim = bngsim.Simulator(_model(tmp_path, _SPELLINGS["mod"], 24.0, 1.0), method="ode")
    r = sim.run(sample_times=edges, rtol=1e-2, atol=1e-4)
    x = np.asarray(r.species)[:, 0]
    assert np.allclose(x[1::2] - x[0:-1:2], 1.0, rtol=0.0, atol=1e-9)
    assert np.allclose(x[2::2] - x[1:-1:2], 0.0, rtol=0.0, atol=1e-9)


def test_from_a_large_start_a_pulse_schedule_follows_the_tolerance(tmp_path):
    """The error test is relative to X: from X(0) = 1e4 at rtol 1e-2 the hour
    on adds a hundredth of `rtol*|X|`, and the stretch was taken as a whole."""
    model = _model(tmp_path, _SPELLINGS["mod"], 24.0, 1.0, x0=1e4)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 24000.0), n_points=3, rtol=1e-2, atol=1e-4
    )
    assert float(np.asarray(r.species)[-1, 0]) - 1e4 == pytest.approx(1000.0, rel=1e-9)


class _Options:
    """What `_apply_crossing_stops` reads and writes, kept where it can be read."""

    max_step_size = 0.0

    def __init__(self, rtol):
        self.rtol = rtol
        self.stops = []

    def set_crossing_stops(self, stops, restarts=()):
        self.stops = [time for time, _clock, _value in stops]

    def set_crossing_probes(self, probes):
        pass


@pytest.mark.parametrize("rtol", [1e-2, 1e-3])
def test_each_stop_is_the_last_double_before_its_change(tmp_path, rtol):
    model = _model(tmp_path, _SPELLINGS["remainder"], 0.3, 0.1)
    opts = _Options(rtol)
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, 0.0, 30.0)
    stops = np.asarray(opts.stops)
    law = "if(time() - 0.3*floor(time()/0.3) < 0.1, 1, 0)"
    on = np.asarray(model._core.evaluate_over_times(law, stops))
    after = np.asarray(model._core.evaluate_over_times(law, np.nextafter(stops, np.inf)))
    assert stops.size == 200  # two a period, the one on the run's last instant too
    assert np.all(on != after)


def _resolved(model, t0, t1, **kw):
    conds = model.time_discontinuity_conditions()
    found, periodic, _ = crossings_with_periodic(model._core, t0, t1, conds, **kw)
    return [s.time for s in found], [s.time for s in periodic]


def test_an_event_trigger_and_a_sensitivity_run_keep_their_stops(tmp_path):
    """An event's trigger is located as a root, and its stop is where the
    resolvers put it (issue #891); a run with sensitivity columns reads its
    switch times off the resolvers' own numbers."""
    model = _model(tmp_path, _SPELLINGS["remainder"], 0.3, 0.1)
    conds = frozenset(model.time_discontinuity_conditions())
    placed = _resolved(model, 0.0, 30.0)
    assert _resolved(model, 0.0, 30.0, touches=conds, before_the_flip=True) == _resolved(
        model, 0.0, 30.0, touches=conds
    )
    assert _resolved(model, 0.0, 30.0, sensitivity_params=("P",), before_the_flip=True) == (
        _resolved(model, 0.0, 30.0, sensitivity_params=("P",))
    )
    assert _resolved(model, 0.0, 30.0, before_the_flip=True) != placed


_COUNTER_NET = """begin parameters
    1 P 0.3
    2 w 0.1
    3 one 1
end parameters
begin species
    1 X() 0
    2 T() 0.7
end species
begin functions
    1 dose() if(t - P*floor(t/P) < w, 1, 0)
end functions
begin reactions
    1 0 1 dose #_R1
    2 0 2 one #clock
end reactions
begin groups
    1 t 2
end groups
"""


def test_a_schedule_on_a_counter_keeps_its_stops(tmp_path):
    """The integrated counter lands a few of its doubles past a stop's value,
    and a stop a double before a change is not a stop before it there."""
    path = tmp_path / "counter.net"
    path.write_text(_COUNTER_NET, encoding="utf-8")
    model = bngsim.Model.from_net(str(path))
    assert _resolved(model, 0.0, 30.0, before_the_flip=True) == _resolved(model, 0.0, 30.0)


@pytest.mark.parametrize(
    "dose",
    [
        "mod(time(), 0.7)",
        "time() - 0.7*floor(time()/0.7)",
        "if(mod(time(), 0.7) < 0.35, mod(time(), 0.7), 0)",
        "if(mod(time(), 2) < 1, 1, 0)*time()",
        "if(mod(time(), 2) < 1, 1, 0)*exp(-0.01*time())",
    ],
    ids=[
        "sawtooth",
        "floor-sawtooth",
        "ramp-inside-a-pulse",
        "pulse-times-time",
        "pulse-times-exp",
    ],
)
def test_a_rate_that_changes_between_its_stops_keeps_them(tmp_path, dose):
    """Spanned by one step, its stretch is read at one end, and the end a
    double before the change is no better than the other: from X(0) = 1e4 at
    rtol 1e-3, `mod(time(), 0.7)` over 70 was 19.3 over a double before each
    change and 0.45 as the resolvers put them, and written with `floor`, as
    SBML's `rem` loads, 19.3 for 1.61."""
    path = tmp_path / "ramp.net"
    path.write_text(_NET.format(P=0.7, w=0.35, dose=dose, x0=0.0), encoding="utf-8")
    model = bngsim.Model.from_net(str(path))
    placed = _resolved(model, 0.0, 70.0)
    assert _resolved(model, 0.0, 70.0, before_the_flip=True) == placed


_TWO_NET = """begin parameters
    1 one 1
end parameters
begin species
    1 X() 0
    2 Y() 0
end species
begin functions
    1 dose() if(mod(time(), 24) < 1, 1, 0)
    2 saw() {saw}
    3 phase() {phase}
end functions
begin reactions
    1 0 1 dose #_R1
    2 0 2 saw #_R2
end reactions
"""


@pytest.mark.parametrize(
    "saw, phase",
    [("one", "mod(time(), 7.3)"), ("0.01*mod(time(), 7.3137)", "0.5")],
    ids=["a-reported-sawtooth", "beside-a-sawtooth"],
)
def test_a_pulse_beside_a_sawtooth_of_its_own_is_read_before_its_changes(tmp_path, saw, phase):
    """The pulse's rate law reads no sawtooth: one in another reaction, or in
    a function that only reports, does not take its stops from it. On main
    the pulse is 3,368.5 for 1,000 beside a reported `mod(time(), 7.3)`, and
    3,329.2 beside the sawtooth in the other reaction. (That sawtooth's
    period, 7.3137, meets none of the pulse's edges within an instant in the
    run, where every stop would go back to where the resolvers put it, #891:
    `mod(time(), 7.31)` meets one at 17,544.)"""
    path = tmp_path / "two.net"
    path.write_text(_TWO_NET.format(saw=saw, phase=phase), encoding="utf-8")
    r = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
        t_span=(0.0, 24000.0), n_points=3, rtol=1e-2, atol=1e-4
    )
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(1000.0, rel=1e-9)


_LAYOUTS = {
    "phase": ["phase() mod(time(), 24)", "dose() if(phase() < 1, 1, 0)"],
    "phase-and-light": [
        "phase() mod(time(), 24)",
        "light() if(phase() < 1, 1, 0)",
        "dose() light()",
    ],
    "on": ["on() mod(time(), 24) < 1", "dose() if(on(), 1, 0)"],
    "floor-phase": ["phase() time() - 24*floor(time()/24)", "dose() if(phase() < 1, 1, 0)"],
    "and-of-two": ["phase() mod(time(), 24)", "dose() if(phase() >= 0 && phase() < 1, 1, 0)"],
}


@pytest.mark.parametrize("funcs", _LAYOUTS.values(), ids=_LAYOUTS.keys())
def test_a_pulse_written_through_a_phase_function_is_read_before_its_changes(tmp_path, funcs):
    """A phase function's own body is a sawtooth, but no rate law reads it
    except through the pulse's comparison: read with the rest of the functions
    a rate law reaches, it kept the pulse at main's 14,627 for 1,000."""
    lines = "\n".join(f"    {i} {f}" for i, f in enumerate(funcs, start=1))
    path = tmp_path / "phase.net"
    path.write_text(
        _NET.replace("    1 dose() {dose}", lines).format(P=24.0, w=1.0, x0=0.0), encoding="utf-8"
    )
    r = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
        t_span=(0.0, 24000.0), n_points=3, rtol=1e-2, atol=1e-4
    )
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(1000.0, rel=1e-9)


def _chain(tmp_path, n):
    funcs = ["    1 dose() if(mod(time(), 24) < 1, 1, 0)", "    2 g0() 0.001*exp(-0.001*time())"]
    funcs += [
        f"    {i + 2} g{i}() 0.999*g{i - 1}() + 0.001*exp(-0.002*time())" for i in range(1, n)
    ]
    species = "\n".join(f"    {i + 2} S{i}() 1.0" for i in range(n))
    reactions = "\n".join(f"    {i + 2} 0 {i + 2} g{i} #_P{i}" for i in range(n))
    path = tmp_path / "chain.net"
    path.write_text(
        "begin parameters\n    1 one 1\nend parameters\n"
        f"begin species\n    1 X() 0\n{species}\nend species\n"
        f"begin functions\n{chr(10).join(funcs)}\nend functions\n"
        f"begin reactions\n    1 0 1 dose #_R1\n{reactions}\nend reactions\n",
        encoding="utf-8",
    )
    return bngsim.Model.from_net(str(path))


def test_a_rate_law_too_long_to_write_out_keeps_its_stops(tmp_path, monkeypatch):
    """A chain of 600 functions, each reading the one before, flattens past
    what the inliner writes out without sympy: it took 27 s to read where the
    run took 0.2, and main reads no rate law there. Not read, it is taken to
    change between its stops."""
    from bngsim import _switch_sensitivity

    model = _chain(tmp_path, 600)
    monkeypatch.setattr(
        _switch_sensitivity, "_write_flats", lambda core, made: pytest.fail("written out")
    )
    assert _switch_sensitivity._rate_laws_read_held(model._core, ()) is None
    assert _resolved(model, 0.0, 240.0, before_the_flip=True) == _resolved(model, 0.0, 240.0)


def test_a_model_s_rate_laws_are_written_out_once_over_its_legs(tmp_path, monkeypatch):
    """Written out on every leg, a loop of legs on a model of 1,000 rate laws
    took twice main's time; a regimen of 600 windows in one rate law, written
    out once a condition, 21.8 s for main's 1.5."""
    from bngsim import _switch_sensitivity

    written = []
    write = _switch_sensitivity._write_flats

    def counted(core, made):
        if not made[0]:
            written.append(core)
        return write(core, made)

    monkeypatch.setattr(_switch_sensitivity, "_write_flats", counted)
    monkeypatch.setattr(_switch_sensitivity, "_FLATS_BY_TEXT", {})
    sim = bngsim.Simulator(_chain(tmp_path, 50), method="ode")
    for leg in range(1, 6):
        r = sim.run_until(2400.0 * leg, rtol=1e-2, atol=1e-4)
    assert len(written) == 1
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(500.0, rel=1e-9)
