"""Issue #726: the cached time-condition scan follows the clock's rate.

A BNGL counter ``0 -> T`` at rate ``k_clock`` is a unit-rate clock at
``k_clock = 1`` and not otherwise, so ``Tobs >= on`` is a condition on time at
one value and a condition on state at another. The model cached the scan the
first time it was asked, and the state-root scan cached what it left out as
covered beside it. So a model whose clock rate moved after its first run kept
the first answer:

- first run at 0, then at 1: no crossing stop, and a 0.01-wide window was
  stepped over (issue #726 as filed; since #897 a state root catches it);
- first run at 1, then at 0.5: the thresholds stayed "covered" by a stop that
  no longer resolves at 0.5, so they had neither a stop nor a root, and the
  window was stepped over: Y = 0 for 10(1 - e^-2) = 8.6466, with no error.

Both caches are now keyed: the scan by what it reads from parameter values
(which species are clocks, and the live compartment size an amount-valued one
is read through), the roots by what the scan covers. ``X -> Y`` at rate 100
inside the window ``on <= Tobs < on + w``, from X = 10: Y = 10(1 - exp(-100 w
/ k_clock)) once the window has passed.
"""

from __future__ import annotations

import logging

import bngsim
import numpy as np
import pytest

_NET = """begin parameters
    1 k_clock 0
    2 on 50
    3 w 0.01
end parameters
begin functions
    1 rate_Y() if(Tobs>=on && Tobs<(on+w),100,0)
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 rate_Y
end reactions
begin groups
    1 Tobs 1
end groups
"""

SPAN = dict(t_span=(0.0, 200.0), n_points=3)
# (first clock rate, then this one): the window passes before t = 200 at both.
MOVES = [(0.0, 1.0), (1.0, 0.5), (0.5, 1.0)]
IDS = ["0-then-1", "1-then-half", "half-then-1"]


def _exact(k_clock: float) -> float:
    return 10.0 * (1.0 - np.exp(-100.0 * 0.01 / k_clock)) if k_clock > 0 else 0.0


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "clock_window.net"
    path.write_text(_NET)
    return str(path)


def _Y(result) -> float:
    return float(np.asarray(result.species)[-1, 2])


def _ran_at(net, k_clock):
    model = bngsim.Model.from_net(net)
    model.set_param("k_clock", k_clock)
    bngsim.Simulator(model, method="ode").run(**SPAN)
    model.reset()
    return model


def _fresh(net, k_clock):
    model = bngsim.Model.from_net(net)
    model.set_param("k_clock", k_clock)
    return model


# What each cache logs when it falls back to its best-effort answer: the
# recovery and its key at DEBUG, the roots at WARNING.
_FALLBACKS = ("unavailable", "detection failed")


@pytest.mark.parametrize(("first", "then"), MOVES, ids=IDS)
def test_set_param_after_a_run(net, first, then, caplog):
    with caplog.at_level(logging.DEBUG, logger="bngsim"):
        model = _ran_at(net, first)
        model.set_param("k_clock", then)
        got = _Y(bngsim.Simulator(model, method="ode").run(**SPAN))
    assert got == pytest.approx(_exact(then), rel=1e-5)
    fresh = _fresh(net, then)
    assert model.time_discontinuity_conditions() == fresh.time_discontinuity_conditions()
    assert model.state_switch_root_conditions() == fresh.state_switch_root_conditions()
    # Neither cache fell back to its best-effort path to get there.
    fallbacks = [r.getMessage() for r in caplog.records]
    assert not [m for m in fallbacks if any(f in m for f in _FALLBACKS)]


@pytest.mark.parametrize(("first", "then"), MOVES, ids=IDS)
def test_parameter_scan(net, first, then):
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    got = [_Y(r) for r in sim.parameter_scan("k_clock", [first, then], **SPAN)]
    assert got == pytest.approx([_exact(first), _exact(then)], rel=1e-5, abs=1e-12)


@pytest.mark.parametrize(("first", "then"), MOVES, ids=IDS)
def test_a_clone_of_a_model_that_ran(net, first, then):
    clone = _ran_at(net, first).clone()
    clone.set_param("k_clock", then)
    got = _Y(bngsim.Simulator(clone, method="ode").run(**SPAN))
    assert got == pytest.approx(_exact(then), rel=1e-5)


@pytest.mark.parametrize(("first", "then"), MOVES, ids=IDS)
def test_run_batch_after_the_parent_ran(net, first, then):
    model = _fresh(net, first)
    sim = bngsim.Simulator(model, method="ode")
    sim.run(**SPAN)
    model.reset()
    got = [_Y(r) for r in sim.run_batch(params=[{"k_clock": then}], **SPAN)]
    assert got == pytest.approx([_exact(then)], rel=1e-5)


def test_the_counter_is_a_time_condition_only_while_it_is_a_clock(net):
    # Not covered by a stop, the thresholds are rooted on; covered, they are not.
    model = bngsim.Model.from_net(net)
    window = ("Tobs>=on", "Tobs<(on+w)")
    for k_clock, conditions, roots in [
        (1.0, window, ()),
        (0.5, (), window),
        (1.0, window, ()),
        (0.0, (), window),
    ]:
        model.set_param("k_clock", k_clock)
        assert model.time_discontinuity_conditions() == conditions
        assert model.state_switch_root_conditions() == roots


def _counting(monkeypatch, name):
    """Record each call of ``_switch_sensitivity.<name>``, which the model looks
    up at call time, and let it through."""
    from bngsim import _switch_sensitivity

    calls = []
    real = getattr(_switch_sensitivity, name)

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(_switch_sensitivity, name, counted)
    return calls


@pytest.mark.parametrize("scan", ["time_discontinuity_conditions", "state_switch_root_conditions"])
def test_an_interrupted_rescan_is_made_again(net, monkeypatch, scan):
    """Each key is written after the answer it keys. Written first, a rescan
    cut short (Ctrl-C in a notebook, a KeyboardInterrupt in a worker) left the
    1.0 answer under the 0.5 key, and every later ask read it back: neither a
    stop nor a root, and Y = 0 for 8.6466 with no error."""
    from bngsim import _switch_sensitivity

    model = _fresh(net, 1.0)
    assert model.state_switch_root_conditions() == ()
    model.set_param("k_clock", 0.5)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(_switch_sensitivity, scan, interrupted)
        with pytest.raises(KeyboardInterrupt):
            model.state_switch_root_conditions()
    assert model.time_discontinuity_conditions() == ()
    assert model.state_switch_root_conditions() == ("Tobs>=on", "Tobs<(on+w)")
    got = _Y(bngsim.Simulator(model, method="ode").run(**SPAN))
    assert got == pytest.approx(_exact(0.5), rel=1e-5)


@pytest.mark.parametrize(
    ("method", "store"),
    [
        ("time_discontinuity_conditions", "self._derived_time_disc_clocks = key"),
        ("_state_switch_roots_beside", "self._state_switch_root_covered = covered"),
    ],
    ids=["conditions", "roots"],
)
def test_an_answer_written_but_not_keyed_is_not_read(net, method, store):
    """Interrupted after its answer is written and before its key is, a slot
    holds a key that matches nothing, so the next ask derives again. Holding
    the previous key there, the 1.0 answer was read back at 0.5: Y = 0 for
    8.6466. (A signal lands between bytecodes; a line trace puts it there.)"""
    import inspect
    import sys

    from bngsim import _model

    lines, start = inspect.getsourcelines(getattr(_model.Model, method))
    at = [start + i for i, line in enumerate(lines) if line.strip() == store]
    assert len(at) == 1

    def trace(frame, event, arg):
        if frame.f_code.co_name != method:
            return None

        def line(frame, event, arg):
            if event == "line" and frame.f_lineno == at[0]:
                sys.settrace(None)
                raise KeyboardInterrupt
            return line

        return line

    model = _fresh(net, 0.5)
    assert model.state_switch_root_conditions() == ("Tobs>=on", "Tobs<(on+w)")
    model.set_param("k_clock", 1.0)
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            model.state_switch_root_conditions()
    finally:
        sys.settrace(None)
    model.set_param("k_clock", 0.5)
    assert model.time_discontinuity_conditions() == ()
    assert model.state_switch_root_conditions() == ("Tobs>=on", "Tobs<(on+w)")
    got = _Y(bngsim.Simulator(model, method="ode").run(**SPAN))
    assert got == pytest.approx(_exact(0.5), rel=1e-5)


def test_a_clone_that_moves_another_rate_does_not_rescan(net, monkeypatch):
    """A fit over a rate constant does not move which species are clocks, so a
    fan-out of clones over one reuses the parent's scans: the key is the probe,
    not the parameter values."""
    model = _fresh(net, 1.0)
    conditions = model.time_discontinuity_conditions()
    roots = model.state_switch_root_conditions()
    assert conditions
    clone = model.clone()
    clone.set_param("w", 0.02)
    scans = _counting(monkeypatch, "time_discontinuity_conditions")
    roots_scans = _counting(monkeypatch, "state_switch_root_conditions")
    assert clone.time_discontinuity_conditions() == conditions
    assert clone.state_switch_root_conditions() == roots
    assert scans == [] and roots_scans == []


def test_a_batch_at_another_verdict_scans_it_once(net, monkeypatch):
    """Rows whose clock rate sits at another verdict than the parent's share
    one scan of it: the memo is the model's and its clones'. Kept in one slot
    per model, every row scanned again, and so did every later batch."""
    model = _fresh(net, 1.0)
    sim = bngsim.Simulator(model, method="ode")
    sim.run(**SPAN)
    model.reset()
    scans = _counting(monkeypatch, "time_discontinuity_conditions")
    roots_scans = _counting(monkeypatch, "state_switch_root_conditions")
    for _ in range(2):
        rows = sim.run_batch(params=[{"k_clock": 0.5}] * 4, num_processors=1, **SPAN)
        assert [_Y(r) for r in rows] == pytest.approx([_exact(0.5)] * 4, rel=1e-5)
    assert len(scans) == 1
    assert len(roots_scans) == 1


def test_a_run_probes_for_its_clocks_once(net, monkeypatch):
    """The stops and the roots of one run read one key: the roots are asked
    beside the conditions the stops were placed for, not after a second probe
    of the right-hand side."""
    model = _fresh(net, 1.0)
    sim = bngsim.Simulator(model, method="ode")
    sim.run(**SPAN)
    model.reset()
    keys = []
    real = bngsim.Model._clock_key

    def counted(self):
        keys.append(self)
        return real(self)

    monkeypatch.setattr(bngsim.Model, "_clock_key", counted)
    assert _Y(sim.run(**SPAN)) == pytest.approx(_exact(1.0), rel=1e-5)
    assert len(keys) == 1


def test_the_key_and_the_scan_read_one_text_gate(net, monkeypatch):
    """Where no text spells a switch the scan returns () unprobed, and the key
    that keeps it current does not probe either. One gate for both, so a
    construct the scan learns to read cannot leave the key blind to it."""
    from bngsim import _switch_sensitivity

    model = _fresh(net, 1.0)
    monkeypatch.setattr(_switch_sensitivity, "texts_spell_a_switch", lambda core: False)
    probes = _counting(monkeypatch, "unit_rate_clock_candidates")
    # The scan on its own, so that its gate is what decides...
    assert _switch_sensitivity.time_discontinuity_conditions(model._core) == ()
    # ...and through the model, whose key reads the same gate and so never probes.
    assert model.time_discontinuity_conditions() == ()
    assert probes == []


def test_a_probe_that_fails_degrades_rather_than_raising(net, monkeypatch):
    """The key's probe is part of the best-effort recovery: a failure there
    leaves the model without the stops, as a failed scan does, and the run
    goes on, rooted on the window instead of stopping at it."""
    from bngsim import _switch_sensitivity

    def broken(core):
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(_switch_sensitivity, "unit_rate_clock_candidates", broken)
    model = _fresh(net, 1.0)
    assert model.time_discontinuity_conditions() == ()
    assert model.state_switch_root_conditions() == ("Tobs>=on", "Tobs<(on+w)")
    got = _Y(bngsim.Simulator(model, method="ode").run(**SPAN))
    assert got == pytest.approx(_exact(1.0), rel=1e-5)


# A substance-only counter fed at its compartment's size: the stored value moves
# at 1 at every size, so it passes the probes at every size, but a rate law reads
# its amount, size * stored, which is a unit-rate clock at size 1 only. X -> Y at
# 100 X inside the windows 50 + 100n <= T < 50.01 + 100n of a floor() pulse: at
# size 1, T = t, and Y = 10(1 - e^-2) once the windows at 50 and 150 have
# passed. The SBML loader registers a root where each floor()'s argument
# changes sign, which is the window at 50; at 150 the argument passes 1, and
# the stop recovered for the floor() jump is all that catches it.
_SIZED_COUNTER = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sized_counter">
    <listOfCompartments>
      <compartment id="C" size="2" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="T" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="X" compartment="C" initialAmount="10" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="Y" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="tick" reversible="false">
        <listOfProducts>
          <speciesReference species="T" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> C </ci></math>
        </kineticLaw>
      </reaction>
      <reaction id="pulse" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Y" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn> 100 </cn><ci> X </ci>
            <apply><minus/>
              <apply><floor/><apply><divide/>
                <apply><minus/><ci> T </ci><cn> 50 </cn></apply><cn> 100 </cn></apply></apply>
              <apply><floor/><apply><divide/>
                <apply><minus/><ci> T </ci><cn> 50.01 </cn></apply><cn> 100 </cn></apply></apply>
            </apply>
          </apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_a_compartment_resize_moves_the_clock_verdict(tmp_path):
    """The key holds whether the live compartment size an amount-valued
    candidate is read through is 1. Keyed on the probes alone, a model first
    run at size 2 (no clock: the amount moves at 2) and then at size 1 kept the
    size-2 scan, had no stop at the second window, and stepped over it: Y =
    6.3212 for 8.6466, with no error."""
    path = tmp_path / "sized_counter.xml"
    path.write_text(_SIZED_COUNTER)
    span = dict(t_span=(0.0, 160.0), n_points=3)
    model = bngsim.Model.from_sbml(str(path))

    def floor_stops(m):
        # The loader registers the floor arguments' sign changes itself; the
        # floor jumps are what the recovery adds, and only on a clock.
        return [c for c in m.time_discontinuity_conditions() if c.startswith("floor(")]

    assert floor_stops(model) == []
    bngsim.Simulator(model, method="ode").run(**span)
    model.reset()
    model.set_param("C", 1.0)
    got = _Y(bngsim.Simulator(model, method="ode").run(**span))
    assert got == pytest.approx(10.0 * (1.0 - np.exp(-2.0)), rel=1e-5)
    fresh = bngsim.Model.from_sbml(str(path))
    fresh.set_param("C", 1.0)
    assert len(floor_stops(fresh)) == 2
    assert model.time_discontinuity_conditions() == fresh.time_discontinuity_conditions()
