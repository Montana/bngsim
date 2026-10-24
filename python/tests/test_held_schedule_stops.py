"""A schedule that reads a species only events write is stopped at stretch by
stretch (issue #1244).

``X' = piecewise(1, time - 24*floor(time/24) < w, 0)`` is a pulse of width w
at the start of each day, and an event at ``time >= 50`` sets w to 2. w is no
run constant, since an event writes it, so the resolvers placed no stop for
the condition at all: CVODE stepped over every pulse but the one the run
starts on, before the event as well as after, and X(240) came to 1 for 17 at
every tolerance, with nothing said.

Between two of its events w holds. Where the events fire at fixed times and
set values known before the run, each stretch between them is resolved with w
as its number there. Where they do not, a condition that changes more than
once is refused by name, and one that changes once is read as before.

Oracle: the pulse train integrated exactly, piece by piece between every edge
and every change of w.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

PULSE = "piecewise(1, (time - 24*floor(time/24)) < w, 0)"


def _model(events: str, law: str = PULSE, extra: str = "") -> bngsim.Model:
    return bngsim.Model.from_antimony_string(
        f"""
model m
  compartment C = 1
  species X in C
  X = 0
  w = 1
  a = 1
  w2 = 2
  {extra}
  J0: -> X; {law}
  {events}
end"""
    )


def _exact(changes, t_end: float, t_start: float = 0.0, w0: float = 1.0) -> float:
    """X(t_end) for a pulse of width w(t) at the start of each day, w set to
    each (t, w) of *changes* from t on."""
    ws = [(t_start, w0), *sorted(changes)]
    edges = {t_start, t_end} | {t for t, _ in ws}
    for day in range(int(t_start // 24), int(t_end // 24) + 2):
        edges |= {24.0 * day} | {24.0 * day + w for _, w in ws}
    edges = sorted(e for e in edges if t_start <= e <= t_end)
    total = 0.0
    for lo, hi in zip(edges, edges[1:], strict=False):
        mid = 0.5 * (lo + hi)
        w = [v for t, v in ws if t <= mid][-1]
        total += (hi - lo) * float(mid - 24 * np.floor(mid / 24) < w)
    return total


def _x(model, times, **kw) -> np.ndarray:
    result = bngsim.Simulator(model, method="ode", **kw.pop("sim", {})).run(
        sample_times=list(times), **kw
    )
    return np.asarray(result.species)[:, list(result.species_names).index("X")]


@pytest.mark.parametrize("rtol", [1e-2, 1e-6, 1e-10])
def test_the_issue_s_schedule_is_stopped_at(rtol):
    """3 pulses of width 1 by t = 50, then width 2 from 72 on: 3, 5 and 17,
    where main gave 1, 1 and 1."""
    x = _x(_model("E1: at (time >= 50): w = 2"), [0, 49, 74, 240], rtol=rtol, atol=rtol / 100)
    np.testing.assert_allclose(x, [0, 3, 5, 17], rtol=10 * rtol, atol=10 * rtol)


@pytest.mark.parametrize(
    ("events", "changes"),
    [
        ("E1: at 3 after (time >= 50): w = 2", [(53, 2)]),
        # inside the pulse at 72: it runs on to 74
        ("E1: at (time >= 72.5): w = 2", [(72.5, 2)]),
        ("E1: at (time > 50): w = 2", [(50, 2)]),
        ("E1: at (time >= 50): w = 2\n  E2: at (time >= 130): w = 0.5", [(50, 2), (130, 0.5)]),
        # a value that reads the species itself, as it holds then
        ("E1: at (time >= 50): w = w + 1\n  E2: at (time >= 130): w = w + 1", [(50, 2), (130, 3)]),
        # a run constant
        ("E1: at (time >= 50): w = w2", [(50, 2)]),
    ],
    ids=["delayed", "inside-a-pulse", "strict", "two-events", "reads-itself", "a-parameter"],
)
def test_each_stretch_is_stopped_at(events, changes):
    x = _x(_model(events), [0, 240], rtol=1e-8, atol=1e-10)
    assert x[-1] == pytest.approx(_exact(changes, 240), rel=1e-6)


@pytest.mark.parametrize(("initial", "w"), [("", 1.0), (", t0=false", 2.0)])
def test_a_run_that_starts_after_the_trigger(initial, w):
    """From t = 60 the trigger time >= 50 holds already: with initialValue
    true it never rises, and w stays 1; with false it fires at the start."""
    x = _x(_model(f"E1: at (time >= 50){initial}: w = 2"), [60, 240], rtol=1e-8, atol=1e-10)
    assert x[-1] - x[0] == pytest.approx(_exact([], 240, t_start=60, w0=w), rel=1e-6)


def test_a_sensitivity_run_is_stopped_at_too():
    """dX/da = X/a with the rate a times the pulse: 17 and 17, where main gave
    1 and 1."""
    for events in ("E1: at (time >= 50): w = 2", "E1: at (time >= 50): w = w2"):
        result = bngsim.Simulator(
            _model(events, law=f"a*{PULSE}"), method="ode", sensitivity_params=["a"]
        ).run(sample_times=[0, 240], rtol=1e-8, atol=1e-10)
        assert np.asarray(result.species)[-1, 0] == pytest.approx(17, rel=1e-6)
        assert np.asarray(result.sensitivities)[-1, 0, 0] == pytest.approx(17, rel=1e-5)


@pytest.mark.parametrize(
    ("events", "sens", "why"),
    [
        ("E1: at (X > 2): w = 2", None, "a trigger that is no fixed time"),
        ("E1: at (time >= 50): w = X", None, "sets it to X, which is not known before the run"),
        ("E1: at (time >= 50): w = w2", ["w2"], "which a requested sensitivity parameter moves"),
    ],
    ids=["a-state-trigger", "a-value-of-the-state", "a-requested-parameter"],
)
def test_a_schedule_not_known_before_the_run_is_refused(events, sens, why):
    """Main stepped over these as over the issue's: X(240) was 1, and dX/dw2
    2e-5 for 7."""
    sim = {"sensitivity_params": sens} if sens else {}
    with pytest.raises(bngsim.SimulationError, match="issue #1244") as e:
        _x(_model(events), [0, 240], rtol=1e-6, atol=1e-8, sim=sim)
    assert why in str(e.value)
    assert "reads w, and only events set w" in str(e.value)


def test_a_condition_that_changes_once_is_read_as_before():
    """``time < toff``, with toff set from 10 to 30 when X reaches 0.5: one
    change as the run starts, which the condition's root sees. Not refused,
    and as main: X(240) = 30."""
    x = _x(
        _model(
            "E1: at (X > 0.5): toff = 30", law="piecewise(1, time < toff, 0)", extra="toff = 10"
        ),
        [0, 240],
        rtol=1e-8,
        atol=1e-10,
    )
    assert x[-1] == pytest.approx(30, rel=1e-6)


def test_a_schedule_no_event_writes_is_as_before():
    x = _x(_model(""), [0, 49, 74, 240], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(x, [0, 3, 4, 10], rtol=1e-5)
