"""SSA/PSA event bisection terminates at any time scale (issue #716).

The event-crossing bisection stopped at an absolute ``hi - lo <= 1e-12``. From
t = 8192 one ulp is 2^-39 ≈ 1.8e-12, so ``hi - lo`` could never get there: the
loop spun forever at 100% CPU, and ``timeout=`` never fired because the wall-
clock budget is checked only in the outer loops. An event at t = 10000 hung SSA,
PSA and the no-reaction idle path; t = 8191.9 was fine. The tolerances are now
floored at a few ulps of the time they are applied at.

A regression here hangs rather than fails, so every case runs in a child
process with a hard cap. The sample grid puts an output time exactly on the
event, which also pins the sample-deferral tolerance at large t: that sample
must record the post-event state, as the ODE backend does.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import bngsim

    t0, method, idle = float(sys.argv[1]), sys.argv[2], sys.argv[3] == "idle"
    rxn = "" if idle else "J1: S => ; 0.0001*S; "
    m = bngsim.Model.from_antimony_string(
        "compartment C=1; species X in C=0; species S in C=10; "
        + rxn
        + f"E1: at (time >= {t0}): X = 1;"
    )
    kw = {"poplevel": 100} if method == "psa" else {}
    r = bngsim.Simulator(m, method=method, **kw).run(
        t_span=(0, 2 * t0), n_points=3, seed=1, timeout=10.0
    )
    x = np.asarray(r.species)[:, list(r.species_names).index("X")]
    print(json.dumps([float(v) for v in x]))
    """
)

# (t0, method, path). 8191.9 is the control the old code already handled.
_CASES = [
    (8191.9, "ssa", "rxn"),
    (10000.0, "ssa", "rxn"),
    (10000.0, "psa", "rxn"),
    (10000.0, "ssa", "idle"),
    (1e6, "ssa", "rxn"),
]


@pytest.mark.parametrize(("t0", "method", "path"), _CASES)
def test_an_event_at_large_time_fires_and_returns(t0, method, path):
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, repr(t0), method, path],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{method}/{path} with an event at t={t0:g} did not return in 60 s")
    assert proc.returncode == 0, proc.stderr[-2000:]
    x = json.loads(proc.stdout.strip().splitlines()[-1])
    assert x == [0.0, 1.0, 1.0]


_CHILD_FIRE_TIME = textwrap.dedent(
    """
    import sys
    import numpy as np
    import bngsim

    t0 = float(sys.argv[1])
    m = bngsim.Model.from_antimony_string(
        "compartment C=1; species X in C=0; species S in C=10; "
        "J1: S => ; 0.0001*S; "
        + f"E1: at (time >= {t0!r}): X = time;"
    )
    r = bngsim.Simulator(m, method="ssa").run(
        t_span=(0, 2 * t0), n_points=3, seed=1, timeout=10.0
    )
    print(repr(float(np.asarray(r.species)[-1, list(r.species_names).index("X")])))
    """
)


# The first three sit where a 2-ulp floor (2^-39 > 1e-12) would already have
# widened the old tolerance and fired those events one ulp late; the last two
# are past t = 8192, where the old loop never ended at all.
@pytest.mark.parametrize("t0", [4163.697195, 5026.437503, 5323.619904, 10000.123456, 123456.789])
def test_an_event_fires_on_the_first_representable_time_its_trigger_holds(t0):
    """``X := time`` records the firing time. ``time >= T0`` first holds at the
    double ``T0`` itself, and the bisection must land there bit for bit: below
    t = 8192 as it always has, and above it now that the loop runs to adjacent
    doubles instead of forever."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_FIRE_TIME, repr(t0)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"an event at t={t0!r} did not return in 60 s")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert float(proc.stdout.strip().splitlines()[-1]) == t0
