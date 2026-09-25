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
