"""Issue #991: a trigger on the rateOf of a species whose own rate is a rateOf.

The right-hand side probes the rateOf buffer as deep as the derivatives read
it (``rateof_flow_nesting``). A trigger can read deeper: at
``rateOf(X0) > 0.3`` with ``X0' = rateOf(X3)`` reads a rateOf whose own rate is
a rateOf, which no derivative does. The root function, and the trigger reads at
the start, took the buffer from the right-hand side, one probe short, so the
trigger read X0's rate off whatever the buffer held before. A smooth crossing
was never found on the interpreted right-hand side and found late on the
compiled one, and a rate identically zero fired at t = 0 on a rate left there
by an evaluation at another state. The trajectory itself was right throughout.

Every expected value is a closed form.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

ENGINES = pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])

# G sets X5 = 1 at t = 0.25. Then X4 = 0.5 (t - 0.25), X3' = 3 X4 and
# X0' = rateOf(X3) = 1.5 (t - 0.25), which passes 0.3 at t = 0.45, and
# X0(1) = 0.75 * 0.75^2.
CROSSING = """compartment c = 1;
species X0 in c = 0, X3 in c = 0, X4 in c = 0, X5 in c = 0;
r0 := 1*rateOf(X3); X0' = r0;
r3 := 2*X4 + 1*X4; X3' = r3;
X4' = 0.5*X5;
X5' = 0;
w := rateOf(X0);
fired = 0; tf = -1; aG = 0.25;
E0: at (TRIGGER), t0=false: fired = fired + 1, tf = time;
G: at (time >= aG): X5 = 1;
"""

# B is 0 for good, so A, S and rateOf(S) are 0 at every time: E0 never fires.
# Y' = 1 makes Y a candidate clock, and the run's search for clocks evaluates
# the right-hand side once more with every species 1 higher
# (unit_rate_clock_candidates). That left A's rate there, B + 1 = 1, in the
# rateOf buffer, and the read at t = 0, one probe short, took it for rateOf(S).
# Y' = 0.3 or 2 is no candidate, and E0 did not fire.
AT_REST = """species S = 0, A = 0, B = 0, Y = 0;
S' = rateOf(A); A' = B; B' = 0; Y' = 1;
fired = 0; tf = -1;
E0: at (rateOf(S) > 0.5), t0=false: fired = fired + 1, tf = time;
"""


def _run(text: str, codegen: bool, sample_times: list[float]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_antimony_string(text)
        result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
            sample_times=sample_times, rtol=1e-9, atol=1e-12
        )
    last = dict(zip(result.observable_names, np.asarray(result.observables)[-1], strict=False))
    return last, result


@ENGINES
@pytest.mark.parametrize(
    "trigger",
    ["rateOf(X0) > 0.3", "w > 0.3", "rateOf(X3) > 0.3", "3*X4 > 0.3"],
    ids=["rateOf-of-a-rateOf", "through-a-rule", "one-deep", "no-rateOf"],
)
def test_every_spelling_fires_where_the_rate_crosses(trigger, codegen):
    """The first two never fired on the interpreted right-hand side, and on
    the compiled one fired at the next sample time, 0.46. The last two are the
    same crossing read one level up and with no rateOf at all."""
    last, result = _run(
        CROSSING.replace("TRIGGER", trigger), codegen, [0, 0.2, 0.3, 0.44, 0.46, 0.6, 1.0]
    )
    assert last["fired"] == 1.0
    assert last["tf"] == pytest.approx(0.45, abs=1e-6)
    x0 = np.asarray(result.species)[-1, list(result.species_names).index("X0")]
    assert x0 == pytest.approx(0.75 * 0.75**2, rel=1e-6)


@ENGINES
def test_a_rate_identically_zero_does_not_fire_at_the_start(codegen):
    """It fired at t = 0 on the rate the search for clocks left behind."""
    last, _ = _run(AT_REST, codegen, [0, 0.1, 0.2, 0.3, 0.5, 1.0])
    assert last["fired"] == 0.0


@ENGINES
def test_the_rate_rising_later_fires_once_where_it_rises(codegen):
    """B = 1 at t = 0.25 makes rateOf(S) = 1 from there: once, at 0.25. It
    fired at t = 0 and again at 0.25."""
    last, _ = _run(
        AT_REST + "G: at (time >= 0.25): B = 1;\n", codegen, [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    )
    assert last["fired"] == 1.0
    assert last["tf"] == pytest.approx(0.25, abs=1e-9)


# Three deep: S' = rateOf(A), A' = rateOf(B), B' = C, C' = D, and G sets D = 1
# at t = 0.25. Then C = t - 0.25 and rateOf(S) = rateOf(A) = rateOf(B) = C,
# which passes 0.3 at t = 0.55, and S(1) = 0.75^2 / 2. A derivative reads
# rateOf(A) two deep; the trigger reads rateOf(S) three deep.
CHAIN = """species S = 0, A = 0, B = 0, C = 0, D = 0;
S' = rateOf(A); A' = rateOf(B); B' = C; C' = D; D' = 0;
fired = 0; tf = -1;
E0: at (TRIGGER), t0=false: fired = fired + 1, tf = time;
G: at (time >= 0.25): D = 1;
"""


@ENGINES
@pytest.mark.parametrize(
    "trigger",
    ["rateOf(S) > 0.3", "rateOf(A) > 0.3", "C > 0.3"],
    ids=["three-deep", "two-deep", "no-rateOf"],
)
def test_a_chain_three_deep_fires_where_the_rate_crosses(trigger, codegen):
    """The first never fired on the interpreted right-hand side and fired at
    0.59 on the compiled one. The crossing is in mid-run, where only the root
    function reads the trigger; one that an event's write opens at a stop was
    already read at full depth there."""
    last, result = _run(CHAIN.replace("TRIGGER", trigger), codegen, [0, 0.2, 0.3, 0.5, 0.6, 1.0])
    assert last["fired"] == 1.0
    assert last["tf"] == pytest.approx(0.55, abs=1e-6)
    s = np.asarray(result.species)[-1, list(result.species_names).index("S")]
    assert s == pytest.approx(0.75**2 / 2, rel=1e-6)
