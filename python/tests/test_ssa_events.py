"""Tests for SBML event support under SSA (Phase 5b).

Mirrors the CVODE event tests in `test_events.py` for the cases that are
in scope for Phase 5b: EventNoDelay only. Delay-bearing events are gated
in C++ at SSA-run entry (delay support under SSA/PSA is issue #526); the
rejection is also exercised here.
"""

import math

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder, SsaSimulator, TimeSpec


def _ts(t_end, n_points, t_start=0.0):
    ts = TimeSpec()
    ts.t_start = t_start
    ts.t_end = t_end
    ts.n_points = n_points
    return ts


def test_ssa_event_time_trigger_fires_at_exact_time():
    """Time-only trigger must fire exactly when `time() >= 5`, not at the
    next τ-step boundary that crosses 5. With S(0)=100 and zero rate, the
    sample at t=5 must already show the post-event value 200.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 100.0)
    b.add_event("jump", "time() >= 5", [(s_idx, "200")], initial_value=False)
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=10.0, n_points=11), 42)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    # Pre-event samples (t<5): S=100. Post (t>=5): S=200.
    for i, ti in enumerate(t):
        if ti < 5.0 - 1e-9:
            assert S[i] == 100.0, f"t={ti}: expected 100 pre-event, got {S[i]}"
        else:
            assert S[i] == 200.0, f"t={ti}: expected 200 post-event, got {S[i]}"


def test_ssa_event_strict_time_trigger_leaves_the_row_at_its_threshold_before_it():
    """`time() > 5` is false at 5: the event fires just after it, so the
    sample at t=5 reads the pre-event 100, as ODE and libRoadRunner read
    it. The bisection placed the event within 1e-12 above 5, and the row,
    within the 1e-9 that a sample is taken as the event's own time,
    recorded 200. Without breakpoints, as a bare SsaSimulator runs."""
    b = ModelBuilder()
    s_idx = b.add_species("S", 100.0)
    b.add_event("jump", "time() > 5", [(s_idx, "200")], initial_value=False)
    model = b.build()

    result = SsaSimulator(model).run(_ts(t_end=10.0, n_points=11), 42)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]
    np.testing.assert_array_equal(S, np.where(t > 5.0, 200.0, 100.0))


# A row exactly at, and one 5e-10 before, the time a trigger on the clock
# turns true, beside the run's ends. Each loop of the engine: a model that no
# reaction changes (it idles to the event), one a reaction keeps firing, one
# with a rate rule, and one with a rate that reads the clock.
_ROWS_AT_THE_THRESHOLD = [0.0, 2.5 - 5e-10, 2.5, 5.0]
_LOOPS = {
    "idle": ("", "c = 3"),
    "firing": ("species Y in c = 0; J: -> Y; 4", "c = 3"),
    "rate rule": ("q = 0; q' = 1", "c = 3"),
    "rate on the clock": ("species Y in c = 0; J: -> Y; 1 + time", "S = 1"),
}


@pytest.mark.parametrize("method", ["ssa", "psa"])
@pytest.mark.parametrize("loop", list(_LOOPS))
@pytest.mark.parametrize(
    ("trigger", "at_threshold"), [("time > 2.5", 3.0), ("time >= 2.5", 1.0)], ids=[">", ">="]
)
def test_a_row_at_a_time_triggers_threshold_reads_the_side_ode_reads(
    trigger, at_threshold, loop, method
):
    """`E1: at time > 2.5: c = 3` beside `species S in c = 3`: the trigger is
    false at 2.5 itself, so nothing has fired there, and ODE and libRoadRunner
    read [S] = 3 in a row at 2.5. With `>=` the event fires at 2.5, and the row
    reads 1, as the SBML suite reads a row at an event's time (00928 at t = 0;
    DSMTS 00028 at 25). SSA and PSA read 1 in both rows, for both triggers,
    in a model with no rate rule and no rate on the clock, at any seed, and 1
    in the row 5e-10 before a `>=` event in the other loops where the step
    that found the event held that row too: a row within 1e-9 below an event
    was taken as the event's own time."""
    extra, assignment = _LOOPS[loop]
    ant = f"compartment c = 1; species S in c = 3; E1: at {trigger}: {assignment}\n{extra}\n"
    want = [3.0, 3.0, at_threshold, 1.0]

    def conc(method, **kw):
        model = bngsim.Model.from_antimony_string(ant)
        run = bngsim.Simulator(model, method=method, **kw).run(
            sample_times=_ROWS_AT_THE_THRESHOLD, seed=7
        )
        return np.asarray(run.as_roadrunner(["[S]"])["[S]"], dtype=float)

    np.testing.assert_allclose(conc("ode"), want, rtol=1e-12)
    np.testing.assert_array_equal(
        conc(method, **({"poplevel": 10} if method == "psa" else {})), want
    )


def test_ssa_event_state_dependent_trigger():
    """Trigger depends only on species state. State is piecewise-constant
    during τ, so the trigger flips only at fire boundaries — must be
    detected by the post-fire trigger sweep, not bisection.

    Model: ∅ → S at rate 1.0 (constant propensity, so S grows monotonically
    by +1 per fire). Event: when S >= 5, set S := 0. Without the event,
    E[S(20)] = 20. With it, S is reset to 0 every time it reaches 5, so
    every sample value is bounded above by 5.
    """
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    s_idx = b.add_species("S", 0.0)
    b.add_reaction([], [s_idx], "elementary", "k")
    b.add_event("threshold", "S >= 5", [(s_idx, "0")], initial_value=False)
    model = b.build()

    # Per-rep: assert S stays bounded and the event fires at least once
    # (detected via a sample-to-sample drop, which only happens on event
    # reset since the only reaction is a +1 birth).
    rebound_count = 0
    for seed in range(20):
        sim = SsaSimulator(model.clone())
        result = sim.run(_ts(t_end=20.0, n_points=21), seed)
        S = np.array(result.species_data)[:, 0]
        assert S.max() <= 5.0 + 1e-9, f"seed={seed}: S exceeded 5 → event missed"
        # Sample-to-sample drops imply a reset between samples.
        if (np.diff(S) < 0).any():
            rebound_count += 1
    assert rebound_count >= 15, f"only {rebound_count}/20 reps showed a post-event drop"


def test_ssa_event_initialvalue_false_fires_at_t0():
    """SBML L3 §3.4.5: initialValue=false ⇒ a t=0 expression that is true
    is a rising-edge fire AT t=0. Recorded value at t=0 must already be
    post-event.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 100.0)
    b.add_event("at_zero", "time() >= 0", [(s_idx, "200")], initial_value=False)
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=5.0, n_points=11), 1)
    S = np.array(result.species_data)[:, 0]
    assert S[0] == 200.0, f"S(0)={S[0]} — initial_value=False must fire at t=0"
    assert S[-1] == 200.0


def test_ssa_event_initialvalue_true_suppresses_t0_fire():
    """initialValue=true ⇒ presumed-prior trigger state is true, so a t=0
    expression that is true is NOT a transition. The trigger never falls
    here, so the event never fires — assignment must NOT apply.

    Compared to the CVODE companion test, this v1 SSA implementation
    detects only one false→true crossing per τ-step (true→false→true
    within a single τ is not detected); covering that corner case is a
    follow-up.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 100.0)
    b.add_event(
        "always_true",
        "time() >= 0",
        [(s_idx, "200")],
        initial_value=True,
    )
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=5.0, n_points=11), 1)
    S = np.array(result.species_data)[:, 0]
    assert S[0] == 100.0, "initial_value=True must suppress t=0 fire"
    # No reactions, no falling edge, no second rising edge: S stays at IC.
    assert S[-1] == 100.0


def test_ssa_event_priority_orders_simultaneous_fires():
    """Two events with the same trigger time. Per SBML L3: higher priority
    fires FIRST; the lower-priority write lands LAST and wins.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 0.0)
    b.add_event("low", "time() >= 1", [(s_idx, "1")], priority=1)
    b.add_event("high", "time() >= 1", [(s_idx, "2")], priority=10)
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=2.0, n_points=21), 1)
    S = np.array(result.species_data)[:, 0]
    assert S[-1] == 1.0, f"high-priority fires first, low writes last → expected 1.0, got {S[-1]}"


def test_ssa_event_priority_refresh_between_simultaneous_fires():
    """SBML L3 §3.4.6: priority is re-evaluated after each fire in a
    simultaneous batch. State-dependent priority must reorder dynamically.
    """
    b = ModelBuilder()
    s1 = b.add_species("S1", 5.0)
    s2 = b.add_species("S2", 0.0)
    b.add_species("dummy", 0.0)
    # Both fire at t>=1. e1 priority is "S1" (initially 5), e2 priority is "2*S2"
    # (initially 0). e1 wins first, sets S2=10. After re-eval: e2 priority becomes
    # 20 > S1 (5 still), so e2 fires next, overwriting S1 to 99.
    b.add_event("e1", "time() >= 1", [(s2, "10")], priority_expr="S1")
    b.add_event("e2", "time() >= 1", [(s1, "99")], priority_expr="2 * S2")
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=2.0, n_points=21), 1)
    species = np.array(result.species_data)
    assert species[-1, 0] == 99.0, f"S1 final={species[-1, 0]}, expected 99 (e2 fired second)"
    assert species[-1, 1] == 10.0, f"S2 final={species[-1, 1]}, expected 10"


def test_ssa_event_non_persistent_cancels_when_higher_priority_falsifies_trigger():
    """Non-persistent event whose trigger is falsified by a same-batch
    higher-priority fire must NOT fire.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 0.0)
    g_idx = b.add_species("guard", 1.0)
    # e1 (high priority): trigger time>=1, sets guard=0 → falsifies e2's trigger.
    # e2 (low priority, non-persistent): trigger "(time>=1) and (guard>0.5)";
    # would write S=99 but its trigger has gone false post-e1.
    b.add_event("e1", "time() >= 1", [(g_idx, "0")], priority=10)
    b.add_event(
        "e2",
        "(time() >= 1) and (guard > 0.5)",
        [(s_idx, "99")],
        priority=1,
        persistent=False,
    )
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=2.0, n_points=21), 1)
    species = np.array(result.species_data)
    assert species[-1, 0] == 0.0, f"S={species[-1, 0]} — e2 should be cancelled (guard fell)"
    assert species[-1, 1] == 0.0, "guard must be 0 (e1 fired)"


def test_ssa_event_persistent_fires_even_if_trigger_falsified():
    """Persistent event fires regardless of post-fire trigger state.
    Same construction as the non-persistent test but with persistent=True
    (the default).
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 0.0)
    g_idx = b.add_species("guard", 1.0)
    b.add_event("e1", "time() >= 1", [(g_idx, "0")], priority=10)
    b.add_event(
        "e2",
        "(time() >= 1) and (guard > 0.5)",
        [(s_idx, "99")],
        priority=1,
        persistent=True,
    )
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=2.0, n_points=21), 1)
    species = np.array(result.species_data)
    assert species[-1, 0] == 99.0, "persistent e2 must fire regardless"


def test_ssa_event_uses_values_from_trigger_time_default_true():
    """SBML L3 default UVFTT=true: assignment RHS is snapshotted at trigger
    time and applied at firing time.

    Two events fire at the same trigger time. e1 changes S; e2's RHS reads
    S. With UVFTT=true, e2 reads the trigger-time S (pre-e1), not the
    post-e1 value.
    """
    b = ModelBuilder()
    s1 = b.add_species("S", 5.0)
    target = b.add_species("target", 0.0)
    # e1 (high priority): S := 100. e2 (low priority): target := S.
    # With UVFTT=true (default) e2's snapshot is taken before any fire,
    # so target = trigger-time S = 5. With UVFTT=false e2 reads post-e1
    # state and target = 100.
    b.add_event("e1", "time() >= 1", [(s1, "100")], priority=10)
    b.add_event("e2", "time() >= 1", [(target, "S")], priority=1)
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=2.0, n_points=21), 1)
    species = np.array(result.species_data)
    assert species[-1, 1] == 5.0, (
        f"target={species[-1, 1]}: UVFTT=true must snapshot S at trigger time (5)"
    )


def test_ssa_event_uses_values_from_trigger_time_false():
    """UVFTT=false: assignment RHS evaluated against current state at
    firing time. Companion to the UVFTT=true test above.
    """
    b = ModelBuilder()
    s1 = b.add_species("S", 5.0)
    target = b.add_species("target", 0.0)
    b.add_event("e1", "time() >= 1", [(s1, "100")], priority=10)
    b.add_event(
        "e2",
        "time() >= 1",
        [(target, "S")],
        priority=1,
        use_values_from_trigger_time=False,
    )
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=2.0, n_points=21), 1)
    species = np.array(result.species_data)
    assert species[-1, 1] == 100.0, (
        f"target={species[-1, 1]}: UVFTT=false must read post-e1 S (100)"
    )


def test_ssa_event_re_arms_after_self_falsifying_assignment():
    """Trigger ``S < 5``; assignment ``S := 10``. The fire falsifies its own
    trigger. The next time S decays back below 5, the trigger must re-fire.
    """
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    s_idx = b.add_species("S", 10.0)
    # S → ∅ at rate k=1 → S decays roughly linearly under SSA mass action
    b.add_reaction([s_idx], [], "elementary", "k")
    b.add_event("rearm", "S < 5", [(s_idx, "10")], initial_value=False)
    model = b.build()

    # Run long enough for several decay-and-reset cycles; count fires by
    # detecting jumps from <5 to 10 in the trace.
    sim = SsaSimulator(model.clone())
    result = sim.run(_ts(t_end=50.0, n_points=501), 1)
    S = np.array(result.species_data)[:, 0]
    # Count rising jumps (>4 increase between adjacent samples)
    jumps = int(np.sum(np.diff(S) > 4))
    assert jumps >= 3, f"expected >=3 re-fires; got {jumps} (S trace max={S.max()}, min={S.min()})"


def test_ssa_event_with_delay_is_rejected():
    """Issue #526 tracks delay support; until then SSA must raise a clear
    error rather than silently ignoring the delay.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 0.0)
    b.add_event("delayed", "time() >= 1", [(s_idx, "100")], delay=0.5)
    model = b.build()

    sim = SsaSimulator(model)
    with pytest.raises(RuntimeError, match=r"delay.*not yet supported") as excinfo:
        sim.run(_ts(t_end=2.0, n_points=21), 1)
    _assert_the_rejection_points_somewhere_real(str(excinfo.value), "delayed")


def _assert_the_rejection_points_somewhere_real(message: str, event_id: str) -> None:
    """Issue #526: the message named the event and the remedy, and then sent the
    reader to "Phase 5c of the SBML SSA Support Plan", a planning document that
    is no longer in the tree. It points at the issue that tracks the work now."""
    assert f"Event '{event_id}'" in message
    assert "issue #526" in message
    assert "method='ode'" in message
    assert "Support Plan" not in message and "Phase 5c" not in message


def test_psa_event_with_delay_is_rejected_with_the_same_message():
    """PSA reaches the same run-entry check, so it says the same thing."""
    b = ModelBuilder()
    s_idx = b.add_species("S", 0.0)
    b.add_event("delayed", "time() >= 1", [(s_idx, "100")], delay=0.5)
    model = b.build()

    sim = SsaSimulator(model)
    with pytest.raises(RuntimeError, match=r"delay.*not yet supported") as excinfo:
        sim.run_psa(_ts(t_end=2.0, n_points=21), 1, 100.0, 0.0)
    _assert_the_rejection_points_somewhere_real(str(excinfo.value), "delayed")


def test_ssa_event_with_delay_expression_is_rejected():
    """Same gate as the static-delay case for state-dependent delay
    expressions.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 1.0)
    b.add_event("delayed_expr", "time() >= 1", [(s_idx, "100")], delay_expr="S")
    model = b.build()

    sim = SsaSimulator(model)
    with pytest.raises(RuntimeError, match=r"delay.*not yet supported") as excinfo:
        sim.run(_ts(t_end=2.0, n_points=21), 1)
    _assert_the_rejection_points_somewhere_real(str(excinfo.value), "delayed_expr")


def test_ssa_no_events_zero_overhead():
    """A model with zero events must run identically to pre-Phase-5b. Run
    a small SSA model; assert it completes and produces sane output. The
    "byte-identical roundtrip 7/7" check in the harness is the load-bearing
    regression test; this is a quick smoke check.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.5)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    model = b.build()

    sim = SsaSimulator(model)
    result = sim.run(_ts(t_end=10.0, n_points=11), 1)
    S = np.array(result.species_data)[:, 0]
    # After t=10 with k=0.5: E[S] = 100*exp(-5) ≈ 0.67 — should be small.
    assert S[0] == 100.0
    assert S[-1] < 10.0  # decayed substantially
    assert math.isfinite(S[-1])
