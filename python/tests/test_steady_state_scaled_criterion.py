"""``steady_state(criterion="scaled")`` (issue #1001).

``||f(y)||_2 / n < tol`` is a number in the model's units of concentration per
time: a model in small units is under it where it starts, or one step later,
and that state came back ``converged=True``. ``criterion="scaled"`` accepts a
state when the Newton step from it to the root is within each species' own
tolerance, which reads the same in any units. These tests cover:

- the issue's cubic at scales 1e-9 and 1e-12 ends at its stable root, by both
  methods, from either side of it;
- the default is the scaled test, and the residual test is kept by name;
- a model with a conservation law converges to the residual test's answer;
- an ``atol`` the caller gives is the one the march takes;
- ``steady_state_batch`` passes the criterion on;
- an unknown criterion is refused;
- the reported step is the Newton step NumPy takes on the reduced system, with
  laws, a dependent smaller than its unknowns, and a mask;
- a model that starts with every species at 0 is held to its own size, not to
  1 in its units;
- a root at which the Jacobian is singular (a catalyst at 0) is converged;
- ``set_tolerances(rtol=...)`` alone leaves the march its own ``atol``;
- above 512 unknowns the solve is the residual one, step for step, and a
  run taken on from it decides;
- a march that stops between the steps the criterion is asked at is asked
  where it stops (a corpus model);
- a mask says which species are measured, not which move: A -> S under the
  pure-sink mask is stepped, A -> B and A -> C converge, a masked C that
  drains steps the A and B it feeds, whatever the species order;
- a differenced Jacobian across a rate law's jump certifies nothing;
- a $-fixed species does not set the floor;
- a batch entry at another scale marches at its own "auto" atol.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest

# X' = -k·x·(x - a)·(x - u), x = X/s: roots 0 and u = 1 stable, a = 0.3 not.
CUBIC_NET = """begin parameters
    1 k   1.0
    2 a   0.3
    3 u   1.0
    4 s   {s!r}
    5 X0  {x0!r}*s
    6 k3  k/(s*s)
    7 k2  k*(u+a)/s
    8 k1  k*a*u
end parameters
begin species
    1 X() X0
end species
begin reactions
    1 1,1,1 1,1 k3
    2 1,1 1,1,1 k2
    3 1 0 k1
end reactions
"""


def _cubic(tmp_path, s, x0):
    p = tmp_path / f"cubic_{s!r}_{x0!r}.net"
    p.write_text(CUBIC_NET.format(s=s, x0=x0))
    return bngsim.Model.from_net(str(p))


@pytest.mark.parametrize("s", [1e-9, 1e-12])
@pytest.mark.parametrize("x0", [0.5, 0.9, 1.5])
@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("by_name", [False, True], ids=["default", "named"])
def test_scaled_reaches_the_stable_root_at_any_scale(tmp_path, s, x0, method, by_name):
    kw = {"criterion": "scaled"} if by_name else {}
    ss = bngsim.Simulator(_cubic(tmp_path, s, x0), method="ode").steady_state(method=method, **kw)
    assert ss.converged
    assert ss.criterion == "scaled"
    assert ss.concentrations[0] / s == pytest.approx(1.0, rel=1e-6)
    assert ss.newton_distance <= 1.0
    assert ss.continued_run_distance <= 1.0


def test_the_residual_test_is_kept_by_name(tmp_path):
    # The parity test, unchanged: it takes no Newton step (a Jacobian more per
    # solve) and no run on, so it reports neither.
    at_one = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode").steady_state(
        criterion="residual"
    )
    assert at_one.criterion == "residual"
    assert at_one.converged
    assert at_one.concentrations[0] == pytest.approx(1.0, rel=1e-6)
    assert math.isnan(at_one.newton_distance)
    assert math.isnan(at_one.continued_run_distance)


# The issue's table: the residual test at a scale of 1e-9 accepts the cubic one
# step from each start. Pinned so a change to the residual path shows.
RESIDUAL_AT_1E_9 = {
    ("integration", 0.5): 0.893865,
    ("integration", 0.9): 1.003209,
    ("integration", 1.5): 0.130999,
    ("newton", 0.5): 0.3,
    ("newton", 0.9): 1.038462,
    ("newton", 1.5): 1.214286,
}


@pytest.mark.parametrize("method,x0", sorted(RESIDUAL_AT_1E_9))
def test_the_residual_solve_is_the_one_main_took(tmp_path, method, x0):
    residual = bngsim.Simulator(_cubic(tmp_path, 1e-9, x0), method="ode").steady_state(
        method=method, criterion="residual"
    )
    assert residual.criterion == "residual"
    assert residual.converged and residual.n_steps == 1
    assert residual.concentrations[0] / 1e-9 == pytest.approx(
        RESIDUAL_AT_1E_9[(method, x0)], rel=1e-5
    )


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_scaled_with_a_conservation_law(data_dir, method):
    path = str(data_dir / "two_species_reversible.net")
    ref = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(
        method=method, tol=1e-14
    )
    ss = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(
        method=method, criterion="scaled"
    )
    assert ss.converged
    assert ss.newton_distance <= 1.0
    for got, want in zip(ss.concentrations, ref.concentrations, strict=False):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-12)


def test_a_given_atol_is_the_one_the_march_takes(tmp_path):
    # At the scalar 1e-8 the march runs a species of 1e-9 through zero; the
    # criterion then does not call where it stops converged.
    ss = bngsim.Simulator(_cubic(tmp_path, 1e-9, 0.5), method="ode").steady_state(
        criterion="scaled", atol=1e-8
    )
    assert not ss.converged
    assert ss.newton_distance > 1.0


def test_batch_passes_the_criterion_on(tmp_path):
    sim = bngsim.Simulator(_cubic(tmp_path, 1e-9, 0.5), method="ode")
    out = sim.steady_state_batch([{}, {"a": 0.2}], criterion="scaled")
    for ss in out:
        assert ss.converged
        assert ss.criterion == "scaled"
        assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


def test_an_unknown_criterion_is_refused(tmp_path):
    sim = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode")
    with pytest.raises((ValueError, bngsim.SimulationError), match="criterion"):
        sim.steady_state(criterion="relative")


# A + B <-> C, C -> D: two laws (A + C + D, B + C + D), so one unknown and three
# species that follow from it, and D a sink a mask can leave out.
BINDING_NET = """begin parameters
    1 kf 1.0
    2 kr 0.01
end parameters
begin species
    1 A() {a0!r}
    2 B() {b0!r}
    3 C() {c0!r}
    4 D() 0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
    3 3 4 kr
end reactions
"""


def _numpy_newton_distance(model, y, rtol, mask=None):
    """The step the criterion measures, by NumPy, and without choosing which
    species each law is solved for. Every species moves by its own equation
    but a $-fixed one and a masked one nothing reads (an empty column of the
    Jacobian's pattern), which is held; the laws are cut down to the ones no
    such species is in; the step is solved on all of it, with a rank check.
    Its size is taken over each kept species' rtol·|y| + rtol·max(|y|, |y0|,
    floor), y0 the state the solve started from (the model's, here) and the
    floor the smallest of it above 0."""
    ns = len(y)
    cl = model.conservation_laws
    laws = np.asarray(cl["coefficients"], float).reshape(cl["n_laws"], ns)
    fixed = np.asarray(model._core.species_is_fixed, bool)
    kept = (np.ones(ns, bool) if mask is None else np.asarray(mask, bool)) & ~fixed
    pattern = model.jacobian(np.asarray(model.get_state(), float), sparse=True).tocsc()
    unread = ~kept & ~fixed & (np.diff(pattern.indptr) == 0)
    if unread.any() and len(laws):
        u, sv, _ = np.linalg.svd(laws[:, unread])
        laws = u[:, int(np.sum(sv > 1e-12 * sv[0])) :].T @ laws
    moves = ~fixed & ~unread
    f = np.asarray(model.rhs(y), float)
    jac = np.asarray(model.jacobian(y), float)
    lhs = np.vstack([laws[:, moves], jac[moves][:, moves]])
    rhs = np.concatenate([np.zeros(len(laws)), -f[moves]])
    step, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    assert rank == moves.sum()
    assert np.allclose(lhs @ step, rhs, rtol=0, atol=1e-9 * np.max(np.abs(rhs)))
    delta = np.zeros(ns)
    delta[moves] = step
    y0 = np.asarray(model.get_state(), float)
    floor = min(abs(v) for v, keep in zip(y0, kept, strict=True) if keep and v != 0)
    scale = rtol * np.abs(y) + rtol * np.maximum(np.maximum(np.abs(y), np.abs(y0)), floor)
    return float(np.max(np.abs(delta[kept]) / scale[kept]))


@pytest.mark.parametrize("a0,b0,c0", [(1.0, 1e-3, 0.0), (1e-9, 1e-12, 1e-10), (5.0, 2.0, 0.5)])
@pytest.mark.parametrize("mask", [None, [True, True, True, False]], ids=["all", "sink-out"])
def test_the_reported_step_is_the_newton_step(tmp_path, a0, b0, c0, mask):
    # Stopped short of the root (max_time 0.5), where the step is large and
    # every species' share of it shows.
    path = tmp_path / "binding.net"
    path.write_text(BINDING_NET.format(a0=a0, b0=b0, c0=c0))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", max_time=0.5, mask=mask
    )
    assert not ss.converged
    want = _numpy_newton_distance(
        bngsim.Model.from_net(str(path)), np.asarray(ss.concentrations), 1e-8, mask
    )
    assert ss.newton_distance == pytest.approx(want, rel=1e-6)


# 0 -> X at ksyn, X -> 0 at 1: X starts at 0 and ends at ksyn.
SYNTHESIS_NET = """begin parameters
    1 ksyn {ksyn!r}
    2 kdeg 1.0
end parameters
begin species
    1 X() 0
end species
begin reactions
    1 0 1 ksyn
    2 1 0 kdeg
end reactions
"""


@pytest.mark.parametrize("ksyn", [1e-9, 1e-12, 1e3])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_empty_start_is_held_to_its_own_size(tmp_path, ksyn, method):
    # With nothing above 0 at the start the floor was 1 in the model's units, a
    # tolerance of 1e-8 against a root of 1e-9: accepted at x = 0.69e-9 after
    # one step.
    path = tmp_path / "synthesis.net"
    path.write_text(SYNTHESIS_NET.format(ksyn=ksyn))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.concentrations[0] / ksyn == pytest.approx(1.0, rel=1e-6)
    assert ss.newton_distance <= 1.0


# E + S -> E + P with no E: nothing moves, and the reduced Jacobian, k·E, is 0.
CATALYST_NET = """begin parameters
    1 k 1.0
end parameters
begin species
    1 E() 0
    2 S() 1e-9
    3 P() 0
end species
begin reactions
    1 1,2 1,3 k
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_root_with_a_singular_jacobian_is_converged(tmp_path, method):
    path = tmp_path / "catalyst.net"
    path.write_text(CATALYST_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.newton_distance == 0.0
    assert list(ss.concentrations) == [0.0, 1e-9, 0.0]


def test_set_tolerances_without_atol_leaves_the_march_its_own(tmp_path):
    # set_tolerances(rtol=...) gave no atol, but it set the scalar 1e-8, which
    # took the march off "auto" and ran X through zero (x = -21.8).
    sim = bngsim.Simulator(_cubic(tmp_path, 1e-9, 0.5), method="ode")
    sim.set_tolerances(rtol=1e-8)
    ss = sim.steady_state(criterion="scaled")
    assert ss.converged
    assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


def test_the_distance_is_measured_on_the_retried_jacobian(data_dir):
    # The closed form stops this march and the solve is retried on
    # differences; measured on the closed form, the distance of the root the
    # retry accepted read inf beside converged=True.
    ss = bngsim.Simulator(
        bngsim.Model.from_net(str(data_dir / "nnxor_parked_residual.net")), method="ode"
    ).steady_state(criterion="scaled", method="newton")
    assert ss.solver_jacobian_retried
    assert ss.converged
    assert ss.newton_distance <= 1.0


def _wide_net(copies):
    # `copies` pairs A <-> B, fed and drained: 2·copies unknowns, no laws.
    species = []
    reactions = []
    for i in range(copies):
        a, b = 2 * i + 1, 2 * i + 2
        species += [f"    {a} A{i}() {1.0 + 0.01 * i!r}", f"    {b} B{i}() 0"]
        reactions += [
            f"    {4 * i + 1} {a} {b} kf",
            f"    {4 * i + 2} {b} {a} kf",
            f"    {4 * i + 3} 0 {a} ks",
            f"    {4 * i + 4} {b} 0 ks",
        ]
    return "\n".join(
        ["begin parameters", "    1 kf 1000.0", "    2 ks 1.0", "end parameters"]
        + ["begin species", *species, "end species"]
        + ["begin reactions", *reactions, "end reactions", ""]
    )


def test_above_512_unknowns_the_solve_is_the_residual_one(tmp_path):
    # The step is not taken there, and the solve stops where the residual one
    # does: with KINSOL's stopping rule left at the scaled one, the polish ran
    # on past the residual's answer (BaruaBCR_2012 was never accepted, 18 s
    # against 2 s). A run taken on from the state decides.
    path = tmp_path / "wide.net"
    path.write_text(_wide_net(260))
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="newton", atol=1e-10)
    residual, scaled = out["residual"], out["scaled"]
    assert scaled.converged and scaled.method_used == "newton"
    assert scaled.criterion == "scaled"
    assert math.isnan(scaled.newton_distance)
    assert scaled.continued_run_distance <= 1.0
    assert scaled.n_steps == residual.n_steps
    assert np.array_equal(scaled.concentrations, residual.concentrations)


_B252 = (
    Path(__file__).resolve().parents[2]
    / "parity_checks/rr_parity/models/BIOMD0000000252/BIOMD0000000252_url.xml"
)


@pytest.mark.skipif(
    not _B252.exists(), reason="rr_parity corpus model BIOMD0000000252 not present"
)
def test_a_march_that_stalls_between_asks_is_asked_where_it_stops():
    # The march stalls at step 5411, between the asks at 4096 and 8192, at a
    # state within its tolerance, and came back unconverged.
    ss = bngsim.Simulator(bngsim.Model.from_sbml(str(_B252)), method="ode").steady_state(
        criterion="scaled"
    )
    assert ss.converged
    assert ss.newton_distance <= 1.0


@pytest.mark.parametrize("s", [1.0, 1e-9])
def test_burst_free_starts_find_the_same_roots_at_any_scale(tmp_path, s):
    # The solve steady_states() runs from each start (issue #524). At 1e-9 the
    # residual test calls most starts roots where they stand; the scaled one
    # finds the cubic's three roots, as at 1. The one at 0 is reached as
    # -5.9e-24, which the negativity check, relative to the largest species
    # and so to itself, threw out until it was given the species' tolerance.
    sim = bngsim.Simulator(_cubic(tmp_path, s, 0.5), method="ode")
    starts = (np.linspace(0.01, 1.6, 40) * s)[:, None]
    out = sim.steady_state_batch(
        states=starts, method="newton", newton_burst=False, criterion="scaled"
    )
    converged = [r for r in out if r.converged]
    assert all(r.newton_distance <= 1.0 for r in converged)
    roots = sorted({round(r.concentrations[0] / s, 6) + 0.0 for r in converged})
    assert roots == [0.0, 0.3, 1.0]


# A -> S, S a write-only sink, one law A + S. Listed A first, A is the law's
# dependent species, which the pure-sink mask keeps while it leaves S out.
DECAY_NETS = {
    "A first": """begin parameters
    1 k 1.0
end parameters
begin species
    1 A() 1.0
    2 S() 0
end species
begin reactions
    1 1 2 k
end reactions
""",
    "S first": """begin parameters
    1 k 1.0
end parameters
begin species
    1 S() 0
    2 A() 1.0
end species
begin reactions
    1 2 1 k
end reactions
""",
}


@pytest.mark.parametrize("order", sorted(DECAY_NETS))
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_masked_sink_takes_up_its_law(tmp_path, order, method):
    # The solvers hold S where integration left it; with A the law's dependent,
    # nothing moved A, and A first was accepted where it started, [1, 0].
    path = tmp_path / "decay.net"
    path.write_text(DECAY_NETS[order])
    model = bngsim.Model.from_net(str(path))
    a = model.species_names.index("A()")
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", method=method, mask=~model.is_pure_sink()
    )
    assert ss.converged
    assert abs(ss.concentrations[a]) < 1e-8
    assert ss.newton_distance <= 1.0


@pytest.mark.parametrize("order", sorted(DECAY_NETS))
def test_a_masked_law_member_follows_the_step(tmp_path, order):
    # Short of the root, the step is A's own: f = -k·A, so δ_A = -A exactly,
    # held to rtol·A + rtol·max(A, 1).
    path = tmp_path / "decay.net"
    path.write_text(DECAY_NETS[order])
    model = bngsim.Model.from_net(str(path))
    a = model.species_names.index("A()")
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", max_time=0.5, mask=~model.is_pure_sink()
    )
    assert not ss.converged
    y = ss.concentrations[a]
    assert ss.newton_distance == pytest.approx(y / (1e-8 * y + 1e-8 * max(y, 1.0)), rel=1e-6)


# A <-> B, one law; a mask that keeps only A, the law's dependent.
PAIR_NET = """begin parameters
    1 kf 1.0
    2 kr 0.5
end parameters
begin species
    1 A() 1.0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_kept_dependent_whose_partner_is_masked_is_stepped(tmp_path, method):
    path = tmp_path / "pair.net"
    path.write_text(PAIR_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method, mask=[True, False]
    )
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(1.0 / 3.0, rel=1e-6)


# X' = if(X < 1, 0.1, 100): positive everywhere, no steady state.
JUMP_NET = """begin parameters
    1 klo 0.1
    2 khi 100
    3 x0 {x0!r}
end parameters
begin species
    1 X() x0
end species
begin reactions
    1 0 1 fr
end reactions
begin groups
    1 Xobs 1
end groups
begin functions
    1 fr() if(Xobs<1,klo,khi)
end functions
"""


@pytest.mark.parametrize("jacobian", ["fd", "auto"])
def test_a_differenced_jacobian_across_a_jump_certifies_nothing(tmp_path, jacobian):
    # The march stops on the switch, and one-sided across it the quotient is
    # the jump over the step: 20 of these 60 starts came back converged.
    for i, x0 in enumerate(np.linspace(0.0, 0.999, 60)):
        path = tmp_path / f"jump_{i}.net"
        path.write_text(JUMP_NET.format(x0=float(x0)))
        ss = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
        ).steady_state(criterion="scaled", max_time=100.0)
        assert not ss.converged, x0


@pytest.mark.parametrize("s", [1.0, 1e-9])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_differenced_jacobian_still_certifies_a_smooth_root(tmp_path, s, method):
    ss = bngsim.Simulator(_cubic(tmp_path, s, 0.5), method="ode", jacobian="fd").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.concentrations[0] / s == pytest.approx(1.0, rel=1e-6)


def _ltype(tmp_path, data_dir, k_v_stim=None):
    source = (data_dir / "ltype_calcium_discontinuous_jacobian.net").read_text()
    if k_v_stim is not None:
        source = re.sub(r"(16 k_v_stim\s+)\S+", rf"\g<1>{k_v_stim}", source)
    path = tmp_path / f"ltype_{k_v_stim}.net"
    path.write_text(source)
    return bngsim.Model.from_net(str(path))


def test_the_retried_march_past_a_switch_converges_away_from_it(tmp_path, data_dir):
    # GH #176's model: its closed-form Jacobian stops the march, the retry
    # differences across the switch on the way. Voltage relaxes to
    # k_v_stim/k_v_leak; at 45 it rests 5 below the switch at 50.
    ss = bngsim.Simulator(_ltype(tmp_path, data_dir, "45"), method="ode").steady_state()
    assert ss.converged
    assert ss.newton_distance <= 1.0
    assert ss.continued_run_distance <= 1.0


@pytest.mark.parametrize("k_v_stim", [None, "49.999999999998"])
def test_a_state_on_its_switch_is_refused(tmp_path, data_dir, k_v_stim):
    # As the model is, Voltage relaxes to 50, the switch's own threshold, and
    # 2e-12 below it is within its tolerance of it: which side it rests on is
    # rounding. The step is within tolerance, and a run kicked by a millionth
    # goes on to the other side's state (Calcium at 188 for 958). The residual
    # test returned it.
    ss = bngsim.Simulator(_ltype(tmp_path, data_dir, k_v_stim), method="ode").steady_state()
    assert ss.newton_distance <= 1.0
    assert not ss.converged
    assert not ss.continued_run_distance <= 1.0
    residual = bngsim.Simulator(_ltype(tmp_path, data_dir, k_v_stim), method="ode").steady_state(
        criterion="residual"
    )
    assert residual.converged


# The synthesis of SYNTHESIS_NET written the BNGL way, from a $-fixed source.
SOURCE_NET = """begin parameters
    1 ksyn {ksyn!r}
    2 kdeg 1.0
end parameters
begin species
    1 $Src() 1
    2 X() 0
end species
begin reactions
    1 1 1,2 ksyn
    2 2 0 kdeg
end reactions
"""


@pytest.mark.parametrize("ksyn", [1e-9, 1e-12])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_fixed_species_does_not_set_the_floor(tmp_path, ksyn, method):
    # $Src = 1 held X to 1e-8 against a root of 1e-9: 0.73 of it after a step.
    path = tmp_path / "source.net"
    path.write_text(SOURCE_NET.format(ksyn=ksyn))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.concentrations[1] / ksyn == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_batch_entry_marches_at_its_own_auto_atol(tmp_path, method):
    # Taken from the model at s = 1, the "auto" ran the entry at s = 1e-9
    # through zero (x = -3.7).
    sim = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode")
    for out in (
        sim.steady_state_batch([{"s": 1e-9}], criterion="scaled", method=method)[0],
        sim.steady_state_batch(
            params=[{"s": 1e-9}], states=[[0.5e-9]], criterion="scaled", method=method
        )[0],
    ):
        assert out.converged
        assert out.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


# A -> S1 at 1, A + S1 -> S2 + S1 at 1e9 (S1 speeds A's loss to S2); one law
# A + S1 + S2, and only A kept. A's one steady state is 0.
CATALYZED_NET = """begin parameters
    1 k1 1.0
    2 k3 1e9
end parameters
begin species
{species}
end species
begin reactions
    1 {a} {s1} k1
    2 {a},{s1} {s2},{s1} k3
end reactions
"""


@pytest.mark.parametrize("order", [("A", "S1", "S2"), ("A", "S2", "S1")])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_two_masked_members_of_a_law_do_not_depend_on_species_order(tmp_path, order, method):
    # Holding one masked member and moving the other made the step depend on
    # which: listed A, S1, S2 the state was accepted where it started, A = 1.
    start = {"A": 1.0, "S1": 0.0, "S2": 0.0}
    index = {name: i + 1 for i, name in enumerate(order)}
    path = tmp_path / "catalyzed.net"
    path.write_text(
        CATALYZED_NET.format(
            species="\n".join(f"    {index[n]} {n}() {start[n]!r}" for n in order),
            a=index["A"],
            s1=index["S1"],
            s2=index["S2"],
        )
    )
    mask = [name == "A" for name in order]
    a = order.index("A")
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    early = sim.steady_state(criterion="scaled", method=method, mask=mask, max_time=1e-12)
    assert not early.converged
    ss = sim.steady_state(criterion="scaled", method=method, mask=mask)
    assert ss.converged
    assert abs(ss.concentrations[a]) < 1e-8


# A -> B at 1 and A -> C at 2: one law A + B + C, B and C write-only sinks.
BRANCH_NETS = {
    "A first": """begin parameters
    1 k1 1.0
    2 k2 2.0
end parameters
begin species
    1 A() 1.0
    2 B() 0
    3 C() 0
end species
begin reactions
    1 1 2 k1
    2 1 3 k2
end reactions
""",
    "A last": """begin parameters
    1 k1 1.0
    2 k2 2.0
end parameters
begin species
    1 B() 0
    2 C() 0
    3 A() 1.0
end species
begin reactions
    1 3 1 k1
    2 3 2 k2
end reactions
""",
}


@pytest.mark.parametrize("order", sorted(BRANCH_NETS))
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_two_sinks_of_one_law_converge(tmp_path, order, method):
    # Moving both sinks left no equation for how A's loss splits between them,
    # and the matrix was singular at every state: never converged.
    path = tmp_path / "branch.net"
    path.write_text(BRANCH_NETS[order])
    model = bngsim.Model.from_net(str(path))
    a = model.species_names.index("A()")
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", method=method, mask=~model.is_pure_sink()
    )
    assert ss.converged
    assert abs(ss.concentrations[a]) < 1e-8


# A + B <-> C, C -> D: two laws as found, D (a sink) in both; C drains.
DRAIN_NET = """begin parameters
    1 kf 2.0
    2 kr 0.3
    3 kd 0.7
end parameters
begin species
    1 A() 1.0
    2 B() 0.5
    3 C() 0
    4 D() 0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
    3 3 4 kd
end reactions
"""


@pytest.mark.parametrize("keep", ["AB", "ABC", "A"])
def test_a_masked_species_that_feeds_kept_ones_is_stepped(tmp_path, keep):
    # Holding C, as the solvers hold a masked species, stepped A and B as if
    # C did not drain; dropping both laws D is in lost B + C - A.
    path = tmp_path / "drain.net"
    path.write_text(DRAIN_NET)
    model = bngsim.Model.from_net(str(path))
    mask = [name[0] in keep for name in model.species_names]
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", max_time=2.0, mask=mask
    )
    assert not ss.converged
    want = _numpy_newton_distance(
        bngsim.Model.from_net(str(path)), np.asarray(ss.concentrations), 1e-8, mask
    )
    assert ss.newton_distance == pytest.approx(want, rel=1e-6)


# X' = if(X < c, klo, 1e3): positive everywhere. Y, settled at 1 beside it,
# makes the difference-quotient step 1.5e-8, past X's switch.
NEAR_ZERO_JUMP_NET = """begin parameters
    1 x0 {x0!r}
    2 one 1.0
end parameters
begin species
    1 X() x0
    2 Y() 1.0
end species
begin reactions
    1 0 1 fr
    2 0 2 one
    3 2 0 one
end reactions
begin groups
    1 Xobs 1
end groups
begin functions
    1 fr() if(Xobs<{c!r},{klo!r},1e3)
end functions
"""


@pytest.mark.parametrize(
    "c,klo,x0",
    [
        (1e-9, 1e-6, 0.0),
        (1e-9, 1e-6, 2.5e-10),
        (1e-9, 1e-6, 9e-10),
        (1e-12, 1e-6, 0.0),
        (1e-20, 1e-30, 5e-21),
    ],
)
@pytest.mark.parametrize("jacobian", ["fd", "auto"])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_jump_within_a_step_above_zero_certifies_nothing(tmp_path, c, klo, x0, jacobian, method):
    # The backward probe would go below 0. Keeping the forward quotient let the
    # jump through; the secant to 0 reads the law's own slope, 0, and where X
    # is too small for that secant to be read (9e-15, 3e-41), a forward quotient
    # at half the step grows across the jump.
    # A closed-form Jacobian of the switch is 0, so that there is no step to
    # take and the runs decide, over max_time: of 1e-30 they see nothing, and
    # at klo = 1e-30 one of 1 moves X by 1e-30, within its tolerance.
    path = tmp_path / "near_zero.net"
    path.write_text(NEAR_ZERO_JUMP_NET.format(x0=x0, c=c, klo=klo))
    for max_time in (1.0, 1e6) if klo > 1e-20 or jacobian == "fd" else (1e6,):
        ss = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
        ).steady_state(criterion="scaled", method=method, max_time=max_time)
        assert not ss.converged, max_time


# A in a 1 mL medium taken up into a 1 fL cell: A -> S, S a write-only sink.
# The law weighs S by the volume ratio, A + 1e-12·S.
UPTAKE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="uptake">
    <listOfCompartments>
      <compartment id="medium" size="{medium!r}" constant="true" spatialDimensions="3"/>
      <compartment id="cell" size="1e-15" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="medium" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S" compartment="cell" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="uptake" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci><ci>medium</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@pytest.mark.parametrize("medium", [1e-3, 1.0])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_sink_in_a_small_compartment_takes_up_its_law(medium, method):
    # Its coefficient, 1e-12 of A's (1e-15 at 1 L), read as rounding: the law
    # kept A for its dependent, and A was accepted where it started.
    model = bngsim.Model.from_sbml_string(UPTAKE_SBML.format(medium=medium))
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", method=method, mask=~model.is_pure_sink()
    )
    assert ss.converged
    assert abs(ss.concentrations[0]) < 1e-8


# A -> B with a rate that reads a counter t, run at 1 from 0: t has no root.
COUNTER_NET = """begin parameters
    1 k1  0.1
    2 one 1.0
end parameters
begin species
    1 A() 100
    2 B() 0
    3 t() 0
end species
begin functions
    1 _rateLaw1() k1*(1+0*t)
end functions
begin reactions
    1 1 2 _rateLaw1
    2 0 3 one
end reactions
begin groups
    1 A_tot  1
    2 B_tot  2
    3 t      3
end groups
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_masked_counter_is_held(tmp_path, method):
    # Its row reads no state, so stepped it made the matrix singular at every
    # state, and A never converged.
    path = tmp_path / "counter.net"
    path.write_text(COUNTER_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method, mask=[True, False, False], max_time=2000.0
    )
    assert ss.converged
    assert ss.concentrations[0] < 1e-6


_B358 = (
    Path(__file__).resolve().parents[2]
    / "parity_checks/rr_parity/models/BIOMD0000000358/BIOMD0000000358_url.xml"
)


@pytest.mark.skipif(
    not _B358.exists(), reason="rr_parity corpus model BIOMD0000000358 not present"
)
def test_a_species_far_below_its_step_keeps_its_forward_column():
    # A species at 1.7e-44: its secant to 0 moves the rows by less than their
    # rounding, the column lost its entries, and the root was refused.
    ss = bngsim.Simulator(
        bngsim.Model.from_sbml(str(_B358)), method="ode", jacobian="fd"
    ).steady_state(criterion="scaled")
    assert ss.converged
    assert ss.newton_distance <= 1.0


# ─── The second test: runs taken on from the state (issue #1001) ─────────────


@pytest.mark.parametrize("s", [1.0, 1e-9, 1e-12])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_unstable_root_is_not_converged(tmp_path, s, method):
    # x = 0.3 is a root in floating point, and the step from it is within
    # tolerance. A run from it stays: kicked by less than the integrator's own
    # tolerance, the implicit method damps the mode it should follow. Kicked by
    # a millionth it goes on to a stable root.
    ss = bngsim.Simulator(_cubic(tmp_path, s, 0.3), method="ode").steady_state(method=method)
    assert ss.newton_distance <= 1.0
    assert not ss.converged
    assert ss.continued_run_distance > 1.0
    assert ss.concentrations[0] / s == pytest.approx(0.3, rel=1e-9)


def test_a_burst_free_newton_solve_keeps_the_first_test(tmp_path):
    # steady_states() enumerates roots, unstable ones with them, and says which
    # are stable: no run is taken on.
    sim = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.3), method="ode")
    [ss] = sim.steady_state_batch(states=[[0.3]], method="newton", newton_burst=False)
    assert ss.converged
    assert ss.root_stability == "unstable"
    assert math.isnan(ss.continued_run_distance)


# X' = 0.001 + 1e8·|X - 1|: positive everywhere, with a kink at 1.
KINK_NET = """begin parameters
    1 x0 0.5
end parameters
begin species
    1 X() x0
end species
begin reactions
    1 0 1 fr
end reactions
begin groups
    1 Xobs 1
end groups
begin functions
    1 fr() if(Xobs<1,0.001+1e8*(1-Xobs),0.001+1e8*(Xobs-1))
end functions
"""


@pytest.mark.parametrize("jacobian", ["auto", "fd"])
def test_a_rate_with_a_floor_at_its_kink_is_not_converged(tmp_path, jacobian):
    # The march parks at the kink, and the step from there, 5e-12, is within
    # tolerance: there is no root. Kicked past the kink the run grows.
    path = tmp_path / "kink.net"
    path.write_text(KINK_NET)
    ss = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
    ).steady_state(max_time=100.0)
    assert ss.newton_distance <= 1.0
    assert not ss.converged


# 0 -> X at k, and nothing reads X: the reduced Jacobian is a zero.
DRIFT_NET = """begin parameters
    1 k {k!r}
end parameters
begin species
    1 X() 1.0
end species
begin reactions
    1 0 1 k
end reactions
"""


@pytest.mark.parametrize("k", [1.0, 1e-12])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_drift_with_no_step_to_take_is_not_converged(tmp_path, k, method):
    # With no step to take the runs decide alone, over max_time: X moves by
    # k·1e6, past its tolerance of 2e-8.
    path = tmp_path / "drift.net"
    path.write_text(DRIFT_NET.format(k=k))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert math.isinf(ss.newton_distance)
    assert not ss.converged
    assert ss.continued_run_distance > 1.0


def test_a_drift_too_slow_to_show_in_max_time_passes(tmp_path):
    # Control: the runs ask that the state stay over max_time, and X' = 1e-20
    # moves X by 1e-14 in 1e6.
    path = tmp_path / "drift.net"
    path.write_text(DRIFT_NET.format(k=1e-20))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state()
    assert math.isinf(ss.newton_distance)
    assert ss.converged


SIR_NET = """begin parameters
    1 beta 0.5
    2 gamma 0.2
end parameters
begin species
    1 S() 0.99
    2 I() 0.01
    3 R() 0
end species
begin reactions
    1 1,2 2,2 beta
    2 2 3 gamma
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_epidemic_that_burns_out_is_converged(tmp_path, method):
    # Every state with I = 0 is a steady state, and the run ends at the one
    # with S = 0.1059, below gamma/beta: a kick of I dies away.
    path = tmp_path / "sir.net"
    path.write_text(SIR_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.converged
    assert ss.continued_run_distance <= 1.0
    s, i, _ = np.asarray(ss.concentrations)
    assert s == pytest.approx(0.105894, rel=1e-4)
    assert abs(i) < 1e-9


def test_above_512_unknowns_the_runs_decide(tmp_path):
    # The cubic at its root 0.3 among 520 unknowns: the residual test, which
    # the solve stops by there, returns it where it starts, and the run kicked
    # off it does not stay.
    wide = _wide_net(260)
    n = 2 * 260 + 1
    text = (
        wide.replace("end parameters", "    3 k3 1.0\n    4 k2 1.3\n    5 k1 0.3\nend parameters")
        .replace("end species", f"    {n} X() 0.3\nend species")
        .replace(
            "end reactions",
            f"    {4 * 260 + 1} {n},{n},{n} {n},{n} k3\n"
            f"    {4 * 260 + 2} {n},{n} {n},{n},{n} k2\n"
            f"    {4 * 260 + 3} {n} 0 k1\nend reactions",
        )
    )
    path = tmp_path / "wide_cubic.net"
    path.write_text(text)
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="newton", atol=1e-10)
    assert out["residual"].converged
    assert math.isnan(out["scaled"].newton_distance)
    assert not out["scaled"].converged
    assert out["scaled"].continued_run_distance > 1.0


def test_an_unstable_root_beside_a_larger_species_is_not_converged(tmp_path):
    # The cubic at 1e-9, at its root 0.3 exactly, beside Y at 1, where Y
    # stays. Each species is measured in its own tolerance: a kick measured in
    # the largest's let X move by 2e-6, and grow to its other root unseen.
    text = (
        CUBIC_NET.format(s=1e-9, x0=0.3)
        .replace("end species", "    2 Y() 1.0\nend species")
        .replace("end reactions", "    4 0 2 k\n    5 2 0 k\nend reactions")
    )
    path = tmp_path / "cubic_beside.net"
    path.write_text(text)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state()
    assert ss.newton_distance <= 1.0
    assert not ss.converged


_B002 = (
    Path(__file__).resolve().parents[2]
    / "parity_checks/rr_parity/models/BIOMD0000000002/BIOMD0000000002_url.xml"
)
_B407 = (
    Path(__file__).resolve().parents[2]
    / "parity_checks/rr_parity/models/BIOMD0000000407/BIOMD0000000407_url.xml"
)


def _run_from(path, y, t_end):
    model = bngsim.Model.from_sbml(str(path))
    model._core.set_state(np.asarray(y, float))
    bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=2, atol="auto")
    return np.asarray(model._core.get_state(), float)


@pytest.mark.skipif(
    not _B002.exists(), reason="rr_parity corpus model BIOMD0000000002 not present"
)
def test_biomd2_comes_out_as_a_run_does():
    # The residual test accepted it 6% from its steady state in a species; a
    # run of 1e6 from where it is returned now moves it by 1e-10 of itself.
    ss = bngsim.Simulator(bngsim.Model.from_sbml(str(_B002)), method="ode").steady_state()
    assert ss.converged
    y = np.asarray(ss.concentrations)
    end = _run_from(_B002, y, 1e6)
    assert np.max(np.abs(end - y) / (np.abs(y) + 1e-6 * np.abs(y).max())) < 1e-8
    residual = bngsim.Simulator(bngsim.Model.from_sbml(str(_B002)), method="ode").steady_state(
        criterion="residual"
    )
    y = np.asarray(residual.concentrations)
    end = _run_from(_B002, y, 1e6)
    assert np.max(np.abs(end - y) / (np.abs(y) + 1e-6 * np.abs(y).max())) > 0.05


@pytest.mark.skipif(
    not _B407.exists(), reason="rr_parity corpus model BIOMD0000000407 not present"
)
def test_biomd407_comes_out_as_a_run_does():
    # The residual test accepted it one step from where it starts. A run is
    # still moving at 1e6 and settles by 1e8: in max_time = 1e6 it is not
    # converged, and in 1e9 it is, where the run ends.
    ss = bngsim.Simulator(bngsim.Model.from_sbml(str(_B407)), method="ode").steady_state()
    assert not ss.converged
    y0 = np.asarray(bngsim.Model.from_sbml(str(_B407))._core.get_state(), float)
    ref = _run_from(_B407, y0, 1e10)
    for method in ("integration", "newton"):
        ss = bngsim.Simulator(bngsim.Model.from_sbml(str(_B407)), method="ode").steady_state(
            method=method, max_time=1e9
        )
        assert ss.converged
        y = np.asarray(ss.concentrations)
        assert np.max(np.abs(y - ref) / (np.abs(ref) + 1e-6 * np.abs(ref).max())) < 1e-8


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_step_along_a_continuum_says_nothing_and_the_runs_decide(data_dir, method):
    # IgE.bngl conserves R(a~1) - R(a~0) + 2·IgE_dimer because its two
    # reactions out of R(a~X) share one rate constant, a law of its rates that
    # no law of its stoichiometry holds: its steady states make a line, the
    # reduced Jacobian is singular but for rounding, and the step along the
    # line was 4e7 tolerances (rounding says how far) at a state no run moves.
    path = str(data_dir / "ige_receptor_cycle.net")
    ss = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(method=method)
    assert ss.converged
    assert ss.continued_run_distance <= 1.0
    y = np.asarray(ss.concentrations)
    run = bngsim.Model.from_net(path)
    run.set_state(y)
    bngsim.Simulator(run, method="ode").run(t_span=(0.0, 1e6), n_points=2, atol="auto")
    np.testing.assert_allclose(np.asarray(run.get_state()), y, rtol=1e-6)


_SHP2 = Path(__file__).resolve().parents[2] / "benchmarks/models/net/ode/SHP2_base_model.net"


@pytest.mark.skipif(not _SHP2.exists(), reason="benchmark net SHP2_base_model not present")
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_solve_seeded_with_its_own_answer_converges_again(method):
    # The floor is the model's, from what it declares. Taken from the state a
    # solve starts at, the species SHP2 drains to nothing set it, and a solve
    # seeded with its own answer was refused there (a step of 7.7e3).
    model = bngsim.Model.from_net(str(_SHP2))
    sim = bngsim.Simulator(model, method="ode")
    first = sim.steady_state(method="integration")
    assert first.converged
    model.set_state(np.asarray(first.concentrations))
    again = sim.steady_state(method=method)
    assert again.converged
    assert again.newton_distance <= 1.0
