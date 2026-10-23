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
    Its size is taken over each kept species' rtol·|y| + min(atol_i,
    rtol·largest): atol_i the march's, the smaller of what atol="auto" derives
    from the start y0 and 1e-8, and `largest` the largest species of y0, y and
    y + δ."""
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
    a0 = np.abs(y0)
    floor = a0[a0 > 1e-14 * a0.max()].min()
    atol = np.minimum(rtol * np.maximum(a0, floor), 1e-8)
    largest = max(a0[~fixed].max(), np.abs(y[~fixed]).max(), np.abs((y + delta)[~fixed]).max())
    scale = rtol * np.abs(y) + np.minimum(atol, rtol * largest)
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
    # against 2 s). That answer is polished, to a thousandth of tol, and a run
    # taken on from it decides.
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
    assert scaled.residual < 1e-12
    np.testing.assert_allclose(scaled.concentrations, residual.concentrations, rtol=1e-8)


@pytest.mark.parametrize("name", ["BIOMD0000000252", "BIOMD0000000685"])
def test_a_march_that_stalls_short_of_its_root_is_polished(data_dir, name):
    # BIOMD0000000252's march stalls at step 5411 at a residual of 2.1e-7, a
    # step of 1.64 tolerances from its root, and BIOMD0000000685's 1.62 from
    # its, where main had stopped under tol at states no run moves. A short
    # Newton polish from there is kept where it passes the parity test and
    # lands where the step said.
    path = str(data_dir / f"{name}.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state()
    assert ss.converged
    assert ss.newton_distance <= 1.0
    assert ss.continued_run_distance <= 1.0
    run = bngsim.Model.from_sbml(path)
    bngsim.Simulator(run, method="ode").run(
        t_span=(0.0, 1e7), n_points=2, rtol=1e-11, atol=1e-16, max_steps=2_000_000
    )
    end = np.asarray(run._core.get_state(), float)
    y = np.asarray(ss.concentrations)
    np.testing.assert_allclose(y, end, rtol=1e-6, atol=1e-9 * np.abs(end).max())


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
    ],
)
@pytest.mark.parametrize("jacobian", ["fd", "auto"])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_jump_within_a_step_above_zero_certifies_nothing(tmp_path, c, klo, x0, jacobian, method):
    # The backward probe would go below 0. Keeping the forward quotient let the
    # jump through; the secant to 0 reads the law's own slope, 0, and where X
    # is too small for that secant to be read (9e-15, 3e-41), a forward quotient
    # at half the step grows across the jump.
    # The step steers where the solve stops and decides nothing; the parity
    # test and the runs decide, over max_time. (At c = 1e-20, klo = 1e-30 the
    # drift is too slow to show in it: X moves by 1e-24 in 1e6, within its
    # tolerance of 2.2e-24, and the state passes.) On differenced columns a
    # march of 1 from 9e-10 ends at X = -86, where X' is 1e-6 and its
    # tolerance 1.7e-6.
    path = tmp_path / "near_zero.net"
    path.write_text(NEAR_ZERO_JUMP_NET.format(x0=x0, c=c, klo=klo))
    short = not (jacobian == "fd" and x0 == 9e-10)
    for max_time in (1.0, 1e6) if short else (1e6,):
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
    # There is no step to take. At k = 1 the residual test refuses; at 1e-12,
    # under tol, a run of max_time moves X by 1e-6, past its tolerance.
    path = tmp_path / "drift.net"
    path.write_text(DRIFT_NET.format(k=k))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert math.isinf(ss.newton_distance)
    assert not ss.converged
    if k < 1e-9:
        assert ss.continued_run_distance > 1.0
    else:
        assert math.isnan(ss.continued_run_distance)


def test_a_drift_too_slow_to_show_in_max_time_passes(tmp_path):
    # Control: the runs ask that the state stay over max_time, and X' = 1e-20
    # moves X by 1e-14 in 1e6.
    path = tmp_path / "drift.net"
    path.write_text(DRIFT_NET.format(k=1e-20))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state()
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


def test_above_512_unknowns_a_march_stopped_short_is_polished_to_its_root(tmp_path):
    # X' = 1e-6·(1 - X²) among 520 unknowns: the march stops by the residual
    # test at X = 0.698, which the residual criterion returns. With no step to
    # tell how far off it is, the state is polished, to a thousandth of tol and
    # on a Jacobian taken at every iterate: on the one it started with, the
    # polish's ten iterations ended 3.7e-5 short of 1.
    n = 2 * 260 + 1
    text = (
        _wide_net(260)
        .replace("end parameters", "    3 kx 1e-6\nend parameters")
        .replace("end species", f"    {n} X() 0\nend species")
        .replace(
            "end reactions",
            f"    {4 * 260 + 1} 0 {n} kx\n    {4 * 260 + 2} {n},{n} {n} kx\nend reactions",
        )
    )
    path = tmp_path / "wide_slow.net"
    path.write_text(text)
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="integration")
    assert out["residual"].converged
    assert abs(out["residual"].concentrations[n - 1] - 1.0) > 0.1
    scaled = out["scaled"]
    assert scaled.converged
    assert math.isnan(scaled.newton_distance)
    assert abs(scaled.concentrations[n - 1] - 1.0) < 1e-10


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


_B002 = Path(__file__).resolve().parents[2] / "tests/data/BIOMD0000000002.xml"
_B407 = Path(__file__).resolve().parents[2] / "tests/data/BIOMD0000000407.xml"


def _run_from(path, y, t_end):
    model = bngsim.Model.from_sbml(str(path))
    model._core.set_state(np.asarray(y, float))
    bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=2, atol="auto")
    return np.asarray(model._core.get_state(), float)


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
    # The species SHP2 drains to nothing sit at 1e-20 and below: taken as the
    # floor of the march's tolerances, they held every species at 0 to that,
    # and a solve seeded with its own answer was refused there.
    model = bngsim.Model.from_net(str(_SHP2))
    sim = bngsim.Simulator(model, method="ode")
    first = sim.steady_state(method="integration")
    assert first.converged
    model.set_state(np.asarray(first.concentrations))
    again = sim.steady_state(method=method)
    assert again.converged
    assert again.continued_run_distance <= 1.0
    np.testing.assert_allclose(
        np.asarray(again.concentrations),
        np.asarray(first.concentrations),
        rtol=1e-6,
        atol=1e-8,
    )


# ─── What the independent review found (issue #1001) ─────────────────────────

# 0 -> X at s, X -> 0 at k1, X + X -> D at k2, D -> 0; G, unrelated, at `big`.
BESIDE_A_LARGE_ONE_NET = """begin parameters
    1 s   1.3
    2 k1  0.21
    3 k2  0.73
    4 kd  0.5
    5 sg  {big!r}
    6 dg  1.0
end parameters
begin species
    1 X() 0
    2 D() 0
    3 G() {big!r}
end species
begin reactions
    1 0 1 s
    2 1 0 k1
    3 1,1 2 k2
    4 2 0 kd
    5 0 3 sg
    6 3 0 dg
end reactions
"""


@pytest.mark.parametrize("big", [1e8, 1e12])
@pytest.mark.parametrize("jacobian", ["auto", "fd"])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_beside_a_large_one_is_solved_to_its_root(tmp_path, big, jacobian, method):
    # Control: main solves it. X starts at 0 beside G at `big`; held to
    # rtol·big, as atol="auto" holds a species with no size of its own, X
    # came back converged 5% off at 1e8 and 129% off at 1e12 (differenced
    # columns) on the way to this change.
    path = tmp_path / "beside.net"
    path.write_text(BESIDE_A_LARGE_ONE_NET.format(big=big))
    ss = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
    ).steady_state(method=method)
    x_root = (-0.21 + math.sqrt(0.21**2 + 8 * 0.73 * 1.3)) / (4 * 0.73)
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(x_root, rel=1e-7)


# A + A -> 0 at 1, 0 -> A at 1e-6: A settles at 7.07e-4 from wherever it starts.
SETTLES_SMALL_NET = """begin parameters
    1 k 1.0
    2 s 1e-6
end parameters
begin species
    1 A() {a0!r}
end species
begin reactions
    1 1,1 0 k
    2 0 1 s
end reactions
"""


@pytest.mark.parametrize("a0", [1e5, 1e7, 1e8])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_that_settles_far_below_its_start_is_solved_to_its_root(tmp_path, a0, method):
    # Marched at atol="auto" from 1e7, A was held to 0.1 and came back
    # converged 91 times its root; from 1e5 the march ran to -1e19.
    path = tmp_path / "settles.net"
    path.write_text(SETTLES_SMALL_NET.format(a0=a0))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(math.sqrt(1e-6 / 2), rel=1e-7)


def test_a_tol_the_call_gives_is_read(tmp_path):
    # Control: main reads it. A <-> B with A0 the rate back: A* = 9/4. Under
    # the default the parity test is asked too, at the tol given.
    path = tmp_path / "ab.net"
    path.write_text(
        "begin parameters\n    1 A0 3\n    2 kf 1\nend parameters\n"
        "begin species\n    1 A() A0\n    2 B() 0\nend species\n"
        "begin reactions\n    1 1 2 kf\n    2 2 1 A0\nend reactions\n"
    )
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(tol=1e-14)
    assert ss.converged and ss.residual < 1e-14
    np.testing.assert_allclose(np.asarray(ss.concentrations), [9 / 4, 3 / 4], rtol=1e-12)


@pytest.mark.parametrize(
    "text,want",
    [
        ("model neg\n V = 0\n J0: -> V; k*(Vrest - V)\n k = 0.5\n Vrest = -65\nend", [-65.0]),
        (
            "model neg2\n V = -60\n n = 0.3\n J0: -> V; k*(Vrest - V) - g*n*(V - Ek)\n"
            " J1: -> n; a*(1 - n) - b*n\n k = 0.5\n Vrest = -65\n g = 0.1\n Ek = -80\n"
            " a = 0.1\n b = 0.2\nend",
            [-65.9375, 1 / 3],
        ),
    ],
)
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_variable_that_rests_below_zero_is_converged(text, want, method):
    # Control: main converges. A membrane potential at -65; a guard on the
    # sign of every species refused it on the way to this change.
    model = bngsim.Model.from_antimony_string(text)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged
    np.testing.assert_allclose(np.asarray(ss.concentrations), want, rtol=1e-7)


# The cubic at its root 0.3 beside two laws whose dependents, C1 and C2, sit at
# 0 and would want kicks of opposite signs.
KICK_BESIDE_LAWS_NET = """begin parameters
    1 k3 1.0
    2 k2 1.3
    3 k1 0.3
    4 k 1.0
end parameters
begin species
    1 X() 0.3
    2 C1() 0
    3 E1() 1
    4 C2() 0
    5 E2() 1
end species
begin reactions
    1 1,1,1 1,1 k3
    2 1,1 1,1,1 k2
    3 1 0 k1
    4 2 3 k
    5 4 5 k
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_unstable_root_beside_a_dependent_at_zero_is_not_converged(tmp_path, method):
    # Scaled down to the room its law's dependents had, the whole kick fell to
    # 0 and the root at 0.3 was returned converged.
    path = tmp_path / "kick.net"
    path.write_text(KICK_BESIDE_LAWS_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.concentrations[0] == pytest.approx(0.3, rel=1e-9)
    assert not ss.converged
    assert ss.continued_run_distance > 1.0


@pytest.mark.parametrize("x0", [0.5, 0.9, 1.5])
def test_a_state_set_at_another_scale_is_held_to_its_own(tmp_path, x0):
    # The cubic at a scale of 1e-9, declared at 1 and started by set_state()
    # and states=. Held to the floor its declaration gave, it was returned
    # where it was set.
    model = _cubic(tmp_path, 1e-9, 1.0 / 1e-9)
    model.set_state([x0 * 1e-9])
    ss = bngsim.Simulator(model, method="ode").steady_state()
    assert ss.converged
    assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)
    [batch] = bngsim.Simulator(
        _cubic(tmp_path, 1e-9, 1.0 / 1e-9), method="ode"
    ).steady_state_batch(states=[[x0 * 1e-9]])
    assert batch.converged
    assert batch.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


def test_a_species_at_roundoff_does_not_set_the_march_tolerance():
    # MODEL2307110001 declares a species at -1.6e-34. Taken as the floor of
    # atol="auto", it held every species at 0 to that, and the march failed.
    sim = (
        bngsim.Simulator(bngsim.Model.from_net(str(_SHP2)), method="ode")
        if _SHP2.exists()
        else None
    )
    if sim is None:
        pytest.skip("benchmark net SHP2_base_model not present")
    state = np.asarray(sim._model.get_state(), float)
    state[0] = -1.6e-34
    atol = np.asarray(sim._steady_state_atol(None, "scaled", state, 1e-8))
    assert atol.min() > 1e-30
    assert atol.max() <= 1e-8


def test_a_run_is_not_moved_by_its_own_error(data_dir):
    # Integrated at the criterion's own tolerances, the run moved the state by
    # 3.45 tolerances, at a hundredth of them by 1.41, at a thousandth by 0.46:
    # its own error. A run that moves the state is taken again tighter.
    path = str(data_dir / "BIOMD0000000038.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state()
    assert ss.converged
    assert ss.continued_run_distance <= 1.0


# ─── What the second review found (issue #1001) ──────────────────────────────

# 0 -> S at k0, S -> 0 at kd, from S = 1: S* = k0/kd, dS*/dkd = -k0/kd^2.
FROM_ABOVE_NET = """begin parameters
    1 k0  5e-14
    2 kd  5e-3
end parameters
begin species
    1 S() 1.0
end species
begin reactions
    1 0 1 k0
    2 1 0 kd
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_settling_far_below_its_start_has_its_root_and_column(tmp_path, method):
    # Control: main's steps land on the root. S settles at 1e-11 from 1, at an
    # atol of 1e-8: stopped within tolerance, S came back 0.4% off, and
    # dS*/dkd with it, on the way to this change. The state is polished.
    path = tmp_path / "from_above.net"
    path.write_text(FROM_ABOVE_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method, sensitivity_params=["kd"]
    )
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(5e-14 / 5e-3, rel=1e-9)
    assert np.asarray(ss.sensitivity)[0, 0] == pytest.approx(-5e-14 / 5e-3**2, rel=1e-6)


UNSTABLE_IN_A_LAW_ANT = """model ul
  D = 0
  X = 0.3
  E = 10
  J1: E -> X; k2*X^2
  J2: X -> E; k3*X^3 + k1*X
  J3: E + D -> D + D; kde*E*D
  J4: D -> E; kd*D
  k1 = 0.3; k2 = 1.3; k3 = 1.0; kde = 0.01; kd = 1.0
end"""

UNSTABLE_BESIDE_A_LOPSIDED_LAW_ANT = """model ui
  B = 1e-6
  A = 1e6
  X = 0.3
  J1: A -> B; kf*A
  J2: B -> A; kr*B
  J3: -> X; k2*X^2
  J4: X -> ; k3*X^3 + k1*X
  kf = 0; kr = 0; k1 = 0.3; k2 = 1.3; k3 = 1.0
end"""


@pytest.mark.parametrize("text", [UNSTABLE_IN_A_LAW_ANT, UNSTABLE_BESIDE_A_LOPSIDED_LAW_ANT])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_unstable_root_in_or_beside_a_law_is_not_converged(text, method):
    # The cubic's root at 0.3 in a law whose dependent sits at 0: its kick was
    # dropped. Beside a law of A at 1e6 and B at 1e-6 (which do not exchange,
    # so that the state rests and the kicked run decides): B, the dependent,
    # was kicked by 1 by A's millionth, and every species let move 5e14
    # tolerances. Each species is measured by the kicks of the unknowns it is
    # coupled to, which here are X's own.
    model = bngsim.Model.from_antimony_string(text)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert not ss.converged
    assert ss.continued_run_distance > 1.0


@pytest.mark.parametrize("x0", [0.5, 0.9, 1.5])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_cubic_beside_a_species_at_1e6_reaches_its_root(tmp_path, x0, method):
    # 1e-14 of the largest species was taken for dust, and X at 5e-10 beside
    # Y at 1e6 was held to 1e-8: the issue's answers came back (0.894, 1.008,
    # 0.443, and 0.3 by Newton). A species keeps its own size.
    text = (
        CUBIC_NET.format(s=1e-9, x0=x0)
        .replace("end species", "    2 Y() 1e6\nend species")
        .replace("end reactions", "    4 0 2 big\n    5 2 0 k\nend reactions")
        .replace("end parameters", "    9 big 1e6\nend parameters")
    )
    path = tmp_path / "cubic_beside_big.net"
    path.write_text(text)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.converged
    assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("max_time", [1e8, 1e10])
def test_a_long_max_time_does_not_lengthen_the_runs_past_the_default(data_dir, max_time):
    # Taken on for max_time, the runs from BIOMD0000000038's steady state at
    # 1e8 could not be taken to their time, and egfr_net's at 1e10 took 228 s;
    # main converged on both.
    path = str(data_dir / "BIOMD0000000038.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state(
        max_time=max_time
    )
    assert ss.converged
    assert ss.continued_run_distance <= 1.0


def test_a_burst_free_solve_says_it_was_decided_by_the_residual_test(tmp_path):
    # Control: main's default is the residual test.
    sim = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode")
    [ss] = sim.steady_state_batch(states=[[0.5]], method="newton", newton_burst=False)
    assert ss.converged
    assert ss.criterion == "residual"


def test_a_march_that_stops_where_f_is_rounding_returns_the_state_it_passed(data_dir):
    # Control: main stops at the step that passes. Monro2008's R rests at 2e12,
    # where f is rounding: 0 at step 517, which falls between the steps the
    # step test is asked at, and 2.6e-6 at step 518, the march's max_time. The
    # march returns the last state that passed.
    path = str(data_dir / "BIOMD0000000776.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state()
    assert ss.converged
    assert ss.residual < 1e-9


SIR_AT_NO_INFECTION_ANT = """model sir0
  S = 0.99; I = 0; R = 0.01
  J1: S + I -> 2 I; kb*S*I
  J2: I -> R; kg*I
  kb = 0.5; kg = 0.2
end"""

COUNTER_AT_NO_PULSE_ANT = """model counter
  S0 = 100; S1 = 0; S2 = 0; S3 = 0; P = 0
  J1: S0 -> S1; k*S0*P
  J2: S1 -> S2; k*S1*P
  J3: S2 -> S3; k*S2*P
  J4: P -> ; kd*P
  k = 10; kd = 1
end"""


@pytest.mark.parametrize("text", [SIR_AT_NO_INFECTION_ANT, COUNTER_AT_NO_PULSE_ANT])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_absent_species_is_not_kicked(text, method):
    # Control: main converges. A species at 0 at a state that rests stays at 0:
    # nothing makes it, and no integrator error makes it from nothing. Kicked,
    # the infection grew (by 4e6 of the kick), and the counter's pulse moved
    # the counter by 50 times what the kick may grow to.
    model = bngsim.Model.from_antimony_string(text)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged
    np.testing.assert_array_equal(ss.concentrations, model.get_state())


SQUARE_DECAY_NET = """begin parameters
    1 k 1.0
end parameters
begin species
    1 X() 1.0
end species
begin reactions
    1 1,1 1 k
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_reaching_zero_as_one_over_t_is_polished_there(tmp_path, method):
    # X' = -X² reaches its root, 0, as 1/t. The residual test returns X at
    # 3e-5, which a run of max_time halves; the march ends at max_time with X
    # near 1e-6, 50 tolerances short. The polish halves X at each of its ten
    # iterations, its steps never small enough to stop it, and so ends at its
    # iteration limit, where the iterate it ends at passes the parity test and
    # is kept.
    path = tmp_path / "square_decay.net"
    path.write_text(SQUARE_DECAY_NET)
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method=method)
    assert out["residual"].converged
    assert out["residual"].concentrations[0] > 1e-5
    scaled = out["scaled"]
    assert scaled.converged
    assert 0.0 < scaled.concentrations[0] < 1e-8
    assert scaled.continued_run_distance <= 1.0


LOPSIDED_LAW_ANT = """model lop
  B = 1e-6
  A = 1e6
  J1: A -> B; kf*A
  J2: B -> A; kr*B
  kf = 1e-12; kr = 1.0
end"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_polish_that_loses_a_dependents_digits_is_not_kept(method):
    # Control for integration: main's march leaves B exact. The start is the
    # steady state. KINSOL solves for A and makes B from the total, 1e6 + 1e-6,
    # which leaves B at 1.0000076e-6, 381 tolerances off, where main's Newton
    # solve returns it. The polish is kept only where it is nearer the root.
    model = bngsim.Model.from_antimony_string(LOPSIDED_LAW_ANT)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged
    np.testing.assert_array_equal(ss.concentrations, model.get_state())


@pytest.mark.parametrize("x0", [0.5, 1.5])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_fixed_species_far_above_the_rest_does_not_coarsen_the_march(tmp_path, x0, method):
    # The cubic at 1e-9 beside a $-fixed Y at 1e20. Counted in the march's atol
    # floor, Y's double-precision resolution, 2e4, held X to the scalar 1e-8,
    # and the march ran X to 9.6 and 29 times its root and refused.
    text = CUBIC_NET.format(s=1e-9, x0=x0).replace("end species", "    2 $Y() 1e20\nend species")
    path = tmp_path / "cubic_beside_fixed.net"
    path.write_text(text)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.converged
    assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


def _near_unstable_ant(a):
    # X' = -1e4·X·(X - a)·(X - 1) at its unstable root a, beside A <-> B at
    # 2^20 and 2^-20, balanced in floating point, which X does not touch.
    return f"""model c
  B = 9.5367431640625e-07
  A = 1048576.0
  X = {a!r}
  J1: A -> B; kf*A
  J2: B -> A; kr*B
  J3: -> X; k2*X^2
  J4: X -> ; k3*X^3 + k1*X
  kf = 9.094947017729282e-13; kr = 1.0
  k3 = 1e4; k2 = {1e4 * (1 + a)!r}; k1 = {1e4 * a!r}
end"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_unrelated_lopsided_law_does_not_hide_an_unstable_root(method):
    # B, the law's dependent, is kicked by 500 of its tolerances; measured
    # against the largest kick of any species, every species was let move 1e4,
    # and X went the 5e3 to its stable root at 1 unrefused.
    model = bngsim.Model.from_antimony_string(_near_unstable_ant(0.9999))
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert not ss.converged
    assert ss.continued_run_distance > 1.0


def test_a_root_the_eigenvalues_call_unstable_is_not_converged():
    # At a = 0.99999 the kick reaches the stable root, 500 tolerances away,
    # before it has grown tenfold, and the kicked run passes; the polish's
    # eigenvalues say the root is unstable.
    model = bngsim.Model.from_antimony_string(_near_unstable_ant(0.99999))
    ss = bngsim.Simulator(model, method="ode").steady_state(method="newton")
    assert ss.root_stability == "unstable"
    assert ss.continued_run_distance <= 1.0
    assert not ss.converged


def test_a_tight_rtol_is_asked_no_finer_than_a_run_can_tell():
    # Control: main converges. Asked for rtol 1e-12, the runs, at 1e-14 and
    # 1e-15, could not be taken to their time and refused after 10.7 s; at
    # 1e-13 they moved the state 3.9 tolerances of 1e-12, their own error.
    path = Path(__file__).resolve().parents[2] / "benchmarks/models/net/ode/Kocieniewski_2012.net"
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        rtol=1e-12, atol=1e-14
    )
    assert ss.converged


def test_a_march_the_step_cannot_stop_returns_the_state_the_residual_test_passed(data_dir):
    # BIOMD0000000463's step reads inf where the residual test passes (a jump
    # beside the state), so the march was taken on past it until it ran away,
    # to a residual of 2e16 with a species at -1.8e5. It returns the state the
    # residual test passed, main's, where the eigenvalues have a positive real
    # part (0.13) and a kicked run leaves.
    path = str(data_dir / "BIOMD0000000463.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state()
    assert not ss.converged
    assert ss.residual < 1e-9
    assert np.min(ss.concentrations) > -1e-6
    assert ss.continued_run_distance > 1.0


BASIN_BESIDE_A_LOPSIDED_LAW_ANT = """model bs
  B = 1e-6
  A = 1e6
  W = 1e-6
  J1: A -> B; kf*A
  J2: B -> A; kr*B
  J3: A -> B; kc*B^2/(K^2 + B^2)
  J4: -> W; B
  J5: W -> ; W
  kf = 1e-12; kr = 1; kc = 2; K = 0.5
end"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_small_dependent_is_kicked_by_a_millionth_of_itself(method):
    # Control for integration: main converges there (its Newton solve leaves
    # B 7.6e-6 off, in its last digits). B, the law's dependent at 1e-6, follows A's
    # kick of a millionth of 1e6: kicked by 1, it crossed into the basin of B's
    # high state at 1.7, which W follows, and a state that rests was refused.
    # A law's kicks are cut to its dependent's own.
    model = bngsim.Model.from_antimony_string(BASIN_BESIDE_A_LOPSIDED_LAW_ANT)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged
    assert ss.concentrations[1] == pytest.approx(1e6, rel=1e-12)
    assert ss.concentrations[2] == pytest.approx(ss.concentrations[0], rel=1e-6)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_kick_answered_by_a_coupled_species_is_measured_by_their_kicks(data_dir, method):
    # Control: main converges. egfr_simple (RuleHub's tutorial, its network
    # generated by BNG2.pl, no EGF): measured against its own kick or the
    # largest of the kicks chosen, a species the kick moved along its
    # continuum was 47 times its allowance away, and a state that rests was
    # refused. Each species is measured by the largest kick among the species
    # it is coupled to.
    path = str(data_dir / "egfr_simple_seed.net")
    ss = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(method=method)
    assert ss.converged


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_march_whose_integrator_fails_returns_the_state_it_passed(data_dir, method):
    # Control: main converges. glioblastoma-egfrviii-signaling (its network
    # generated by BNG2.pl, no actions run) passes the residual test at step
    # 2075, between the steps the step test is asked at, and its integrator
    # then fails, as it does again on difference quotients: the state it
    # failed at came back, with a residual of NaN.
    path = str(data_dir / "glioblastoma_egfrviii_seed.net")
    ss = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(method=method)
    assert ss.converged
    assert ss.residual < 1e-9


def _sir_in_order(order):
    init = {"S": "0.99", "I": "0", "R": "0.01"}
    decl = "\n".join(f"  {s} = {init[s]}" for s in order)
    return f"""model sir0
{decl}
  J1: S + I -> 2 I; kb*S*I
  J2: I -> R; kg*I
  kb = 0.5; kg = 0.2
end"""


@pytest.mark.parametrize("order", ["SIR", "SRI", "ISR", "IRS", "RSI", "RIS"])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_absent_species_is_not_kicked_as_a_laws_dependent(order, method):
    # Control: main converges in every order. Where I is the dependent of
    # S + I + R, it followed S's and R's kicks and grew, and SIR at I = 0 was
    # refused in two orders of six. A law whose dependent is absent takes up
    # the kicks with its largest member present.
    model = bngsim.Model.from_antimony_string(_sir_in_order(order))
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged


def _near_unstable_coupled_ant(coupled, zero):
    prod = "k2*X^2*(A/1048576.0)" if coupled else "k2*X^2"
    z = "  Z = 0\n  J5: Z -> ; Z\n" if zero else ""
    return f"""model c
  B = 9.5367431640625e-07
  A = 1048576.0
  X = 0.9999
{z}  J1: A -> B; kf*A
  J2: B -> A; kr*B
  J3: -> X; {prod}
  J4: X -> ; k3*X^3 + k1*X
  kf = 9.094947017729282e-13; kr = 1.0
  k3 = 1e4; k2 = 19999.0; k1 = 9999.0
end"""


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_unstable_root_coupled_to_a_lopsided_law_is_not_converged(coupled, zero, method):
    # X's production also reads A (A/2^20 = 1 there), which couples X to the
    # law: measured by the largest kick in its coupling, B's, the dependent's,
    # cut to 500 of its tolerances, X was let move 1e4, and went its 5e3 to its
    # stable root at 1 unrefused. With an unrelated Z at 0 beside it, the
    # eigenvalues were not asked.
    model = bngsim.Model.from_antimony_string(_near_unstable_coupled_ant(coupled, zero))
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert not ss.converged


GLUED_BY_A_FIXED_SPECIES_NET = """begin parameters
    1 kf 9.094947017729282e-13
    2 kr 1.0
    3 k3 1e4
    4 k2 19999.0
    5 k1 9999.0
end parameters
begin species
    1 B() 9.5367431640625e-07
    2 A() 1048576.0
    3 X() 0.9999
    4 $E() 1
end species
begin reactions
    1 2,4 1,4 kf
    2 1 2 kr
    3 3,3,4 3,3,3,4 k2
    4 3,3,3 3,3 k3
    5 3 0 k1
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_fixed_species_does_not_couple_what_it_touches(tmp_path, method):
    # X's production and the law's forward rate both read a $-fixed E at 1.
    # Coupled through E's column, X was measured by B's kick and its unstable
    # root returned converged.
    path = tmp_path / "glued.net"
    path.write_text(GLUED_BY_A_FIXED_SPECIES_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert not ss.converged
