"""``steady_state(criterion="scaled")`` (issue #1001).

``||f(y)||_2 / n < tol`` is a number in the model's units of concentration per
time: a model in small units is under it where it starts, or one step later,
and that state came back ``converged=True``. ``criterion="scaled"`` accepts a
state when the Newton step from it to the root is within each species' own
tolerance, which reads the same in any units. These tests cover:

- the issue's cubic at scales 1e-9 and 1e-12 ends at its stable root, by both
  methods, from either side of it;
- the default stays the residual test, which takes no step;
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
- above 512 unknowns the solve is the residual one, step for step;
- a march that stops between the steps the criterion is asked at is asked
  where it stops (a corpus model);
- a law with a species the mask leaves out is solved for that species, so the
  species the mask keeps are stepped (A -> S under the pure-sink mask);
- a differenced Jacobian across a rate law's jump certifies nothing;
- a $-fixed species does not set the floor;
- a batch entry at another scale marches at its own "auto" atol.
"""

from __future__ import annotations

import math
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
def test_scaled_reaches_the_stable_root_at_any_scale(tmp_path, s, x0, method):
    ss = bngsim.Simulator(_cubic(tmp_path, s, x0), method="ode").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.criterion == "scaled"
    assert ss.concentrations[0] / s == pytest.approx(1.0, rel=1e-6)
    assert ss.newton_distance <= 1.0


def test_default_is_the_residual_test(tmp_path):
    # The parity test, unchanged: it takes no Newton step (a Jacobian more per
    # solve), so it reports none.
    at_one = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode").steady_state()
    assert at_one.criterion == "residual"
    assert at_one.converged
    assert at_one.concentrations[0] == pytest.approx(1.0, rel=1e-6)
    assert math.isnan(at_one.newton_distance)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_default_solve_is_the_residual_solve(tmp_path, method):
    # Where the two tests part (the cubic at 1e-9, which the residual test
    # accepts one step from its start), the default is the residual solve to
    # the bit.
    out = {}
    for criterion in (None, "residual"):
        kw = {} if criterion is None else {"criterion": criterion}
        out[criterion] = bngsim.Simulator(_cubic(tmp_path, 1e-9, 0.5), method="ode").steady_state(
            method=method, **kw
        )
    default, residual = out[None], out["residual"]
    assert default.criterion == "residual"
    assert default.n_steps == residual.n_steps
    assert np.array_equal(default.concentrations, residual.concentrations)
    scaled = bngsim.Simulator(_cubic(tmp_path, 1e-9, 0.5), method="ode").steady_state(
        method=method, criterion="scaled"
    )
    assert not np.array_equal(scaled.concentrations, default.concentrations)


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
    species each law is solved for: the Newton step on the species that move,
    with every law kept (L·δ = 0) and every kept species' equation solved
    (f + J·δ = 0 there), a masked species in no law held; over each kept
    species' rtol·|y| + rtol·max(|y|, floor)."""
    ns = len(y)
    cl = model.conservation_laws
    laws = np.asarray(cl["coefficients"], float).reshape(cl["n_laws"], ns)
    kept = np.ones(ns, bool) if mask is None else np.asarray(mask, bool)
    moves = kept | np.any(laws != 0, axis=0)
    f = np.asarray(model.rhs(y), float)
    jac = np.asarray(model.jacobian(y), float)
    lhs = np.vstack([laws[:, moves], jac[kept][:, moves]])
    rhs = np.concatenate([np.zeros(cl["n_laws"]), -f[kept]])
    step, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    assert np.allclose(lhs @ step, rhs, rtol=0, atol=1e-9 * np.max(np.abs(rhs)))
    delta = np.zeros(ns)
    delta[moves] = step
    y0 = np.asarray(model.get_state(), float)
    floor = min(abs(v) for v, keep in zip(y0, kept, strict=True) if keep and v != 0)
    scale = rtol * np.abs(y) + rtol * np.maximum(np.abs(y), floor)
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
    # The step is not taken there, and nothing else of "scaled" applies: with
    # KINSOL's stopping rule left at the scaled one, the polish ran on past the
    # residual's answer (BaruaBCR_2012 was never accepted, 18 s against 2 s).
    path = tmp_path / "wide.net"
    path.write_text(_wide_net(260))
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="newton", atol=1e-10)
    residual, scaled = out["residual"], out["scaled"]
    assert scaled.converged and scaled.method_used == "newton"
    # The test `converged` was decided by.
    assert scaled.criterion == "residual"
    assert math.isnan(scaled.newton_distance)
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
def test_a_law_with_a_masked_member_is_solved_for_it(tmp_path, order, method):
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


def test_the_retried_march_past_a_switch_still_converges(data_dir):
    # GH #176's model: its closed-form Jacobian stops the march, the retry
    # differences across the switch on the way, and its root is away from it.
    ss = bngsim.Simulator(
        bngsim.Model.from_net(str(data_dir / "ltype_calcium_discontinuous_jacobian.net")),
        method="ode",
    ).steady_state(criterion="scaled")
    assert ss.converged
    assert ss.newton_distance <= 1.0


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
