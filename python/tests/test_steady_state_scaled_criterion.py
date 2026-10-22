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
  where it stops (a corpus model).
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
    """The step the criterion measures, by NumPy: the unknowns' Newton step on
    the reduced system, each law's dependent following from it, over each
    species' rtol·|y| + rtol·max(|y|, floor)."""
    ns = len(y)
    cl = model.conservation_laws
    laws = np.asarray(cl["coefficients"], float).reshape(cl["n_laws"], ns)
    included = np.ones(ns, bool) if mask is None else np.asarray(mask, bool)
    unknowns = [i for i in cl["independent"] if included[i]]
    lift = np.zeros((ns, len(unknowns)))
    for j, u in enumerate(unknowns):
        step = np.zeros(ns)
        step[u] = 1.0
        for k, dep in enumerate(cl["dependent"]):
            step[dep] = -(laws[k] @ step - laws[k, dep] * step[dep]) / laws[k, dep]
        lift[:, j] = step
    f = np.asarray(model.rhs(y), float)
    jac = np.asarray(model.jacobian(y), float)
    delta = lift @ np.linalg.solve((jac @ lift)[unknowns, :], -f[unknowns])
    y0 = np.asarray(model.get_state(), float)
    floor = min(abs(v) for v, keep in zip(y0, included, strict=True) if keep and v != 0)
    scale = rtol * np.abs(y) + rtol * np.maximum(np.abs(y), floor)
    return float(np.max(np.abs(delta[included]) / scale[included]))


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
    assert scaled.criterion == "scaled"
    assert math.isnan(scaled.newton_distance)
    assert scaled.n_steps == residual.n_steps
    assert np.array_equal(scaled.concentrations, residual.concentrations)


_B252 = Path("parity_checks/rr_parity/models/BIOMD0000000252/BIOMD0000000252_url.xml")


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
