"""lanl/bngsim#1002: a difference quotient that steps across a jump in a rate law.

Where a run or a steady-state solve has no closed-form Jacobian (``jacobian="fd"``,
or one law the closed form declines, which takes it away from every law), the
solver differences the right-hand side. From a state within one step of a
threshold of ``if``, a comparison or ``ceil``, the step crosses the threshold
and the quotient reads the jump over the step, Δf/h, where the row has a slope:
-1.3e6 for -0.02 beside ``if(X > 5, 0.04, 0.02)`` at X = 5. On main that did
two things, with no error or warning:

* CVODE's Newton iteration under the matrix took corrections a millionth of
  their size, called them converged, and kept the matrix: the run followed its
  predictor, X = 5 - 0.1 t, 166% off at t = 60 (negative by then).
* A species held at 0 under ``if(I > 0, ...)`` is stepped by CVODE's floor,
  far under 1e-12. Through that entry the LU's rounding put 1e-36 into I, which
  turned the condition true: 490% off, and a steady state 5x off with
  ``converged=True``.

bngsim now forms the dense quotient CVODE formed (bngsim/dq_jacobian.hpp), the
same matrix bit for bit, for runs and the steady-state march, and mends a row
whose step crosses a jump (bngsim/fd_jacobian.hpp); a run's colored sparse
quotient does the same. KINSOL's quotient is left as it was. Every closed form
here is piecewise exponential.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

RTOL = 1e-6
ATOL = 1e-9
# The runs here are within a few 1e-6 of their closed forms once nothing steps
# across a jump; main was off by 2e-2 to 5.
OK = 1e-4

_NET = """begin parameters
    1 kx 1
end parameters
begin species
{species}
end species
begin groups
{groups}
end groups
begin functions
{functions}
end functions
begin reactions
{reactions}
end reactions
"""


def _net(
    tmp_path: Path,
    species: list[str],
    groups: list[str],
    functions: list[str],
    reactions: list[str],
    *,
    declined: bool = False,
) -> bngsim.Model:
    """A .net model; ``declined`` adds a reaction on two species of its own whose
    law (``rint``) the closed form declines, so the whole model is differenced."""
    species, groups, functions, reactions = (
        list(species),
        list(groups),
        list(functions),
        list(reactions),
    )
    if declined:
        n = len(species)
        species += ["W() 1", "Z() 0"]
        groups += [f"Wobs {n + 1}"]
        functions += ["gdecl() 1e-3*rint(Wobs*10)"]
        reactions += [f"{n + 1} {n + 2} gdecl"]
    body = _NET.format(
        species="\n".join(f"    {i + 1} {s}" for i, s in enumerate(species)),
        groups="\n".join(f"    {i + 1} {g}" for i, g in enumerate(groups)),
        functions="\n".join(f"    {i + 1} {f}" for i, f in enumerate(functions)),
        reactions="\n".join(f"    {i + 1} {r}" for i, r in enumerate(reactions)),
    )
    path = tmp_path / f"m{len(list(tmp_path.iterdir()))}.net"
    path.write_text(body)
    model = bngsim.Model.from_net(str(path))
    return model


def _run(model: bngsim.Model, t_end: float, n: int, **kw) -> tuple[np.ndarray, np.ndarray]:
    result = bngsim.Simulator(model, method="ode", **kw).run(
        t_span=(0.0, t_end), n_points=n, rtol=RTOL, atol=ATOL
    )
    if kw.get("jacobian") != "fd":
        assert model.analytical_jacobian_decline is not None, "the closed form was not declined"
    return np.asarray(result.time), np.asarray(result.species)


def _worst(x: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(np.abs(x / exact - 1.0)))


# ── X decays at kx·k(X)·X, k = 0.04 above 5 and 0.02 at or below ────────────


def _threshold(tmp_path: Path, x0: float, cond: str = "Xobs > 5", **kw) -> bngsim.Model:
    return _net(
        tmp_path,
        [f"X() {x0!r}", "Y() 0"],
        ["Xobs 1"],
        [f"f() kx*if({cond}, 0.04, 0.02)"],
        ["1 2 f"],
        **kw,
    )


def _threshold_exact(t: np.ndarray, x0: float) -> np.ndarray:
    if x0 <= 5.0:
        return x0 * np.exp(-0.02 * t)
    t5 = math.log(x0 / 5.0) / 0.04
    return np.where(t <= t5, x0 * np.exp(-0.04 * t), 5.0 * np.exp(-0.02 * (t - t5)))


@pytest.mark.parametrize(
    ("declined", "kw"),
    [
        pytest.param(False, {"jacobian": "fd"}, id="fd"),
        pytest.param(True, {}, id="declined-law"),
        pytest.param(
            False, {"jacobian": "fd", "force_sparse_linear_solver": True}, id="fd-sparse"
        ),
    ],
)
def test_a_run_that_starts_on_a_threshold(tmp_path: Path, declined: bool, kw: dict) -> None:
    """X0 = 5 on `X > 5`, falling: the step up crosses the threshold. Main: 1.66
    with "fd" (X = 5 - 0.1 t), 2.1e-2 with a declined law, 1.66 on the colored
    sparse quotient."""
    t, s = _run(_threshold(tmp_path, 5.0, declined=declined), 60.0, 61, **kw)
    assert _worst(s[:, 0], _threshold_exact(t, 5.0)) < OK


@pytest.mark.parametrize(("declined", "kw"), [(False, {"jacobian": "fd"}), (True, {})])
def test_a_run_that_leaves_the_threshold_it_starts_on(
    tmp_path: Path, declined: bool, kw: dict
) -> None:
    """X0 = 5 on `X >= 5`: at the start the step crosses nothing, but X falls
    below 5 at once and the matrix formed a hair under it does. Main: 1.7 with
    "fd", 7.8e-2 with a declined law."""
    t, s = _run(_threshold(tmp_path, 5.0, cond="Xobs >= 5", declined=declined), 60.0, 61, **kw)
    assert _worst(s[:, 0], 5.0 * np.exp(-0.02 * t)) < OK


_FD_PATHS = pytest.mark.parametrize(
    "kw",
    [{"jacobian": "fd"}, {"jacobian": "fd", "force_sparse_linear_solver": True}],
    ids=["fd", "fd-sparse"],
)


@_FD_PATHS
def test_sensitivities_of_a_run_that_starts_on_a_threshold(tmp_path: Path, kw: dict) -> None:
    """X = 5·exp(-0.02·kx·t), so dX/dkx = -0.02·t·X, at most 1.84 in size. Main:
    the state 1.66 off and dX/dkx off by 4.19."""
    result = bngsim.Simulator(
        _threshold(tmp_path, 5.0), method="ode", sensitivity_params=["kx"], **kw
    ).run(t_span=(0.0, 60.0), n_points=61, rtol=RTOL, atol=ATOL)
    t = np.asarray(result.time)
    exact = 5.0 * np.exp(-0.02 * t)
    assert _worst(np.asarray(result.species)[:, 0], exact) < OK
    dxdk = np.asarray(result.sensitivities)[:, 0, 0]
    assert float(np.max(np.abs(dxdk + 0.02 * t * exact))) < 1e-3


@_FD_PATHS
def test_a_law_steep_enough_to_step_like_a_jump(tmp_path: Path, kw: dict) -> None:
    """kx·(0.03 + 0.01·tanh(1e10·(X − 5)))·X is smooth, but across the quotient's
    step of 7.5e-8 it is a jump from 0.02 to 0.04, and X0 = 5 sits on it: X =
    5·exp(-0.02 t) after a transient of 1e-10. Main: 1.66 off. A test of a rate
    law's operators would not have looked at this one."""
    model = _net(
        tmp_path,
        ["X() 5", "Y() 0"],
        ["Xobs 1"],
        ["f() kx*(0.03 + 0.01*tanh(1e10*(Xobs - 5)))"],
        ["1 2 f"],
    )
    t, s = _run(model, 60.0, 61, **kw)
    assert _worst(s[:, 0], 5.0 * np.exp(-0.02 * t)) < OK


def test_a_crossing_during_the_run(tmp_path: Path) -> None:
    """Control. X0 = 10 crosses 5 at t = 17.3: no matrix is formed within a step
    of the threshold. Right on main too."""
    t, s = _run(_threshold(tmp_path, 10.0), 60.0, 61, jacobian="fd")
    assert _worst(s[:, 0], _threshold_exact(t, 10.0)) < OK


# ── The 2026-10-19 comment: a ceil() law over the state ─────────────────────


def _ceil_exact(t: float, x: float = 10.0, at: float = 0.0) -> float:
    # X falls at 0.01·c between the steps of c = ceil(0.4·X): 0.04 to X = 7.5, then 0.03, ...
    while True:
        c = math.ceil(0.4 * x - 1e-12)
        leaves = (c - 1) / 0.4
        jump = at + math.log(x / leaves) / (0.01 * c)
        if t <= jump:
            return x * math.exp(-0.01 * c * (t - at))
        x, at = leaves, jump


@pytest.mark.parametrize(("declined", "kw"), [(False, {"jacobian": "fd"}), (True, {})])
def test_a_run_that_starts_on_a_step_of_ceil(tmp_path: Path, declined: bool, kw: dict) -> None:
    """X0 = 10 is on a step of ceil(0.4·X). Main: 3.4e-2 with "fd", 4.9e-2 with a
    declined law."""
    model = _net(
        tmp_path,
        ["X() 10", "Y() 0"],
        ["Xobs 1"],
        ["f() kx*0.01*ceil(Xobs*0.4)"],
        ["1 2 f"],
        declined=declined,
    )
    t, s = _run(model, 30.0, 31, **kw)
    assert _worst(s[:, 0], np.array([_ceil_exact(ti) for ti in t])) < OK


# ── A species held at 0 under `> 0` ──────────────────────────────────────────


def _held_inhibitor(tmp_path: Path, **kw) -> bngsim.Model:
    """I = 0 for good (nothing makes it); Y is made at 1 and lost at 0.05 while
    I = 0 (0.01 were I > 0). Steady state Y = 20."""
    return _net(
        tmp_path,
        ["I() 0", "Y() 10", "S() 1"],
        ["Iobs 1"],
        ["f() kx*if(Iobs > 0, 0.01, 0.05)", "p() 1"],
        ["2 0 f", "3 2,3 p"],
        **kw,
    )


@pytest.mark.parametrize(
    ("declined", "kw"),
    [(False, {"jacobian": "fd"}), (True, {})],
)
def test_a_species_held_at_zero_stays_there(tmp_path: Path, declined: bool, kw: dict) -> None:
    """I = 0 for good; Y is lost at 0.05 while I = 0 (0.01 were I > 0), so Y =
    10·exp(-0.05 t). Main put 1e-35 into I through the LU, and Y fell at 0.01:
    4.9 off at t = 60 with "fd", 10 with a declined law. The leak's sign is the
    LU's rounding (on other output grids main leaked below 0 and Y came out
    right), so I is also held to exactly 0."""
    model = _net(
        tmp_path,
        ["I() 0", "Y() 10"],
        ["Iobs 1"],
        ["f() kx*if(Iobs > 0, 0.01, 0.05)"],
        ["2 0 f"],
        declined=declined,
    )
    t, s = _run(model, 60.0, 121, **kw)
    assert _worst(s[:, 1], 10.0 * np.exp(-0.05 * t)) < OK
    assert np.all(s[:, 0] == 0.0)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_steady_state_of_a_species_held_at_zero(tmp_path: Path, method: str) -> None:
    """Main: Y = 100 for 20 with converged=True, by either method."""
    result = bngsim.Simulator(_held_inhibitor(tmp_path), method="ode", jacobian="fd").steady_state(
        method=method
    )
    assert result.converged
    c = np.asarray(result.concentrations)
    assert c[0] == 0.0
    assert abs(c[1] / 20.0 - 1.0) < OK


def _gate(tmp_path: Path, **kw) -> bngsim.Model:
    """X = 0 for good; Y is made at 0.5 while X = 0 (1 were X > 0), from a source
    S, and lost at kx·Y. Y(t) = 0.5 + 0.5·exp(-t)."""
    return _net(
        tmp_path,
        ["X() 0", "Y() 1", "S() 1"],
        ["Xobs 1", "Sobs 3"],
        ["g() if(Xobs > 0, 1, 0.5)/Sobs"],
        ["3 2,3 g", "2 0 kx"],
        **kw,
    )


@pytest.mark.parametrize(("declined", "kw"), [(False, {"jacobian": "fd"}), (True, {})])
def test_a_gate_held_at_zero(tmp_path: Path, declined: bool, kw: dict) -> None:
    """Main: Y went to 1 for 0.5, 100% off, with "fd" and with a declined law."""
    t, s = _run(_gate(tmp_path, declined=declined), 60.0, 61, **kw)
    assert np.all(s[:, 0] == 0.0)
    assert _worst(s[:, 1], 0.5 + 0.5 * np.exp(-t)) < OK


def test_a_gate_held_at_zero_under_equality(tmp_path: Path) -> None:
    """`X == 0` at X = 0: the steps both ways cross, so neither one-sided quotient
    is the row's slope and the entry is 0. Main: Y went to 1 for 0.5."""
    model = _net(
        tmp_path,
        ["X() 0", "Y() 1", "S() 1"],
        ["Xobs 1", "Sobs 3"],
        ["g() if(Xobs == 0, 0.5, 1)/Sobs"],
        ["3 2,3 g", "2 0 kx"],
    )
    t, s = _run(model, 60.0, 61, jacobian="fd")
    assert np.all(s[:, 0] == 0.0)
    assert _worst(s[:, 1], 0.5 + 0.5 * np.exp(-t)) < OK


def test_a_stiff_run_that_starts_on_a_threshold(tmp_path: Path) -> None:
    """X0 = 5 on `X > 5` with rates 2000 above and 1000 at or below: X =
    5·exp(-1000 t). Main: 1.73 off at t = 1e-3 (35% of X0)."""
    model = _net(
        tmp_path, ["X() 5", "Y() 0"], ["Xobs 1"], ["f() kx*if(Xobs > 5, 2000, 1000)"], ["1 2 f"]
    )
    result = bngsim.Simulator(model, method="ode", jacobian="fd").run(
        t_span=(0.0, 0.01), n_points=11, rtol=RTOL, atol=1e-12
    )
    t, x = np.asarray(result.time), np.asarray(result.species)[:, 0]
    assert float(np.max(np.abs(x - 5.0 * np.exp(-1000.0 * t)))) < 5.0 * OK


@pytest.mark.parametrize(("declined", "kw"), [(False, {"jacobian": "fd"}), (True, {})])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_steady_state_of_a_gate_held_at_zero(
    tmp_path: Path, declined: bool, kw: dict, method: str
) -> None:
    """Main: Y = 1 for 0.5 with converged=True, by either method."""
    result = bngsim.Simulator(_gate(tmp_path, declined=declined), method="ode", **kw).steady_state(
        method=method
    )
    assert result.converged
    c = np.asarray(result.concentrations)
    assert c[0] == 0.0
    assert abs(c[1] / 0.5 - 1.0) < OK


def test_a_run_that_starts_on_a_threshold_between_basins(tmp_path: Path) -> None:
    """X' = a(X) - 0.1·X, a = 1.1 above 5, 0.1 on [0.5, 5] and 0 below 0.5: from
    X0 = 5 the run falls to the middle steady state, X = 1 + 4·exp(-0.1 t). Main,
    with a declined law, followed its predictor below 0.5 and ended in the lower
    basin: X(200) = -6.4e-8."""
    model = _net(
        tmp_path,
        ["X() 5", "S() 1"],
        ["Xobs 1", "Sobs 2"],
        ["a() if(Xobs > 5, 1.1, if(Xobs < 0.5, 0.0, 0.1))/Sobs", "d() 0.1"],
        ["2 1,2 a", "1 0 d"],
        declined=True,
    )
    t, s = _run(model, 200.0, 41)
    assert _worst(s[:, 0], 1.0 + 4.0 * np.exp(-0.1 * t)) < OK


def test_a_model_with_no_jump_has_nothing_mended(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Control: MODEL1812040006 has no jump in it, and nothing of it is mended.
    Two smooth shapes read as jumps to a looser test. A species at 1.6e-8 is
    stepped by CVODE's floor, 2.5e-2, and its points back are negative amounts,
    where the saturating law it feeds is nothing like itself: 52 mends with
    those points taken. And a row whose terms cancel moves by a last bit of
    them: 2 mends against a floor of the row's value instead of its terms.
    Either moved the run off main's bits."""
    monkeypatch.setenv("BNGSIM_JAC_DEBUG", "1")
    model = bngsim.Model.from_sbml(str(data_dir / "MODEL1812040006.xml"))
    bngsim.Simulator(model, method="ode", jacobian="fd").run(
        t_span=(0.0, 100.0), n_points=101, rtol=RTOL, atol=ATOL
    )
    assert "across a jump" not in capfd.readouterr().err
