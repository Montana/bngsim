"""A Jacobian still non-finite after the nonnegative clamp is not handed to CVODE (issue #867).

``S -> ; C*k*sqrt(S)`` with S(0) = 1 and C = k = 1 empties S at t = 2: S(t) =
(1 - t/2)^2, then 0. Pinned at 0, d/dS sqrt(S) is infinite, and no clamp makes it
finite (GH #135's clamp retry is for a negative base). The dense Jacobian
callbacks (interpreted and compiled) kept the infinite matrix and returned
success. Its infinite pivot in S's row zeroed S's Newton correction on every
step, so CVODE's error test, which reads that correction, passed while S followed
the predictor off: S(40) = -2.27, and -2.1e6 beside an unrelated species that only
changes the step sequence, with no error. ``jacobian="fd"`` gets ~-2e-8.

Each callback now returns CVODE's recoverable code there, as the RHS callback does
for a non-finite right-hand side. The run then fails as a CVODE failure, so
``jacobian="auto"`` retries on the difference-quotient Jacobian (GH #176) and
returns what ``jacobian="fd"`` returns, and ``jacobian="analytical"`` raises,
naming the Jacobian (GH #336) and the derivative that is infinite at 0 rather than
the domain advice written for a right-hand side.

A ``run_batch`` row places the same bet. It used to call the core directly, with
no retry, so once the Jacobian was refused a batch raised where the same
``run()`` returned. That broke the batch/loop contract (GH #203). Where an event
pins S at exactly 0, main returned S = 0 on every path, so a batch row that
raised would have turned a correct answer into a failure.

The sparse (KLU) callbacks got the same change, but they had no silent failure to
fix. KLU already failed on the non-finite matrix, and the colored difference
quotient the retry falls back to differences the unclamped right-hand side, which
is NaN a hair below 0. So both sparse runs raised on main and still raise. That
test is a guard against the sparse path ever returning the drifted S, not a
regression test.
"""

from __future__ import annotations

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest

SINK = "species S; S = 1; C = 1; k = 1;\nR1: S -> ; C*k*sqrt(S)\n"
MODELS = {
    "S alone": SINK,
    # Y does not read S; it changes the step sequence, which took S to -2.1e6.
    "beside an unrelated species": SINK + "species Y; Y = 1; R2: Y -> ; 0.1*Y\n",
}
# An event sets S to exactly 0. No species is below zero and no rate law is
# non-finite there; only d/dS sqrt(S) is.
PINNED = "species S; S = 1; k = 0.1;\nR1: S -> ; k*sqrt(S)\nE1: at time >= 1: S = 0;\n"
CODEGEN = {"interpreted": False, "compiled": True}


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _sim(text: str, **kw) -> bngsim.Simulator:
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), "ode", **kw)
    if kw.get("codegen"):
        assert sim.codegen_backend in ("cc", "mir")
    return sim


def _species(sim: bngsim.Simulator) -> np.ndarray:
    return np.asarray(sim.run(t_span=(0.0, 40.0), n_points=5).species)


def _end(sim: bngsim.Simulator) -> np.ndarray:
    return _species(sim)[-1]


@pytest.mark.parametrize("codegen", CODEGEN.values(), ids=CODEGEN.keys())
@pytest.mark.parametrize("text", MODELS.values(), ids=MODELS.keys())
def test_auto_returns_what_the_difference_quotient_returns(text, codegen):
    sim = _sim(text, codegen=codegen)
    species = _species(sim)
    assert abs(species[-1, 0]) < 1e-6, f"S(40) = {species[-1, 0]}: S is empty from t = 2"
    assert sim.jacobian_strategy == "fd"
    # The retry selects the FD Jacobian on the same right-hand side, so it is the
    # explicit FD run bit for bit.
    assert np.array_equal(species, _species(_sim(text, jacobian="fd", codegen=codegen)))


@pytest.mark.parametrize("codegen", CODEGEN.values(), ids=CODEGEN.keys())
@pytest.mark.parametrize("text", MODELS.values(), ids=MODELS.keys())
def test_an_explicit_analytical_jacobian_fails_loudly(text, codegen):
    with pytest.raises(bngsim.SimulationError, match=r"CVODE integration failed.*Jacobian"):
        _end(_sim(text, jacobian="analytical", codegen=codegen))


@pytest.mark.parametrize("codegen", CODEGEN.values(), ids=CODEGEN.keys())
def test_the_failure_names_the_slope_not_the_domain(codegen):
    # The RHS advice ("a sqrt ... of a negative value stays nan ... Constrain the
    # species") is wrong here: S is exactly 0, inside the domain, and no
    # constraint on S removes an infinite slope at 0.
    with pytest.raises(bngsim.SimulationError) as info:
        _end(_sim(PINNED, jacobian="analytical", codegen=codegen))
    msg = str(info.value)
    assert "Jacobian returned a non-finite value" in msg, msg
    assert "Species below zero" not in msg, msg
    assert "the slope of sqrt(S)" in msg, msg
    assert "Constrain the species" not in msg, msg


@pytest.mark.parametrize("codegen", CODEGEN.values(), ids=CODEGEN.keys())
@pytest.mark.parametrize("text", [*MODELS.values(), PINNED], ids=[*MODELS.keys(), "S pinned at 0"])
def test_a_batch_row_falls_back_as_run_does(text, codegen):
    sim = _sim(text, codegen=codegen)
    rows = sim.run_batch(t_span=(0.0, 40.0), n_points=5, params=[{}, {}])
    expected = _species(_sim(text, codegen=codegen))
    for row in rows:
        species = np.asarray(row.species)
        assert abs(species[-1, 0]) < 1e-6, f"S(40) = {species[-1, 0]}"
        assert np.array_equal(species, expected)


@pytest.mark.parametrize("codegen", CODEGEN.values(), ids=CODEGEN.keys())
@pytest.mark.parametrize("text", MODELS.values(), ids=MODELS.keys())
def test_the_sparse_callbacks_do_not_return_a_drifted_trajectory(text, codegen):
    # A guard, not a regression test: see the module docstring. A run either
    # returns S empty or raises; it never returns S driven off.
    try:
        s_end = _end(_sim(text, codegen=codegen, force_sparse_linear_solver=True))[0]
    except bngsim.SimulationError as exc:
        assert "CVODE integration failed" in str(exc)
    else:
        assert abs(s_end) < 1e-6, f"S(40) = {s_end}"
    with pytest.raises(bngsim.SimulationError, match=r"CVODE integration failed.*Jacobian"):
        _end(_sim(text, jacobian="analytical", codegen=codegen, force_sparse_linear_solver=True))
