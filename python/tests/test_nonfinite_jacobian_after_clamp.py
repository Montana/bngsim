"""A Jacobian still non-finite after the nonnegative clamp is not handed to CVODE (issue #867).

``S -> ; C*k*sqrt(S)`` with S(0) = 1 and C = k = 1 empties S at t = 2: S(t) =
(1 - t/2)^2, then 0. Pinned at 0, d/dS sqrt(S) is infinite, and no clamp makes it
finite (GH #135's clamp retry is for a negative base). The four Jacobian
callbacks (dense and sparse, interpreted and compiled) kept the infinite matrix
and returned success. Its infinite pivot in S's row zeroed S's Newton correction
on every step, so CVODE's error test, which reads that correction, passed while S
followed the predictor off: S(40) = -2.27, and -2.1e6 beside an unrelated species
that only changes the step sequence, with no error. ``jacobian="fd"`` gets ~-2e-8.

Each callback now returns CVODE's recoverable code there, as the RHS callback does
for a non-finite right-hand side. The run then fails as a CVODE failure, so
``jacobian="auto"`` retries on the difference-quotient Jacobian (GH #176) and
returns what ``jacobian="fd"`` returns, and ``jacobian="analytical"`` raises,
naming the Jacobian (GH #336).
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
CODEGEN = {"interpreted": False, "compiled": True}


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _sim(text: str, **kw) -> bngsim.Simulator:
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), "ode", **kw)
    if kw.get("codegen"):
        assert sim.codegen_backend in ("cc", "mir")
    return sim


def _end(sim: bngsim.Simulator) -> np.ndarray:
    return np.asarray(sim.run(t_span=(0.0, 40.0), n_points=5).species)[-1]


@pytest.mark.parametrize("codegen", CODEGEN.values(), ids=CODEGEN.keys())
@pytest.mark.parametrize("text", MODELS.values(), ids=MODELS.keys())
def test_auto_returns_what_the_difference_quotient_returns(text, codegen):
    sim = _sim(text, codegen=codegen)
    s_end = _end(sim)[0]
    assert abs(s_end) < 1e-6, f"S(40) = {s_end}: S is empty from t = 2"
    assert sim._ode_jacobian_fell_back is True
    np.testing.assert_allclose(s_end, _end(_sim(text, jacobian="fd"))[0], atol=1e-9)


@pytest.mark.parametrize("codegen", CODEGEN.values(), ids=CODEGEN.keys())
@pytest.mark.parametrize("text", MODELS.values(), ids=MODELS.keys())
def test_an_explicit_analytical_jacobian_fails_loudly(text, codegen):
    with pytest.raises(bngsim.SimulationError, match=r"CVODE integration failed.*Jacobian"):
        _end(_sim(text, jacobian="analytical", codegen=codegen))


@pytest.mark.parametrize("codegen", CODEGEN.values(), ids=CODEGEN.keys())
@pytest.mark.parametrize("text", MODELS.values(), ids=MODELS.keys())
def test_the_sparse_callbacks_do_not_return_a_drifted_trajectory(text, codegen):
    # The sparse (KLU) analytical callbacks hand the run to CVODE the same way.
    # A run either returns S empty or raises; it never returns S driven off.
    try:
        s_end = _end(_sim(text, codegen=codegen, force_sparse_linear_solver=True))[0]
    except bngsim.SimulationError as exc:
        assert "CVODE integration failed" in str(exc)
    else:
        assert abs(s_end) < 1e-6, f"S(40) = {s_end}"
    with pytest.raises(bngsim.SimulationError, match=r"CVODE integration failed.*Jacobian"):
        _end(_sim(text, jacobian="analytical", codegen=codegen, force_sparse_linear_solver=True))
