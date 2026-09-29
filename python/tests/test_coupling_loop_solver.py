"""The linear solver across the steps of a coupling loop.

A ``ReactionKernel`` loop is one integration cut into many short runs, each of
which re-enters the warm CVODE path through ``CVodeReInit``. Two pieces of
solver state used to be thrown away at every re-entry, although nothing about
the loop called for it:

* the GH #132 adaptive factorization count, which decides when the dense solver
  opted into by ``BNGSIM_LAPACK_DENSE`` moves from the built-in LU to the BLAS
  factor. It restarted at every step, so a loop whose steps each factor a few
  times never earned the BLAS factor a single run of the same length would.
  ``run_until`` (and so ``advance``) now marks its run as continuing the
  trajectory, which keeps the count; an independent ``run()`` still restarts it.
* KLU's symbolic analysis. ``CVodeReInit`` re-initializes the linear solver, and
  SUNDIALS' KLU module takes that as a cue to redo the ordering and a full
  factorization, although the Newton matrix's pattern is the model's and cannot
  change. The warm path now refactors in the kept order instead.

Both are costs, not answers: the tests pin the counts the first one is about,
and that the second leaves the loop's trajectory where a cold rebuild at every
step puts it.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"

# LinearSolverKind codes — mirror include/bngsim/result.hpp.
LS_DENSE, LS_KLU, LS_LAPACK = 0, 1, 2

# The GH #132 gate's defaults (src/lapack_dense_linsol.cpp): the first K
# factorizations of a count are built-in, and only systems of at least MIN_N
# ever take the BLAS factor.
GATE_K = 5
GATE_MIN_N = 256

_WARM_OFF = "BNGSIM_NO_WARM_CVODE"


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not _NETS_DIR.is_dir():
        pytest.skip(f"benchmark net corpus not available: {_NETS_DIR}")
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
    return str(path)


@pytest.mark.skipif(not bngsim.HAS_LAPACK_DENSE, reason="no BLAS dense backend")
class TestAdaptiveGateAcrossSteps:
    """``egfr_net`` (356 species, over GATE_MIN_N) on the opted-in dense solver.

    ``force_dense_linear_solver`` keeps it off the LU-fill route, which takes the
    BLAS factor from the first factorization and so has no gate to test.
    """

    @pytest.fixture
    def kernel(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_LAPACK_DENSE", "1")
        for var in ("BNGSIM_LAPACK_DENSE_K", "BNGSIM_LAPACK_DENSE_MIN_N", _WARM_OFF):
            monkeypatch.delenv(var, raising=False)
        model = bngsim.Model.from_net(_net("egfr_net.net"))
        assert model.n_species >= GATE_MIN_N
        return bngsim.ReactionKernel(model, method="ode", force_dense_linear_solver=True)

    @staticmethod
    def _loop(kernel, n_steps: int = 12, dt: float = 1.0) -> list[dict]:
        stats = []
        for _ in range(n_steps):
            kernel.advance(dt)
            stats.append(kernel.last_result.solver_stats)
        return stats

    def test_steps_after_the_first_factor_with_blas_throughout(self, kernel):
        """Once the loop as a whole has factored past K, a later step takes the
        BLAS factor from its first factorization. With the count restarted per
        step, each step's first K factorizations were built-in again."""
        stats = self._loop(kernel)
        assert all(st["linear_solver"] == LS_LAPACK for st in stats)
        assert stats[0]["n_jac_evals"] > GATE_K  # the first step crosses the gate
        for k, st in enumerate(stats[1:], start=1):
            assert st["n_jac_evals"] > 0
            # Every Jacobian evaluation is followed by a factorization.
            assert st["n_dense_blas_factorizations"] >= st["n_jac_evals"], (k, st)

    def test_the_count_is_per_run(self, kernel):
        """A continuing run keeps the solver's count but reports only its own
        factorizations: each is followed by at least one Newton iteration, so a
        run cannot report more than it iterated — which a count carried over
        from the steps before it would exceed."""
        stats = self._loop(kernel)
        blas = [st["n_dense_blas_factorizations"] for st in stats]
        assert sum(blas) > max(st["n_nonlin_iters"] for st in stats)  # premise
        for st in stats:
            assert st["n_dense_blas_factorizations"] <= st["n_nonlin_iters"], st

    def test_an_independent_run_restarts_the_count(self, kernel):
        self._loop(kernel)
        sim = kernel.simulator
        assert sim._continues_trajectory is False  # run_until's flag does not leak
        short = sim.run(t_span=(kernel.time, kernel.time + 1e-3), n_points=2).solver_stats
        assert short["linear_solver"] == LS_LAPACK
        assert 0 < short["n_jac_evals"] <= GATE_K  # premise: a run inside the gate
        assert short["n_dense_blas_factorizations"] == 0


@pytest.mark.skipif(not bngsim.HAS_KLU, reason="KLU not compiled")
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("metapop_sir_100.net", {}),
        # A factor that fills in: more for a stale pivot order to get wrong.
        ("egfr_net.net", {"force_sparse_linear_solver": True}),
    ],
)
def test_warm_klu_loop_matches_a_cold_rebuild_per_step(monkeypatch, name, kwargs):
    """The warm KLU path refactors in the order of its first analysis across every
    re-entry, including across state jumps a coupling loop makes (species scaled
    by 0.2x-5x); a cold run rebuilds the solver and re-analyzes each step. The two
    loops must land on the same state. (On these networks KLU's
    diagonal-preferring pivot keeps the same pivots, so they agree bit for bit;
    the contract asserted is agreement to integrator accuracy.)"""

    def loop(warm: bool) -> np.ndarray:
        if warm:
            monkeypatch.delenv(_WARM_OFF, raising=False)
        else:
            monkeypatch.setenv(_WARM_OFF, "1")
        kernel = bngsim.ReactionKernel(bngsim.Model.from_net(_net(name)), method="ode", **kwargs)
        rng = np.random.default_rng(7)
        for step in range(15):
            state = kernel.get_state()
            if step % 3 == 1:
                kernel.set_state(state * rng.uniform(0.2, 5.0, size=state.shape))
            kernel.advance(10.0)
            assert kernel.last_result.solver_stats["linear_solver"] == LS_KLU
        return kernel.get_state()

    warm, cold = loop(True), loop(False)
    scale = float(np.max(np.abs(cold)))
    np.testing.assert_allclose(warm, cold, rtol=1e-8, atol=1e-10 * scale)
