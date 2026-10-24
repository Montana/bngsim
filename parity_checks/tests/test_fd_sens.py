"""The finite-difference sensitivity oracle (issue #804, parity_checks/fd_sens).

The step, the Richardson extrapolation, the convergence gate and the columns a
difference can judge, on synthetic data; then one real model end to end, with
the reference corrupted to show the verdict catches a wrong column, with the
analytic side corrupted toward zero to show the silent zero is judged rather
than left out, and with an unconverged, non-finite or failed reference to show
it is refused rather than blamed on the analytic side.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_FD = Path(__file__).resolve().parent.parent / "fd_sens"
# The variables that pick the repository git answers for (python/tests/_net_corpus.py).
_REPO_LOCATING = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)


def _load():
    spec = importlib.util.spec_from_file_location("fd_sens_run", _FD / "fd_sens_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load()


def test_the_step_is_relative_and_absolute_at_zero():
    assert F.fd_step(2.0) == pytest.approx(2.0 * F.FD_REL_STEP)
    assert F.fd_step(-4.0) == pytest.approx(4.0 * F.FD_REL_STEP)
    assert F.fd_step(0.0) == F.FD_REL_STEP


def test_richardson_removes_the_h_squared_term():
    true = np.array([1.5, -2.0])
    a = np.array([0.3, 0.7])
    h = 1e-2
    d_h, d_h2 = true + a * h**2, true + a * (h / 2) ** 2
    np.testing.assert_allclose(F.richardson(d_h, d_h2), true, rtol=1e-14)


def _tensor(*columns):
    """``(n_t, n_sp, n_p)`` from per-(species, parameter) time series, given as
    ``columns[parameter][species]``."""
    return np.stack([np.stack(c, axis=-1) for c in columns], axis=-1)


def test_step_disagreement_is_on_each_columns_own_scale():
    t = np.array([0.0, 1e-9, 10.0])
    ref = _tensor([t])
    assert F.step_disagreement(ref, ref, np.zeros_like(ref)) == 0.0
    # A cell near zero moving by 1e-9 is nothing on a column whose peak is 10.
    gap = _tensor([np.array([0.0, 1e-9, 0.0])])
    assert F.step_disagreement(ref, ref, gap) == pytest.approx(1e-10)
    # Noise: the two quotients of an unresolved column.
    assert F.step_disagreement(ref, ref, _tensor([np.array([0.0, 6.0, 6.0])])) > 0.5
    z = np.zeros((3, 1, 1))
    assert F.step_disagreement(z, z, z) == 0.0


def test_step_disagreement_judges_each_species_as_the_verdict_does():
    """The verdict judges each (species, parameter) column on its own peak, so the
    gate does too. Species 1 peaks at 0.1, past differ's significance floor of
    1e-3 of the tensor's 10, and its gap of 1e-3 is 1% of that; beside species
    0, the parameter column's peak would read it as 1e-4."""
    big, small = np.array([1.0, 5.0, 10.0]), np.array([0.02, 0.05, 0.1])
    ref = _tensor([big, small])
    gap = _tensor([np.zeros(3), np.array([0.0, 0.0, 1e-3])])
    assert F.step_disagreement(ref, ref, gap) == pytest.approx(1e-2)
    # A species under the significance floor is not one the verdict counts.
    tiny = np.array([0.0, 1e-6, 2e-6])
    ref = _tensor([big, tiny])
    gap = _tensor([np.zeros(3), np.array([0.0, 0.0, 2e-6])])
    assert F.step_disagreement(ref, ref, gap) == 0.0


def test_step_disagreement_covers_the_columns_the_movement_gate_judges():
    """mm_tqssa_stiff's shape: dE/dE_0 = 1 holds still and sets the peak, so
    differ's movement gate takes its scale from a column of 1e-12 noise, which
    the peak test alone would skip, and fails it. The gate must see it first."""
    still = np.ones(4)
    noise = np.array([0.0, -2.8e-12, 4.4e-12, 8.7e-18])
    ref = _tensor([still, noise])
    gap = _tensor([np.zeros(4), np.array([0.0, 2.1e-12, 3.3e-12, 6.6e-18])])
    assert F.step_disagreement(ref, ref, gap) > 0.5


def test_a_reference_that_is_not_finite_never_passes_the_gate():
    ref = _tensor([np.array([1.0, np.nan, 2.0])])
    assert F.step_disagreement(ref, ref, np.zeros_like(ref)) == float("inf")
    ok = _tensor([np.array([1.0, 2.0, 3.0])])
    assert F.step_disagreement(ok, ok, ref) == float("inf")
    # A non-finite analytic cell is the verdict's to fail, and sets no scale here.
    bad = _tensor([np.array([1.0, np.inf, 3.0])])
    assert F.step_disagreement(bad, ok, np.zeros_like(ok)) == 0.0


def test_only_columns_clearing_both_floors_are_judged():
    # (n_t, n_sp, n_p): column 0 large, column 1 under the verdict's atol/|p|,
    # column 2 over that but under the difference's own noise (a tiny parameter
    # value makes a tiny step).
    analytic = np.zeros((3, 2, 3))
    analytic[..., 0] = 1.0
    analytic[..., 1] = 1e-12
    analytic[..., 2] = 1e-3
    values = [2.0, 2.0, 1e-9]
    judged = F.judgeable_columns(analytic, analytic, x_max=1e4, values=values, atol=1e-12)
    assert judged == [0]


def test_a_column_either_side_resolves_is_judged():
    """A column wrong in either direction is judged: one the difference resolves
    but the analytic run reads as zero (the silent zero), and one holding a
    non-finite analytic cell, which the verdict then fails."""
    values, atol = [2.0, 2.0, 2.0], 1e-12
    analytic = np.ones((3, 2, 3))
    reference = np.ones((3, 2, 3))
    analytic[..., 1] = 0.0
    analytic[1, 0, 2] = np.nan
    judged = F.judgeable_columns(analytic, reference, x_max=10.0, values=values, atol=atol)
    assert judged == [0, 1, 2]
    # Under both floors on both sides, a column is still left out.
    reference[..., 1] = 1e-14
    judged = F.judgeable_columns(analytic, reference, x_max=10.0, values=values, atol=atol)
    assert judged == [0, 2]


def test_a_scheduler_result_still_says_what_happened():
    spec = {"model_id": "m", "sens_method": "staggered"}
    timeout = F.to_result(spec, {"status": "timeout", "cap": 300.0, "wall": 301.2}, {})
    assert timeout.outcome == "TIMEOUT"
    assert "300" in timeout.exception
    assert timeout.wall_sec == 301.2
    dead = F.to_result(spec, {"status": "dead", "exitcode": -11}, {})
    assert dead.outcome == "EXCEPTION"
    assert "-11" in dead.exception


def _worker(spec):
    out = []

    class Q:
        def put(self, item):
            out.append(item)

    F._worker(spec, Q())
    (res,) = out
    return res


@pytest.fixture
def spec():
    pytest.importorskip("bngsim")
    jobs = json.loads((_FD / "fd_sens_jobs.json").read_text())["jobs"]
    job = next(j for j in jobs if j["model_id"] == "derived_rate_const")
    return {
        "key": "derived_rate_const|sens/staggered",
        "model_id": job["model_id"],
        "path": job["path"],
        "t_start": 0.0,
        "t_end": float(job["t_end"]),
        "n_points": int(job.get("n_points", 51)),
        "rtol": F.DEFAULT_RTOL,
        "atol": F.DEFAULT_ATOL,
        "fd_tol": 1e-3,
        "param_cap": 6,
        "sens_method": "staggered",
        "timeout": 120.0,
    }


def test_a_real_model_passes_against_its_own_differences(spec):
    res = _worker(spec)
    assert res["status"] == "pass", res
    assert res["n_columns_judged"] == len(res["params"]) == 3
    assert res["fd_step_disagreement"] < 1e-4
    assert res["timing"]["bngsim"]["work"]["n_steps"] > 0


def test_a_wrong_column_is_a_diff(spec, monkeypatch):
    """The reference 5% off in one column: the verdict says DIFF."""
    real = F.fd_tensor

    def off(path, params, values, job):
        ref, dis = real(path, params, values, job)
        ref = ref.copy()
        ref[..., 1] *= 1.05
        return ref, dis

    monkeypatch.setattr(F, "fd_tensor", off)
    res = _worker(spec)
    assert res["status"] == "diff", res
    assert res["value"] > 1e-3


def test_an_unconverged_reference_is_refused_not_blamed(spec, monkeypatch):
    real = F.fd_tensor

    def noisy(path, params, values, job):
        ref, gap = real(path, params, values, job)
        gap = gap.copy()
        gap[..., 0] = 0.5 * np.abs(ref[..., 0]).max()
        return ref, gap

    monkeypatch.setattr(F, "fd_tensor", noisy)
    res = _worker(spec)
    assert res["status"] == "bad_test", res
    assert "did not converge" in res["why"]


def test_a_reference_that_is_not_finite_is_refused_not_blamed(spec, monkeypatch):
    real = F.fd_tensor

    def broken(path, params, values, job):
        ref, gap = real(path, params, values, job)
        ref = ref.copy()
        ref[3, 0, 1] = np.nan
        return ref, gap

    monkeypatch.setattr(F, "fd_tensor", broken)
    res = _worker(spec)
    assert res["status"] == "bad_test", res
    assert "not finite" in res["why"]


def test_a_plain_run_that_raises_is_the_references_failure(spec, monkeypatch):
    def fails(path, params, values, job):
        raise RuntimeError("CVODE: too much work")

    monkeypatch.setattr(F, "fd_tensor", fails)
    res = _worker(spec)
    assert res["status"] == "reference_failed", res
    assert "too much work" in res["exception"]


def _analytic_altered(monkeypatch, alter):
    """Make every analytic run hand back ``alter(sensitivities)``."""
    import bngsim

    real = bngsim.Simulator

    class Result:
        def __init__(self, run):
            self._run = run

        def __getattr__(self, name):
            return getattr(self._run, name)

        @property
        def sensitivities(self):
            return alter(np.array(self._run.sensitivities, dtype=float))

    class Simulator(real):
        def run(self, *args, **kwargs):
            return Result(super().run(*args, **kwargs))

    monkeypatch.setattr(bngsim, "Simulator", Simulator)


@pytest.mark.parametrize(
    "wrong",
    [
        pytest.param(lambda col: np.zeros_like(col), id="zero"),
        pytest.param(lambda col: col * 1e-12, id="vanishing"),
        pytest.param(lambda col: np.full_like(col, np.nan), id="nan"),
    ],
)
def test_an_analytic_column_wrong_toward_zero_is_a_diff(spec, monkeypatch, wrong):
    """The silent zero: an analytic column that comes back zero, vanishingly
    small or NaN where the difference resolves a sensitivity is judged, and
    fails, rather than being left out of the comparison."""

    def alter(sx):
        sx[..., 1] = wrong(sx[..., 1])
        return sx

    _analytic_altered(monkeypatch, alter)
    res = _worker(spec)
    assert res["status"] == "diff", res
    assert res["n_columns_judged"] == 3, res


def test_the_manifest_names_tracked_models_and_why_some_are_left_out():
    payload = json.loads((_FD / "fd_sens_jobs.json").read_text())
    repo = _FD.parents[1]
    ids = [j["model_id"] for j in payload["jobs"]]
    assert len(ids) == len(set(ids)) >= 30
    for job in payload["jobs"]:
        assert job["path"].endswith(".net"), job
        assert job["t_end"] > 0
    for model_id, reason in payload["excluded"].items():
        assert model_id not in ids
        assert len(reason) > 40, f"{model_id}: give the measurement, not a label"
    if (repo / ".git").exists():
        import os
        import subprocess

        # An exported GIT_DIR or GIT_WORK_TREE would answer for another repository.
        env = {k: v for k, v in os.environ.items() if k not in _REPO_LOCATING}
        tracked = set(
            subprocess.run(
                ["git", "ls-files", *[j["path"] for j in payload["jobs"]]],
                cwd=repo,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.split()
        )
        assert tracked == {j["path"] for j in payload["jobs"]}, "a manifest model is not tracked"
