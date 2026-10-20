"""The finite-difference sensitivity oracle (issue #804, parity_checks/fd_sens).

The step, the Richardson extrapolation, the convergence gate and the columns a
difference can judge, on synthetic data; then one real model end to end, with
the reference corrupted to show the verdict catches a wrong column, and with an
unconverged reference to show the gate refuses it rather than blaming the
analytic side.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_FD = Path(__file__).resolve().parent.parent / "fd_sens"


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


def test_step_disagreement_is_on_the_columns_own_scale():
    big = np.array([0.0, 1e-9, 10.0])
    assert F.step_disagreement(big, big) == 0.0
    # A cell near zero moving by 1e-9 is nothing on a column whose peak is 10.
    assert F.step_disagreement(big + [0, 1e-9, 0], big) == pytest.approx(1e-10)
    # Noise: the two quotients of an unresolved column.
    assert F.step_disagreement(np.array([5.0, -3.0]), np.array([-4.0, 2.0])) > 0.5
    assert F.step_disagreement(np.zeros(3), np.zeros(3)) == 0.0


def test_only_columns_clearing_both_floors_are_judged():
    # (n_t, n_sp, n_p): column 0 large, column 1 under the verdict's atol/|p|,
    # column 2 over that but under the difference's own noise (a tiny parameter
    # value makes a tiny step).
    analytic = np.zeros((3, 2, 3))
    analytic[..., 0] = 1.0
    analytic[..., 1] = 1e-12
    analytic[..., 2] = 1e-3
    values = [2.0, 2.0, 1e-9]
    judged = F.judgeable_columns(analytic, x_max=1e4, values=values, atol=1e-12)
    assert judged == [0]


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
        ref, dis = real(path, params, values, job)
        return ref, [0.5] + list(dis[1:])

    monkeypatch.setattr(F, "fd_tensor", noisy)
    res = _worker(spec)
    assert res["status"] == "bad_test", res
    assert "did not converge" in res["why"]


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
        import subprocess

        tracked = set(
            subprocess.run(
                ["git", "ls-files", *[j["path"] for j in payload["jobs"]]],
                cwd=repo,
                capture_output=True,
                text=True,
            ).stdout.split()
        )
        assert tracked == {j["path"] for j in payload["jobs"]}, "a manifest model is not tracked"
