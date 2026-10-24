"""`parity_golden.py --verify` (issue #582): the committed goldens as a gate.

A re-run job is identical when its checksum matches, within tolerance when each
fingerprint stat satisfies |x - y| <= rtol·|y| + atol, and drift otherwise. A
job that failed to run, produced nothing, or has no golden entry fails, and so
does a run that compared fewer than --min-compared jobs. Hermetic: synthetic
.gdat output in a temporary sweep tree, no engine.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_BNG_PARITY = Path(__file__).resolve().parent.parent / "bng_parity"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _BNG_PARITY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PG = _load("parity_golden")
LOAD = PG._load_parity_diff().load_normalized
JOB = SimpleNamespace(model_id="fast/x/toy/toy.bngl", method="ode")


def _write_run(sweep: Path, values: np.ndarray, status: str = "ok") -> Path:
    out = sweep / "fast/x/toy/toy/det"
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, values.shape[0])
    rows = "\n".join(
        " ".join(f"{x:.12e}" for x in (ti, *vi)) for ti, vi in zip(t, values, strict=False)
    )
    (out / "model.gdat").write_text("#          time            A            B\n" + rows + "\n")
    (out / "_run.log").write_text(f"# status={status}\n")
    return out


def _golden(tmp: Path, values: np.ndarray) -> Path:
    """golden.json for JOB, built from `values` the way the generator builds it."""
    out = _write_run(tmp / "golden_src", values)
    (checksum, fingerprint, _), _ = PG.build_golden_for_job(out, LOAD)
    path = tmp / "golden.json"
    path.write_text(
        json.dumps(
            {
                "_meta": {},
                "golden": [
                    {
                        "model_id": JOB.model_id,
                        "method": "ode",
                        "checksum": checksum,
                        "fingerprint": fingerprint,
                    }
                ],
            }
        )
    )
    return path


BASE = np.column_stack([np.linspace(1.0, 2.0, 11), np.linspace(0.0, 3e-8, 11)])


def _verify(tmp: Path, values: np.ndarray, *, status="ok", min_compared=1, capsys=None):
    golden = _golden(tmp, BASE)
    _write_run(tmp / "sweep", values, status)
    code = PG.verify_against_golden(golden, tmp / "sweep", [JOB], LOAD, 1e-5, 1e-6, min_compared)
    return code, capsys.readouterr().out if capsys else ""


def test_an_identical_run_passes(tmp_path, capsys):
    code, out = _verify(tmp_path, BASE, capsys=capsys)
    assert code == 0 and "IDENTICAL" in out


def test_noise_below_the_absolute_tolerance_passes(tmp_path, capsys):
    """B near zero moved by 7e-9, 23% of itself and under atol: within, where a
    purely relative comparison read it as drift (e2f-rb's CKI_Inhibition)."""
    noisy = BASE.copy()
    noisy[:, 1] *= 0.77
    code, out = _verify(tmp_path, noisy, capsys=capsys)
    assert code == 0 and "WITHIN_TOL" in out


def test_a_relative_change_past_rtol_is_drift(tmp_path, capsys):
    moved = BASE.copy()
    moved[-1, 0] *= 1.001
    code, out = _verify(tmp_path, moved, capsys=capsys)
    assert code == 1 and "DRIFT" in out


def test_a_failed_run_fails(tmp_path, capsys):
    code, out = _verify(tmp_path, BASE, status="timeout", capsys=capsys)
    assert code == 1 and "RUN_FAILED" in out


def test_a_job_with_no_golden_entry_fails(tmp_path, capsys):
    golden = _golden(tmp_path, BASE)
    _write_run(tmp_path / "sweep", BASE)
    other = SimpleNamespace(model_id="fast/x/toy/toy2.bngl", method="ode")
    toy2 = tmp_path / "sweep/fast/x/toy/toy2/det"
    toy2.mkdir(parents=True)
    for f in ("model.gdat", "_run.log"):
        (toy2 / f).write_text((tmp_path / "sweep/fast/x/toy/toy/det" / f).read_text())
    code = PG.verify_against_golden(golden, tmp_path / "sweep", [JOB, other], LOAD, 1e-5, 1e-6, 1)
    assert code == 1 and "NO_GOLDEN" in capsys.readouterr().out


def test_comparing_too_little_fails(tmp_path, capsys):
    code, out = _verify(tmp_path, BASE, min_compared=2, capsys=capsys)
    assert code == 1 and "too little" in out


def test_a_structural_change_is_infinitely_far():
    a = {"artifacts": {"model.gdat": {"n_time": 2, "vars": {}}}}
    b = {"artifacts": {"other.gdat": {"n_time": 2, "vars": {}}}}
    assert PG.job_fingerprint_distance(a, b, 1e-5, 1e-6) == float("inf")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_blow_up_on_one_side_is_infinitely_far(bad):
    def fp(x):
        return {
            "artifacts": {
                "m.gdat": {
                    "n_time": 2,
                    "vars": {"A": {"last": x, "min": 0.0, "max": 1.0, "mean": 0.5}},
                }
            }
        }

    assert PG.job_fingerprint_distance(fp(bad), fp(1.0), 1e-5, 1e-6) == float("inf")


def test_a_tolerance_override_judges_its_model_by_its_own(tmp_path, capsys, monkeypatch):
    """0.1% is drift at the default rtol and within a model's 1e-2 override."""
    moved = BASE.copy()
    moved[-1, 0] *= 1.001
    monkeypatch.setitem(PG.VERIFY_TOLERANCE_OVERRIDES, JOB.model_id, (1e-2, 1e-6, "test"))
    code, out = _verify(tmp_path, moved, capsys=capsys)
    assert code == 0 and "WITHIN_TOL" in out and "rtol 0.01" in out


def test_every_override_is_a_gated_model_with_a_reason():
    for model_id, (rtol, atol, reason) in PG.VERIFY_TOLERANCE_OVERRIDES.items():
        assert model_id in PG.GATED_MODELS, model_id
        assert rtol > 0 and atol > 0
        assert len(reason) > 40, f"{model_id}: give the measurement, not a label"


def test_every_gated_model_is_deterministic_in_the_manifest():
    """--gate sweeps --regime ode, so a stochastic gated model would never run
    and every pull request's gate would fail on a count."""
    _meta, jobs = PG.read_manifest(PG.HERE / "jobs.json")
    method = {j.model_id: j.method for j in jobs}
    assert PG.GATED_MODELS
    for model_id in PG.GATED_MODELS:
        assert method.get(model_id) == "ode", (model_id, method.get(model_id))


def test_a_run_that_ends_elsewhere_is_drift():
    """Same rows, same values (a model at rest), another end time."""

    def fp(t_last):
        return {
            "artifacts": {
                "m.gdat": {
                    "n_time": 2,
                    "t_last": t_last,
                    "vars": {"A": {"last": 1.0, "min": 1.0, "max": 1.0, "mean": 1.0}},
                }
            }
        }

    assert PG.job_fingerprint_distance(fp(500.0), fp(500.0), 1e-5, 1e-6) == 0.0
    assert PG.job_fingerprint_distance(fp(1000.0), fp(500.0), 1e-5, 1e-6) > 1.0


def _stale_toy2(tmp: Path, values: np.ndarray) -> SimpleNamespace:
    """An earlier run's output for a second job, left in the sweep root."""
    toy2 = tmp / "sweep/fast/x/toy/toy2/det"
    toy2.mkdir(parents=True)
    src = _write_run(tmp / "stale_src", values)
    for f in ("model.gdat", "_run.log"):
        (toy2 / f).write_text((src / f).read_text())
    return SimpleNamespace(model_id="fast/x/toy/toy2.bngl", method="ode")


def test_output_the_sweep_did_not_produce_is_not_judged(tmp_path, capsys):
    """The default sweep root keeps every earlier run's output: verify judged
    all 895 jobs of a July sweep under a nine-model --gate run."""
    golden = _golden(tmp_path, BASE)
    _write_run(tmp_path / "sweep", BASE)
    stale = _stale_toy2(tmp_path, BASE * 2.0)  # would be NO_GOLDEN if judged
    code = PG.verify_against_golden(
        golden, tmp_path / "sweep", [JOB, stale], LOAD, 1e-5, 1e-6, 1, swept={JOB.model_id}
    )
    out = capsys.readouterr().out
    assert code == 0 and "1 compared" in out and "toy2" not in out


def test_stale_output_does_not_fill_the_compared_floor(tmp_path, capsys):
    golden = _golden(tmp_path, BASE)
    _write_run(tmp_path / "sweep", BASE)
    stale = _stale_toy2(tmp_path, BASE)
    code = PG.verify_against_golden(
        golden, tmp_path / "sweep", [JOB, stale], LOAD, 1e-5, 1e-6, 2, swept={JOB.model_id}
    )
    assert code == 1 and "too little" in capsys.readouterr().out


def test_an_expected_model_the_sweep_left_nothing_for_fails_by_name(tmp_path, capsys):
    golden = _golden(tmp_path, BASE)
    _write_run(tmp_path / "sweep", BASE)
    gone = "fast/x/toy/gone.bngl"
    code = PG.verify_against_golden(
        golden, tmp_path / "sweep", [JOB], LOAD, 1e-5, 1e-6, 1, expected=[JOB.model_id, gone]
    )
    lines = capsys.readouterr().out.splitlines()
    assert code == 1
    assert [ln.split()[1] for ln in lines if ln.startswith(gone)] == ["NOT_SWEPT"]


def test_the_swept_set_is_the_sweeps_own_results():
    summary = {
        "root": "/m",
        "results": [
            {"bngl": "/m/fast/a/b.bngl", "role": "det"},
            {"bngl": "/m/slow/c/d.bngl", "role": "seed1"},
            {"bngl": "/elsewhere/e.bngl", "role": "det"},
        ],
    }
    assert PG._swept_model_ids(summary) == {"fast/a/b.bngl", "slow/c/d.bngl"}
    assert PG._swept_model_ids({"root": "/m"}) is None
    assert PG._swept_model_ids({}) is None


def _g(model_id: str, checksum: str, trajectory=None):
    return PG.Golden(model_id, "ode", checksum, {"artifacts": {}}, trajectory)


def test_merge_replaces_only_the_fresh_entries_and_their_replay_verdicts():
    old_meta = {
        "generated": "2026-07-22",
        "versions": {"bngsim": "0.11.35"},
        "protocol_replay": {
            "a": {"dirty_carryover": True, "replayed": True},
            "b": {"dirty_carryover": True, "replayed": True},
            "c": {"dirty_carryover": True, "replayed": False},
        },
        "regenerations": [{"models": ["a"]}],
    }
    old = [_g("a", "a0", "trajectories/a.json"), _g("b", "b0"), _g("c", "c0")]
    fresh = [_g("b", "b1"), _g("c", "c1")]
    meta = {
        "git_rev": "abc",
        "versions": {"bngsim": "0.16.0"},
        "bionetgen_commit": "bng",
        "bngsim_build_commit": "abc",
    }
    # b fell back this run; c recorded no verdict this run.
    replay = {"b": {"dirty_carryover": True, "replayed": False}}
    merged, golden = PG.merge_regeneration(old_meta, old, fresh, meta, replay)
    assert [(g.model_id, g.checksum) for g in golden] == [("a", "a0"), ("b", "b1"), ("c", "c1")]
    assert merged["generated"] == "2026-07-22" and merged["versions"] == {"bngsim": "0.11.35"}
    assert merged["protocol_replay"] == {
        "a": {"dirty_carryover": True, "replayed": True},
        "b": {"dirty_carryover": True, "replayed": False},
    }
    assert merged["protocol_replay_summary"]["fallback"] == 1
    assert merged["protocol_replay_summary"]["models"] == 2
    assert merged["n_trajectories"] == 1
    assert merged["regenerations"][0] == {"models": ["a"]}
    last = merged["regenerations"][-1]
    assert last["models"] == ["b", "c"] and last["versions"] == {"bngsim": "0.16.0"}
