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
