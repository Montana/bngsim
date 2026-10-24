"""The S4 agreement check judges the engines against a converged reference
(issue #1250).

It compared libRoadRunner's and COPASI's final state with bngsim's at the
timing tolerance and called the model in agreement under 5%. On Koo2013 at
rtol 1e-6 the final state swings by up to 7% with the number of output rows,
in every engine, so the verdict measured the tolerance, and the grid passed
or failed it. Integrated at rtol 1e-12 the three engines agree to 2e-8.

The check now integrates a reference at ``REF_RTOL`` (libRoadRunner, or
bngsim where it fails). ``agree`` asks whether each engine at those
tolerances is within ``AGREE_TOL`` of it, and each engine's error at the
figure's tolerances is reported beside it, not judged. The six S4 models are
not in a checkout, so this runs the check on a first-order decay.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("roadrunner")

from .test_pickle_refusals import DECAY_SBML  # k = 0.1, X(0) = 100

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/suites/ode_engines_s4_sbml/check_sbml_engine_agreement.py"
)


def _check(tmp_path, monkeypatch, rtol=1e-6, atol=1e-9):
    """The check's module, set up to run on the decay alone."""
    spec = importlib.util.spec_from_file_location("s4_agreement_check", _SCRIPT)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    (tmp_path / "models" / "DECAY").mkdir(parents=True)
    (tmp_path / "models" / "DECAY" / "decay.xml").write_text(DECAY_SBML)
    jobs = {
        "jobs": [
            {
                "model_id": "DECAY",
                "model": "models/DECAY/decay.xml",
                "params": {
                    "t_start": 0.0,
                    "t_end": 20.0,
                    "n_points": 21,
                    "rtol": rtol,
                    "atol": atol,
                },
            }
        ]
    }
    (tmp_path / "ode_jobs.json").write_text(json.dumps(jobs))
    monkeypatch.setattr(check, "MODELS", [("DECAY", "Decay", 1)])
    monkeypatch.setattr(check, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(check, "ODE_JOBS", tmp_path / "ode_jobs.json")

    # COPASI is not a test dependency, and importing it sets the process's
    # locale, which later tests read their subprocesses' output under.
    def no_copasi(*_):
        raise ModuleNotFoundError("No module named 'COPASI'")

    monkeypatch.setattr(check, "_copasi_final", no_copasi)
    monkeypatch.setitem(__import__("sys").modules, "COPASI", None)
    return check


def _row(check, tmp_path) -> dict:
    out = tmp_path / "report.json"
    assert check.main(["--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["_meta"]["agree_tol"] == check.AGREE_TOL
    assert doc["_meta"]["reference"]["rtol"] == check.REF_RTOL
    (row,) = doc["results"]
    return row


def test_agreement_is_judged_converged_and_accuracy_is_reported(tmp_path, monkeypatch):
    """Loose timing tolerances (rtol 1e-3): each engine is well off the exact
    100·e^(-2) at the final time, and the engines still agree."""
    check = _check(tmp_path, monkeypatch, rtol=1e-3, atol=1e-6)
    row = _row(check, tmp_path)
    assert row["reference"]["engine"] == "roadrunner"
    assert row["bngsim_vs_reference_converged_mre"] < 1e-8
    assert row["agree"] is True
    # Accuracy at the timing tolerances, against the reference: reported.
    timed = row["bngsim_error_at_timing_tolerances"]
    assert 1e-8 < timed < 1e-1
    assert row["worst_converged_mre"] <= check.AGREE_TOL


def test_an_engine_off_the_trajectory_disagrees(tmp_path, monkeypatch):
    """An engine whose converged final state is 1% off the reference computes
    a different trajectory, whatever its timing-tolerance numbers."""
    check = _check(tmp_path, monkeypatch)
    real = check._bngsim_final

    def off(xml, span, rtol, atol):
        names, x = real(xml, span, rtol, atol)
        return names, np.asarray(x) * 1.01

    monkeypatch.setattr(check, "_bngsim_final", off)
    row = _row(check, tmp_path)
    # relative to the larger of the two: 0.01/1.01
    assert row["bngsim_vs_reference_converged_mre"] == pytest.approx(0.01 / 1.01, rel=1e-6)
    assert row["agree"] is False


def test_bngsim_is_the_reference_where_roadrunner_fails(tmp_path, monkeypatch):
    """As on Smith2013, where libRoadRunner does not complete."""
    check = _check(tmp_path, monkeypatch)

    def fails(*_):
        raise RuntimeError("CVODE Error: CV_CONV_FAILURE")

    monkeypatch.setattr(check, "_roadrunner_final", fails)
    row = _row(check, tmp_path)
    assert row["reference"]["engine"] == "bngsim"
    assert "CV_CONV_FAILURE" in row["roadrunner_error"]
    assert "roadrunner_error_at_timing_tolerances" not in row
