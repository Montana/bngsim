"""The path-agreement runner (issue #1257, parity_checks/differential).

Its checks are tested on shape models in python/tests/test_differential_shapes.py,
each also against the defect it is there to see. These are the runner's own: how
a model's verdicts become report rows, which models the manifests give it, and
that a machine without a comma-decimal locale stops the run instead of reporting
every model as failing under de_DE.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_DIFF = Path(__file__).resolve().parent.parent / "differential"


def _load():
    sys.path.insert(0, str(_DIFF))
    spec = importlib.util.spec_from_file_location(
        "differential_run", _DIFF / "differential_run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load()
D = R.D
CHECKS = list(D.CHECKS)


def _spec(checks=CHECKS):
    job = D.Job("m", Path("m.net"))
    return {"key": "m", "job": job.as_dict(), "checks": checks, "timeout": 300.0}


def _rows(results):
    return {r.method: r.outcome for r in results}


def test_each_check_is_a_row_with_its_outcome():
    statuses = ["pass", "diff", "unsupported", "skip", "exception", "no_base"]
    statuses += ["pass"] * (len(CHECKS) - len(statuses))
    r = {
        "rows": {
            c: {"status": s, "value": 0.5, "note": f"{s} note", "extra": {"x": 1}}
            for c, s in zip(CHECKS, statuses, strict=True)
        },
        "wall_sec": 1.0,
    }
    rows = _rows(R.to_results(_spec(), r, {}))
    assert rows == {
        c: {"pass": "PASS", "diff": "DIFF", "unsupported": "UNSUPPORTED", "skip": "SKIP"}.get(
            s, "EXCEPTION" if s == "exception" else "SKIP"
        )
        for c, s in zip(CHECKS, statuses, strict=True)
    }


def test_a_finding_carries_its_note_and_a_pass_its_comment():
    r = {
        "rows": {
            "klu_vs_dense": {"status": "diff", "value": 3e-4, "note": "3.19e-04 off", "extra": {}},
            "locale": {"status": "pass", "value": 0.0, "note": "0.00e+00", "extra": {}},
        }
    }
    out = {x.method: x for x in R.to_results(_spec(["klu_vs_dense", "locale"]), r, {})}
    assert out["klu_vs_dense"].exception == "3.19e-04 off" and out["klu_vs_dense"].value == 3e-4
    assert out["locale"].exception == "" and out["locale"].comment == "0.00e+00"


@pytest.mark.parametrize(
    ("status", "outcome"),
    [("timeout", "TIMEOUT"), ("dead", "EXCEPTION"), ("no_base", "SKIP"), ("over", "SKIP")],
)
def test_a_model_without_rows_gives_every_check_its_outcome(status, outcome):
    """Timed out, died, did not load, or over the species cap: one row per
    check all the same, so a model that drops out is a case that changed, not
    one that silently disappeared from the night."""
    rows = _rows(R.to_results(_spec(), {"status": status, "why": "why"}, {}))
    assert rows == dict.fromkeys(CHECKS, outcome)


def test_a_value_that_is_not_finite_is_written_as_none():
    r = {"rows": {"locale": {"status": "diff", "value": float("inf"), "note": "", "extra": {}}}}
    assert R.to_results(_spec(["locale"]), r, {})[0].value is None


def test_the_fd_sens_manifest_gives_every_model_on_disk():
    jobs = R.fd_sens_jobs(R.FD_SENS_JOBS)
    assert len(jobs) >= 45
    assert all(j.path.is_file() for j in jobs), [j.path for j in jobs if not j.path.is_file()]


def test_the_biomodels_jobs_are_those_on_disk_and_under_the_size_cap(tmp_path):
    (tmp_path / "small.xml").write_text("x" * 10)
    (tmp_path / "big.xml").write_text("x" * 1000)
    manifest = tmp_path / "ode_jobs.json"
    params = {"t_start": 0.0, "t_end": 5.0, "n_points": 1001, "rtol": 1e-6, "atol": 1e-9}
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {"model_id": m, "model": f"{m}.xml", "params": params}
                    for m in ("small", "big", "absent")
                ]
            }
        )
    )
    jobs = R.rr_jobs(manifest, max_bytes=100, max_points=101)
    assert [j.model_id for j in jobs] == ["small"]
    assert jobs[0].n_points == 101 and jobs[0].t_end == 5.0 and jobs[0].rtol == 1e-6


def test_without_a_comma_locale_the_run_stops(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(D, "comma_locale", lambda: None)
    out = tmp_path / "report.json"
    assert R.main(["--rr-jobs", "", "--out", str(out)]) == 2
    assert not out.exists()
    assert "locale-gen de_DE.UTF-8" in capsys.readouterr().err


def test_an_unknown_check_is_refused():
    with pytest.raises(SystemExit):
        R.main(["--checks", "sens_runs,no_such_check", "--no-locale"])
