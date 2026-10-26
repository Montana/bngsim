"""The RoadRunner reference archive (issue #805).

``rr_run.py --write-reference-archive`` records each job's RoadRunner side;
``--reference-archive`` compares bngsim against that record without running
RoadRunner. An entry that no longer matches its job is stale and never
compared. The end-to-end test records the committed smoke subset, then replays
it with RoadRunner blocked from importing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rr_run as R

_RR = Path(R.__file__).resolve().parent
SMOKE_JOBS = _RR / "smoke" / "ode_jobs_smoke.json"


def _spec(tmp_path: Path, **params) -> dict:
    xml = tmp_path / "m.xml"
    if not xml.exists():
        xml.write_text("<sbml>the model</sbml>")
    p = {"t_start": 0.0, "t_end": 10.0, "n_points": 11, "rtol": 1e-6, "atol": 1e-9}
    p.update(params)
    return {
        "key": "BIOMD1:ode",
        "model_id": "BIOMD1",
        "method": "ode",
        "xml": str(xml),
        "params": p,
    }


def _rr():
    t = np.linspace(0.0, 10.0, 11)
    return t, np.column_stack([np.exp(-t), 1 - np.exp(-t)]), ["A", "B"]


def test_a_recorded_reference_comes_back_as_it_was(tmp_path):
    spec = _spec(tmp_path)
    R.archive_save(tmp_path / "arch", spec, _rr(), "", {"integrate_sec": 0.01})
    rr, exc, timing, meta = R.archive_load(tmp_path / "arch", spec)
    assert exc == "" and timing == {"integrate_sec": 0.01}
    for got, want in zip(rr[:2], _rr()[:2], strict=False):
        np.testing.assert_array_equal(got, want)
    assert rr[2] == ["A", "B"]
    assert meta["model_sha256"] == R._file_sha256(spec["xml"])


def test_a_reference_that_raised_replays_its_exception(tmp_path):
    spec = _spec(tmp_path)
    R.archive_save(
        tmp_path / "arch", spec, None, "roadrunner: RuntimeError: missing a value", None
    )
    rr, exc, _timing, _meta = R.archive_load(tmp_path / "arch", spec)
    assert rr is None and exc == "roadrunner: RuntimeError: missing a value"
    # ...and classifies as the live refusal did.
    assert R.classify_reference_refusal(exc) == "overstrict_missing_value"


def test_no_entry_is_stale(tmp_path):
    rr, exc, _t, _m = R.archive_load(tmp_path / "arch", _spec(tmp_path))
    assert rr is None and exc.startswith("archive_stale") and "no entry" in exc


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"rtol": 1e-12}, "rtol"),
        ({"atol": 1e-16}, "atol"),
        ({"n_points": 101}, "n_points"),
        ({"t_end": 20.0}, "t_end"),
    ],
)
def test_a_changed_grid_or_tolerance_is_stale(tmp_path, change, field):
    """#1099's change of BIOMD0000000374's rtol is the case this exists for:
    a reference at the old tolerance is no reference for the new job."""
    R.archive_save(tmp_path / "arch", _spec(tmp_path), _rr(), "", None)
    rr, exc, _t, _m = R.archive_load(tmp_path / "arch", _spec(tmp_path, **change))
    assert rr is None and exc == f"archive_stale: the job's {field} changed"


def test_a_changed_model_file_is_stale(tmp_path):
    spec = _spec(tmp_path)
    R.archive_save(tmp_path / "arch", spec, _rr(), "", None)
    Path(spec["xml"]).write_text("<sbml>another model</sbml>")
    _rr_, exc, _t, _m = R.archive_load(tmp_path / "arch", spec)
    assert exc == "archive_stale: the job's model_sha256 changed"


def test_a_stale_row_has_its_own_refusal_class():
    exc = "roadrunner (archive): archive_stale: the job's rtol changed"
    assert R.classify_reference_refusal(exc) == "archive_stale"
    assert "archive_stale" not in R.SETTLED_REFUSALS  # it wants re-recording


def test_an_unreadable_entry_is_refused_rather_than_raised(tmp_path):
    spec = _spec(tmp_path)
    path = R._archive_file(tmp_path / "arch", spec["key"])
    path.parent.mkdir()
    path.write_bytes(b"not an npz")
    rr, exc, _t, _m = R.archive_load(tmp_path / "arch", spec)
    assert rr is None and exc.startswith("archive_unreadable: ")
    assert R.classify_reference_refusal(f"roadrunner (archive): {exc}") == "archive_unreadable"
    assert "archive_unreadable" not in R.SETTLED_REFUSALS


def test_keys_that_sanitize_alike_get_their_own_files(tmp_path):
    assert R._archive_file(tmp_path, "a/b:ode") != R._archive_file(tmp_path, "a_b:ode")


class _Queue(list):
    put = list.append


def _replay(tmp_path, monkeypatch, spec, *, bn):
    """Drive the worker in-process on a replay, with RoadRunner's own call made
    to fail the test and bngsim's answered by ``bn``."""
    monkeypatch.setattr(R.rc, "measure_warmup", lambda include_roadrunner=True: {})
    monkeypatch.setattr(R.rc, "rr_ode", lambda *a, **k: pytest.fail("RoadRunner ran on a replay"))
    monkeypatch.setattr(R.rc, "bn_ode", bn)
    q = _Queue()
    R._worker(
        {**spec, "metric": "max_rel_err", "tol": 1e-4, "reference_archive": tmp_path / "arch"}, q
    )
    (res,) = q
    return res


@pytest.mark.parametrize("kind", ["archive_stale", "archive_unreadable"])
def test_an_unusable_entry_is_refused_before_bngsim_runs(tmp_path, monkeypatch, kind):
    """A bngsim raise beside an unusable entry used to make the row BAD_TEST,
    "neither engine ran", with the refusal lost; the row is never compared, so
    bngsim is not run at all."""
    R.archive_save(tmp_path / "arch", _spec(tmp_path), _rr(), "", None)
    spec = _spec(tmp_path, rtol=1e-12)
    if kind == "archive_unreadable":
        spec = _spec(tmp_path)
        R._archive_file(tmp_path / "arch", spec["key"]).write_bytes(b"truncated")
    res = _replay(
        tmp_path,
        monkeypatch,
        spec,
        bn=lambda *a, **k: pytest.fail("bngsim ran for a row that is never compared"),
    )
    assert (res["status"], res["archive_refusal"]) == ("reference_failed", kind)
    assert R.classify_reference_refusal(res["exception"]) == kind


def test_a_replay_scores_the_record_and_names_the_recorded_version(tmp_path, monkeypatch):
    monkeypatch.setattr(R.versions, "roadrunner_version", lambda: "2.7.0")
    R.archive_save(tmp_path / "arch", _spec(tmp_path), _rr(), "", {"integrate_sec": 0.01})
    monkeypatch.setattr(R.versions, "roadrunner_version", lambda: "9.9.9")  # the local one
    res = _replay(tmp_path, monkeypatch, _spec(tmp_path), bn=lambda *a, **k: (*_rr(), {}))
    assert res["status"] == "pass" and res["reference_version"] == "2.7.0"
    assert res["timing"]["roadrunner"]["replayed_from_archive"] is True
    assert res["timing"]["roadrunner"]["reference_version"] == "2.7.0"


def test_an_archive_refusal_is_judged_by_no_override(tmp_path, monkeypatch):
    """An adjudication would turn archive_stale into a settled class, and an
    invalid-reference entry would call itself stale and ask to be pruned; a row
    that had no reference at all can do neither. The report also names the
    RoadRunner version the archive recorded, not a local one."""
    adjudicated, invalid = "BIOMD0000000003", "BIOMD0000000004"

    def overrides(job):
        if job.model_id == adjudicated:
            return None, None, None, ("confirm", "an independent oracle agreed", "#117")
        if job.model_id == invalid:
            return None, None, ("RoadRunner answers NaN", "#485"), None
        return None, None, None, None

    def schedule(specs, worker, **kw):
        rows = []
        for s in specs:
            row = {k: s[k] for k in ("key", "model_id", "method", "metric", "tol")}
            if s["model_id"] == "BIOMD0000000001":
                row.update(status="pass", value=0.0, reference_version="2.7.0")
            else:
                row.update(
                    status="reference_failed",
                    exception="roadrunner (archive): archive_stale: the job's rtol changed",
                    archive_refusal="archive_stale",
                )
            rows.append(row)
        return rows

    monkeypatch.setattr(R, "_job_overrides", overrides)
    monkeypatch.setattr(R.rc, "schedule", schedule)
    out = tmp_path / "report.json"
    argv = ["rr_run.py", "--jobs", str(SMOKE_JOBS), "--reference-archive", str(tmp_path / "a")]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out)])
    assert R.main() == 0
    report = json.loads(out.read_text())
    rows = {r["model_id"]: r for r in report["results"]}
    for model_id in (adjudicated, invalid):
        assert rows[model_id]["outcome"] == "REFERENCE_FAILED"
        assert rows[model_id]["reference_refusal"] == "archive_stale"
        assert "adjudicated" not in rows[model_id]["comment"]
        assert "STALE" not in rows[model_id]["comment"]
    assert rows["BIOMD0000000001"]["versions"]["roadrunner"] == "2.7.0"
    assert rows[adjudicated]["versions"]["roadrunner"] is None
    assert report["_meta"]["versions"]["roadrunner"] == "2.7.0"


def test_the_two_flags_are_exclusive(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "rr_run.py"),
            "--reference-archive",
            "a",
            "--write-reference-archive",
            "b",
        ],
        capture_output=True,
        text=True,
        cwd=_RR,
        timeout=120,
    )
    assert proc.returncode != 0 and "--write-reference-archive records one" in proc.stderr


def _run(tmp_path, *args, env=None):
    out = tmp_path / f"report{len(list(tmp_path.glob('report*')))}.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "rr_run.py"),
            "--jobs",
            str(SMOKE_JOBS),
            "--workers",
            "2",
            "--out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=_RR,
        timeout=600,
        env=env,
    )
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return {r["model_id"]: r for r in json.loads(out.read_text())["results"]}


def test_a_replay_without_roadrunner_reproduces_the_live_verdicts(tmp_path):
    pytest.importorskip("roadrunner")
    pytest.importorskip("bngsim")
    if not SMOKE_JOBS.is_file():
        pytest.skip("smoke subset not in this checkout")
    arch = tmp_path / "arch"
    live = _run(tmp_path, "--write-reference-archive", str(arch))
    assert len(list(arch.glob("*.npz"))) == len(live)
    # RoadRunner cannot be imported by the replay or any worker it spawns.
    blocked = tmp_path / "no_roadrunner" / "roadrunner"
    blocked.mkdir(parents=True)
    (blocked / "__init__.py").write_text("raise ImportError('blocked for this test')\n")
    env = {
        **os.environ,
        "PYTHONPATH": f"{blocked.parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    replay = _run(tmp_path, "--reference-archive", str(arch), env=env)
    assert set(replay) == set(live)
    for model_id, row in live.items():
        got = replay[model_id]
        assert (got["outcome"], got["value"], got["comment"]) == (
            row["outcome"],
            row["value"],
            row["comment"],
        ), model_id
        assert got["timing"]["roadrunner"]["replayed_from_archive"] is True
