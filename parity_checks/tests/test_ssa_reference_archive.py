"""The SSA screen's RoadRunner reference archive (issue #805).

``ssa_screen.py --write-reference-archive`` records every RoadRunner result the
screen used for a model; ``--reference-archive`` replays them without
RoadRunner. An entry that no longer matches its model is stale, and a replay
that needs a result the recorded run never produced is unreplayable; neither
is compared. The end-to-end test records a few BioModels and replays them with
RoadRunner blocked from importing; it skips where the corpus is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import ssa_attribution as SA
import ssa_screen as SS

_RR = Path(SS.__file__).resolve().parent


def _spec(tmp_path: Path, **change) -> dict:
    xml = tmp_path / "m.xml"
    if not xml.exists():
        xml.write_text("<sbml>the model</sbml>")
    return {"name": "BIOMD1", "sbml_path": str(xml), "t_end": 100.0, "n_steps": 10, **change}


def _fake_rr_ssa(calls):
    def rr_ssa(xml, t_start, t_end, n_points, n, seed_base):
        calls.append(t_end)
        if t_end > 1000.0:
            raise ValueError("gillespie cannot do rate rules")
        t = np.linspace(t_start, t_end, n_points)
        return t, np.full((n, n_points, 2), t_end), ["A", "B"], {"ensemble_sec": 0.5}

    return rr_ssa


def test_every_result_the_live_run_used_comes_back(tmp_path, monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(SS.rc, "rr_ssa", _fake_rr_ssa(calls))
    monkeypatch.setattr(SS, "_rr_ode", lambda xml, t_end, n_steps: (np.ones((11, 2)), ["A", "B"]))
    spec, arch = _spec(tmp_path), tmp_path / "arch"
    live = SS.Reference(spec, 30, 7, record_to=arch)
    for h in (100.0, 1000.0):  # the full horizon, then one extended-horizon rung
        live.ssa("m.xml", 0.0, h, 11, 30, 7)
    with pytest.raises(ValueError):
        live.ssa("m.xml", 0.0, 10000.0, 11, 30, 7)
    live.ode("m.xml", 100.0, 10)

    replay = SS.Reference(spec, 30, 7, replay=arch)
    assert replay.stale == ""
    t, arr, names, timing = replay.ssa("m.xml", 0.0, 1000.0, 11, 30, 7)
    np.testing.assert_array_equal(arr, np.full((30, 11, 2), 1000.0))
    assert names == ["A", "B"] and timing == {"ensemble_sec": 0.5, "replayed_from_archive": True}
    assert replay.elapsed_sec(100.0) == live.elapsed_sec(100.0)
    # A recorded failure is raised again under its class name, as the live one was.
    with pytest.raises(RuntimeError) as exc:
        replay.ssa("m.xml", 0.0, 10000.0, 11, 30, 7)
    assert type(exc.value).__name__ == "ValueError" and "rate rules" in str(exc.value)
    np.testing.assert_array_equal(replay.ode("m.xml", 100.0, 10)[0], np.ones((11, 2)))
    assert calls == [100.0, 1000.0, 10000.0]  # the replay itself ran nothing


def test_a_result_the_recorded_run_never_produced_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(SS.rc, "rr_ssa", _fake_rr_ssa([]))
    spec, arch = _spec(tmp_path), tmp_path / "arch"
    SS.Reference(spec, 30, 7, record_to=arch).ssa("m.xml", 0.0, 100.0, 11, 30, 7)
    replay = SS.Reference(spec, 30, 7, replay=arch)
    with pytest.raises(SS.ArchiveMiss, match="t=0→10,"):
        replay.ssa("m.xml", 0.0, 10.0, 11, 30, 7)  # a partial-horizon rung
    with pytest.raises(SS.ArchiveMiss, match="no fail to attribute"):
        replay.ode("m.xml", 100.0, 10)
    entry = SS._not_from_archive({"max_mean_z": 0.0}, replay, SS.ArchiveMiss("x"))
    assert entry["status"] == "archive_unreplayable" and "max_mean_z" not in entry


def test_an_entry_that_no_longer_matches_its_model_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(SS.rc, "rr_ssa", _fake_rr_ssa([]))
    arch = tmp_path / "arch"
    SS.Reference(_spec(tmp_path), 30, 7, record_to=arch).ssa("m.xml", 0.0, 100.0, 11, 30, 7)
    for spec, n, seed, field in (
        (_spec(tmp_path, t_end=50.0), 30, 7, "t_end"),
        (_spec(tmp_path, n_steps=20), 30, 7, "n_steps"),
        (_spec(tmp_path), 100, 7, "n_rep"),
        (_spec(tmp_path), 30, 8, "seed_base"),
    ):
        assert SS.Reference(spec, n, seed, replay=arch).stale == f"the model's {field} changed"
    Path(_spec(tmp_path)["sbml_path"]).write_text("<sbml>another model</sbml>")
    ref = SS.Reference(_spec(tmp_path), 30, 7, replay=arch)
    assert ref.stale == "the model's model_sha256 changed"
    with pytest.raises(SS.ArchiveMiss):
        ref.ssa("m.xml", 0.0, 100.0, 11, 30, 7)
    assert SS.Reference(_spec(tmp_path, name="X"), 30, 7, replay=arch).stale == (
        "no entry for this model"
    )


@pytest.mark.parametrize("status", ["archive_stale", "archive_unreplayable"])
def test_an_archive_refusal_is_an_uncompared_reference_failure(status):
    case = {
        "name": "BIOMD1",
        "status": status,
        "max_mean_z": 0.0,
        "error": f"roadrunner (archive): {status}: the model's rtol changed",
    }
    r = SA.case_to_job_result(case)
    assert (r.outcome, r.reference_refusal, r.value) == ("REFERENCE_FAILED", status, None)
    assert SA.is_expected(r.outcome, r.subclass)
    assert "the model's rtol changed" in r.comment


def _screen(tmp_path, *args, env=None):
    out = tmp_path / f"core{len(list(tmp_path.glob('core*')))}.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "ssa_screen.py"),
            "--corpus",
            "biomodels",
            "--jobs",
            "4",
            "--out",
            str(tmp_path / "native.json"),
            "--core-out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=_RR,
        timeout=1200,
        env=env,
    )
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc, out


# A pass, a vacuous pass that walks the extended ladder, an extended-horizon
# verdict, a partial-horizon verdict and an attributed fail.
_E2E = "BIOMD0000000417,BIOMD0000000414,BIOMD0000000573,BIOMD0000000760,BIOMD0000001026"


def test_a_replay_without_roadrunner_reproduces_the_live_verdicts(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(_RR / "ssa_screen.py"),
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
    pytest.importorskip("roadrunner")
    pytest.importorskip("bngsim")
    sbml = Path(os.environ.get("BIOMODELS_SBML_DIR", "/nonexistent"))
    if not all((sbml / f"{m}.xml").is_file() for m in _E2E.split(",")):
        pytest.skip("BioModels SSA corpus not present (set BIOMODELS_SBML_DIR)")
    arch = tmp_path / "arch"
    _p, live = _screen(tmp_path, "--models", _E2E, "--write-reference-archive", str(arch))
    blocked = tmp_path / "no_roadrunner" / "roadrunner"
    blocked.mkdir(parents=True)
    (blocked / "__init__.py").write_text("raise ImportError('blocked for this test')\n")
    env = {
        **os.environ,
        "PYTHONPATH": f"{blocked.parent}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    _p, replay = _screen(tmp_path, "--models", _E2E, "--reference-archive", str(arch), env=env)

    def rows(path):
        return {r["model_id"]: r for r in json.loads(path.read_text())["results"]}

    live_rows, replay_rows = rows(live), rows(replay)
    assert set(live_rows) == set(replay_rows) == set(_E2E.split(","))
    for model_id, row in live_rows.items():
        got = replay_rows[model_id]
        assert (got["outcome"], got["subclass"], got["value"], got["comment"]) == (
            row["outcome"],
            row["subclass"],
            row["value"],
            row["comment"],
        ), model_id
        if (row.get("timing") or {}).get("roadrunner"):
            assert got["timing"]["roadrunner"]["replayed_from_archive"] is True
