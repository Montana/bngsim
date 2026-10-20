"""What the SBML SSA screen needs to run as a nightly suite (issue #806).

Hermetic: no RoadRunner and no corpus. The SBML of a BioModels candidate
resolves from the rr_parity materialization the nightly restores, a candidate
with no SBML here can be left out instead of reported as a skip, and the serial
re-validation of time-event models runs under a wall cap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import ssa_screen as S


def test_a_candidate_resolves_to_the_rr_parity_materialization(monkeypatch):
    monkeypatch.setattr(S, "BIOMODELS_SBML_DIR", None)
    path = S.biomodels_sbml_path("BIOMD0000000414")
    assert path == S.HERE / "models" / "BIOMD0000000414" / "BIOMD0000000414_url.xml"


def test_an_explicit_directory_still_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "BIOMODELS_SBML_DIR", tmp_path)
    assert S.biomodels_sbml_path("BIOMD0000000414") == tmp_path / "BIOMD0000000414.xml"


def test_269_of_the_306_candidates_have_a_pinned_source():
    """The issue's count: the other 37 have no pinned provenance, so no fetch
    produces them, and the nightly runs the candidates present."""
    names = [m["name"] for m in json.loads(S.BIOMODELS_MANIFEST.read_text())["models"]]
    materialized = S._materialized_sbml()
    assert len(names) == 306
    assert sum(1 for n in names if n in materialized) == 269


def _args(**kw):
    base = dict(
        corpus="biomodels",
        effort="high",
        models="",
        max_models=0,
        t_end=S.DEFAULT_T_END,
        n_steps=S.DEFAULT_N_STEPS,
        present_only=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_present_only_leaves_out_the_candidates_with_no_sbml(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(S, "BIOMODELS_SBML_DIR", tmp_path)
    (tmp_path / "BIOMD0000000414.xml").write_text("<sbml/>")
    everything = S._build_corpus(_args())
    assert len(everything) == 306
    assert sum(1 for s in everything if s["exists"]) == 1
    present = S._build_corpus(_args(present_only=True))
    assert [s["name"] for s in present] == ["BIOMD0000000414"]
    assert "305 candidate(s) with no SBML here left out" in capsys.readouterr().out


class _Q:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_the_revalidation_worker_returns_the_verdict(monkeypatch):
    monkeypatch.setattr(
        S, "_time_event_verdict", lambda **kw: {"status": "rr_time_event", "n_compared": 7}
    )
    q = _Q()
    S._revalidate_worker({"key": "m", "call": {}}, q)
    assert q.items == [{"key": "m", "status": "rr_time_event", "n_compared": 7}]


def test_the_revalidation_worker_reports_an_error_instead_of_raising(monkeypatch):
    def boom(**kw):
        raise RuntimeError("no signal")

    monkeypatch.setattr(S, "_time_event_verdict", boom)
    q = _Q()
    S._revalidate_worker({"key": "m", "call": {}}, q)
    assert q.items[0]["key"] == "m" and "no signal" in q.items[0]["error"]


def _case(name):
    return {"name": name, "status": "rr_time_event", "bn_tracks_own_ode": None}


def _spec(name):
    return {"name": name, "sbml_path": f"{name}.xml", "t_end": 10.0, "n_steps": 10}


def test_the_tail_is_capped_and_an_overrun_keeps_the_classification(monkeypatch):
    seen = {}

    def fake_schedule(specs, target, *, workers, timeout_of, on_done=None):
        seen["workers"] = workers
        seen["caps"] = [timeout_of(s) for s in specs]
        return [
            {"key": "slow", "status": "timeout"},
            {"key": "fast", "status": "rr_time_event_unverified", "n_compared": 3},
        ]

    monkeypatch.setattr(S.rc, "schedule", fake_schedule)
    cases = [_case("slow"), _case("fast")]
    args = argparse.Namespace(
        n=8, seed_base=1, min_mean_count=1.0, self_z_tol=4.0, revalidate_timeout=120.0
    )
    S._revalidate_time_events(cases, [_spec("slow"), _spec("fast")], args)
    assert seen == {"workers": 1, "caps": [120.0, 120.0]}  # serial, each capped
    slow, fast = cases
    assert slow["status"] == "rr_time_event" and slow["revalidation"] == "timeout"
    assert fast["status"] == "rr_time_event_unverified" and fast["n_compared"] == 3


@pytest.mark.parametrize("flag", ["--present-only", "--revalidate-timeout"])
def test_the_screen_takes_the_nightly_flags(flag):
    src = Path(S.__file__).read_text()
    assert f'"{flag}"' in src
