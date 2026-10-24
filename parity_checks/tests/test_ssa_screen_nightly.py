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
import ssa_attribution as SA
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


def _tail_args(cap=120.0):
    return argparse.Namespace(
        n=8, seed_base=1, min_mean_count=1.0, self_z_tol=4.0, revalidate_timeout=cap
    )


def test_the_tail_is_capped_and_an_overrun_is_unverified(monkeypatch):
    seen = {}

    def fake_schedule(specs, target, *, workers, timeout_of, on_done=None):
        seen["workers"] = workers
        seen["caps"] = [timeout_of(s) for s in specs]
        return [
            {"key": "slow", "status": "timeout"},
            {"key": "fast", "status": "rr_time_event", "n_compared": 3, "bn_tracks_own_ode": True},
        ]

    monkeypatch.setattr(S.rc, "schedule", fake_schedule)
    cases = [_case("slow"), _case("fast")]
    S._revalidate_time_events(cases, [_spec("slow"), _spec("fast")], _tail_args())
    assert seen == {"workers": 1, "caps": [120.0, 120.0]}  # serial, each capped
    slow, fast = cases
    # An overrun validated nothing: bngsim is neither confirmed nor faulted.
    assert slow["status"] == "rr_time_event_unverified" and slow["revalidation"] == "timeout"
    assert fast["status"] == "rr_time_event" and fast["n_compared"] == 3


def test_a_child_that_dies_or_raises_is_a_bngsim_error(monkeypatch):
    """A re-validation whose child died (a segfault in bngsim) or raised kept
    the worker's ``rr_time_event``, which reads as validated and green. Both are
    bngsim errors, reported as the worker reports a bngsim raise."""

    def fake_schedule(specs, target, *, workers, timeout_of, on_done=None):
        return [
            {"key": "died", "status": "dead", "exitcode": -11},
            {"key": "raised", "error": "RuntimeError: no signal"},
        ]

    monkeypatch.setattr(S.rc, "schedule", fake_schedule)
    cases = [_case("died"), _case("raised")]
    S._revalidate_time_events(cases, [_spec("died"), _spec("raised")], _tail_args())
    died, raised = cases
    assert died["status"] == "error" and died["revalidation"] == "dead"
    assert died["error"] == "bngsim: the time-event re-validation died (exit -11)"
    assert raised["status"] == "error" and raised["revalidation"] == "error"
    assert raised["error"] == "bngsim: RuntimeError: no signal"
    assert SA.case_to_job_result(died).outcome == "EXCEPTION"
    assert SA.case_to_job_result(raised).outcome == "EXCEPTION"


def test_a_tail_that_gave_no_verdict_never_reads_as_validated(monkeypatch):
    """The baseline from the PR's own run has seven rr_time_event rows that read
    "tracks its own ODE on all 0 above-floor cells (validated)": nothing was
    compared on them. The comment says validated only where the re-validation
    ran and bngsim tracked its ODE."""
    validated = {**_case("ok"), "n_compared": 5, "bn_tracks_own_ode": True}
    assert "(validated)" in SA.case_to_job_result(validated).comment
    # A row the tail never reached (the worker's tag, untouched).
    untouched = _case("none")
    c = SA.case_to_job_result(untouched).comment
    assert "(validated)" not in c and "gave no verdict" in c
    # A row the cap stopped.
    timed_out = {**_case("slow"), "status": "rr_time_event_unverified", "revalidation": "timeout"}
    r = SA.case_to_job_result(timed_out)
    assert r.outcome == "DIFF" and r.subclass == "rr_time_event_unverified"
    assert "(validated)" not in r.comment and "(timeout)" in r.comment


def test_bngsim_timing_out_while_growing_the_horizon_is_unverified(monkeypatch):
    """Inside the verdict itself, a bngsim timeout while the horizon grew for
    signal returned ``rr_time_event`` with nothing compared: validated on 0 cells."""
    import bngsim

    def no_ensemble(*a, **kw):
        raise bngsim.SimulationTimeout("too slow")

    monkeypatch.setattr(S.rc, "bn_ssa_replicates", no_ensemble)
    out = S._time_event_verdict(
        xml="unused.xml",
        t_end=10.0,
        n_steps=10,
        n=8,
        seed_base=1,
        min_mean_count=1.0,
        self_z_tol=4.0,
        rep_timeout=2.0,
    )
    assert out["status"] == "rr_time_event_unverified" and out["n_compared"] == 0
    assert out["bn_tracks_own_ode"] is None and "timed out" in out["revalidation"]
    # The tail carries the reason onto the row.
    monkeypatch.setattr(S.rc, "schedule", lambda specs, target, **kw: [{"key": "m", **out}])
    cases = [_case("m")]
    S._revalidate_time_events(cases, [_spec("m")], _tail_args())
    assert cases[0]["status"] == "rr_time_event_unverified"
    assert "timed out" in SA.case_to_job_result(cases[0]).comment


@pytest.mark.parametrize("flag", ["--present-only", "--revalidate-timeout"])
def test_the_screen_takes_the_nightly_flags(flag):
    src = Path(S.__file__).read_text()
    assert f'"{flag}"' in src
