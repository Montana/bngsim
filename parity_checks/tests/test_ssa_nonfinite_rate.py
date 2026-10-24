"""A NaN or infinite propensity in the SSA screen: the model's, or bngsim's (issue #1211).

bngsim refuses a rate law that evaluates to NaN or infinity (issue #809), and the
screen used to stop there with a red EXCEPTION. RoadRunner's gillespie raises
nothing at such a state: it fires no further reaction and holds the state to the
end of the run. So the screen now runs it on the same seeds and reads its rates.
Where its rate of the reaction bngsim named is non-finite too, or where its
gillespie declines the model, neither engine has a trajectory and the row is a
BAD_TEST. Anything else stays a red EXCEPTION.

The model divides by a count that reaches 0: ``0 -> B`` at ``p*A/(g*A)`` while A
decays from 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import ssa_attribution as SA
import ssa_screen as SS


def _sbml(path: Path, j1_law: str, *, rate_rule: bool = False) -> str:
    libsbml = pytest.importorskip("libsbml")

    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId("zero_over_zero")
    c = m.createCompartment()
    c.setId("C")
    c.setSize(1)
    c.setConstant(True)
    for sid, ic in (("A", 1.0), ("B", 0.0)):
        sp = m.createSpecies()
        sp.setId(sid)
        sp.setCompartment("C")
        sp.setInitialAmount(ic)
        sp.setHasOnlySubstanceUnits(True)
        sp.setBoundaryCondition(False)
        sp.setConstant(False)
    for pid, v in (("kd", 5.0), ("p", 1.0), ("g", 1.0)):
        par = m.createParameter()
        par.setId(pid)
        par.setValue(v)
        par.setConstant(True)
    if rate_rule:
        # RoadRunner's gillespie refuses any model with a rate rule.
        par = m.createParameter()
        par.setId("P")
        par.setValue(1.0)
        par.setConstant(False)
        rule = m.createRateRule()
        rule.setVariable("P")
        rule.setMath(libsbml.parseL3Formula("0.1"))
    for rid, reactant, product, law in (("J0", "A", None, "kd*A"), ("J1", None, "B", j1_law)):
        r = m.createReaction()
        r.setId(rid)
        r.setReversible(False)
        for sid, make in ((reactant, r.createReactant), (product, r.createProduct)):
            if sid:
                ref = make()
                ref.setSpecies(sid)
                ref.setStoichiometry(1)
                ref.setConstant(True)
        r.createKineticLaw().setMath(libsbml.parseL3Formula(law))
    path.write_text(libsbml.writeSBMLToString(doc))
    return str(path)


def _spec(xml: str, **extra) -> dict:
    return {
        "name": "zero_over_zero",
        "source": "synthetic",
        "sbml_path": xml,
        "exists": True,
        "t_end": 20.0,
        "n_steps": 10,
        "effort": "low",
        **extra,
    }


class _Queue:
    def __init__(self):
        self.items: list = []

    def put(self, item):
        self.items.append(item)


def _run_worker(spec: dict, n: int = 30, q: _Queue | None = None) -> dict:
    """The row the worker finished with: the last it put, as the parent reads it."""
    q = q if q is not None else _Queue()
    SS._worker(
        spec,
        n,
        2000,
        SS.DEFAULT_MEAN_Z_TOL,
        5,
        SS.DEFAULT_MIN_MEAN_COUNT,
        SS.DEFAULT_SELF_Z_TOL,
        q,
    )
    return q.items[-1]


def _no_roadrunner(*args, **kwargs):
    raise AssertionError("RoadRunner was asked for its rates")


@pytest.fixture
def stub_roadrunner(monkeypatch):
    """No live RoadRunner in the worker: its warmup and quieting are stubbed, and
    the returned setter puts a stand-in for its rates probe."""
    monkeypatch.setattr(SS.rc, "measure_ssa_warmup", lambda **k: {})
    monkeypatch.setattr(SS.rc, "set_rr_quiet", lambda: None)

    def set_probe(probe):
        monkeypatch.setattr(SS.rc, "rr_ssa_nonfinite_rate", probe)

    return set_probe


# ── RoadRunner's side


def test_roadrunner_reaches_the_rate_bngsim_refuses(tmp_path):
    pytest.importorskip("roadrunner")
    xml = _sbml(tmp_path / "m.xml", "p*A/(g*A)")
    found = SS.rc.rr_ssa_nonfinite_rate(xml, 0.0, 20.0, 11, 5, 2000)
    assert found == {
        "replicates": 1,
        "seed": 2000,
        "boundary": [],
        "nonfinite": [{"reaction": "J1", "rate": "nan", "net": {"B": 1.0}}],
    }


def test_roadrunner_keeps_a_defined_rate_finite(tmp_path):
    pytest.importorskip("roadrunner")
    xml = _sbml(tmp_path / "m.xml", "p*A/(g + A)")
    found = SS.rc.rr_ssa_nonfinite_rate(xml, 0.0, 20.0, 11, 5, 2000)
    assert found == {"replicates": 5, "seed": None, "boundary": [], "nonfinite": []}


# ── The reaction bngsim's refusal names


@pytest.mark.parametrize(
    ("label", "net"),
    [
        ("R2 (0 -> B)", {"B": 1.0}),
        ("R3 (A -> 0)", {"A": -1.0}),
        ("R43 (A + S -> A)", {"S": -1.0}),  # a catalyst cancels
        ("R7 (A + A -> B)", {"A": -2.0, "B": 1.0}),
        ("R1 (A -> A)", {}),
        ("R5 (? -> B)", None),  # a species the label could not name
    ],
)
def test_the_refused_reaction_is_read_from_the_message(label, net):
    message = f"SSA: the propensity of reaction {label} is nan at t = 0.5, with A = 0. A rate"
    assert SS._refused_net(message) == net


def test_a_refusal_of_the_total_names_no_reaction():
    message = (
        "SSA: the total propensity is inf (every reaction's is finite; their sum "
        "overflowed) at t = 1. " + SS.NONFINITE_PROPENSITY
    )
    assert SS._refused_net(message) is None


# ── The screen's verdict


def test_a_rate_law_undefined_where_the_ssa_goes_is_the_models(tmp_path):
    pytest.importorskip("roadrunner")
    row = _run_worker(_spec(_sbml(tmp_path / "m.xml", "p*A/(g*A)")))
    assert row["status"] == "nonfinite_rate"
    # bngsim's refusal stays on the row, which the report carries as the exception.
    assert row["error"].startswith("bngsim: SimulationError")
    assert "the propensity of reaction R2 (0 -> B) is nan at t = " in row["error"]
    assert row["nonfinite_rate"]["match"]["reaction"] == "J1"
    result = SA.case_to_job_result(row)
    assert result.outcome == "BAD_TEST"
    assert SA.is_expected(result.outcome, result.subclass)
    assert "reaches a nan rate of the same reaction, J1 (seed 2000)" in result.comment
    assert result.exception == row["error"]


def test_a_model_roadrunners_gillespie_declines_is_a_bad_test_too(tmp_path):
    pytest.importorskip("roadrunner")
    row = _run_worker(_spec(_sbml(tmp_path / "m.xml", "P*A/(g*A)", rate_rule=True)))
    assert row["status"] == "nonfinite_rate"
    assert "rate rules" in row["nonfinite_rate"]["roadrunner_error"]
    result = SA.case_to_job_result(row)
    assert result.outcome == "BAD_TEST"
    assert "RoadRunner's gillespie cannot run the model" in result.comment


def test_the_red_row_is_queued_before_roadrunner_runs(tmp_path, stub_roadrunner):
    """RoadRunner's run can outlast the parent's per-model cap. A killed worker is
    recovered from the last row it put, so the red row has to be there first, or
    the parent writes too_slow, a TIMEOUT this suite counts as expected."""
    q = _Queue()
    queued_when_asked: list = []

    def probe(*args):
        queued_when_asked.extend(dict(item) for item in q.items)
        net = {"B": 1.0}
        return {
            "replicates": 1,
            "seed": 2000,
            "boundary": [],
            "nonfinite": [{"reaction": "J1", "rate": "nan", "net": net}],
        }

    stub_roadrunner(probe)
    row = _run_worker(_spec(_sbml(tmp_path / "m.xml", "p*A/(g*A)")), q=q)
    assert [item["status"] for item in queued_when_asked] == ["error"]
    assert "is nan at t = " in queued_when_asked[0]["error"]
    assert SA.case_to_job_result(queued_when_asked[0]).outcome == "EXCEPTION"
    assert row["status"] == "nonfinite_rate" and len(q.items) == 2


def test_a_nonfinite_rate_of_another_reaction_stays_a_red_exception(tmp_path, stub_roadrunner):
    other = {"reaction": "J0", "rate": "nan", "net": {"A": -1.0}}
    stub_roadrunner(
        lambda *a: {"replicates": 3, "seed": 2002, "boundary": [], "nonfinite": [other]}
    )
    row = _run_worker(_spec(_sbml(tmp_path / "m.xml", "p*A/(g*A)")))
    assert row["status"] == "error" and "match" not in row["nonfinite_rate"]
    result = SA.case_to_job_result(row)
    assert result.outcome == "EXCEPTION"
    assert "non-finite rate of J0 (seed 2002), not of the reaction bngsim refused" in (
        result.comment
    )


def test_a_boundary_species_is_left_out_of_the_match(tmp_path, stub_roadrunner):
    """RoadRunner's stoichiometry has no row for a boundary species, so a species
    bngsim's label names that RoadRunner holds fixed does not count against it."""
    hit = {"reaction": "J1", "rate": "nan", "net": {}}
    stub_roadrunner(
        lambda *a: {"replicates": 1, "seed": 2000, "boundary": ["B"], "nonfinite": [hit]}
    )
    row = _run_worker(_spec(_sbml(tmp_path / "m.xml", "p*A/(g*A)")))
    assert row["status"] == "nonfinite_rate" and row["nonfinite_rate"]["match"] == hit


def test_a_roadrunner_error_that_is_not_a_gillespie_refusal_stays_red(tmp_path, stub_roadrunner):
    def probe(*args):
        raise RuntimeError("could not compile the model")

    stub_roadrunner(probe)
    row = _run_worker(_spec(_sbml(tmp_path / "m.xml", "p*A/(g*A)")))
    assert row["status"] == "error"
    assert row["nonfinite_rate"] == {
        "roadrunner_error": "RuntimeError: could not compile the model"
    }
    result = SA.case_to_job_result(row)
    assert result.outcome == "EXCEPTION"
    assert "RoadRunner's gillespie raised (RuntimeError: could not compile the model)" in (
        result.comment
    )


def test_a_nan_roadrunner_does_not_reach_stays_a_red_exception(tmp_path, stub_roadrunner):
    finite = {"replicates": 30, "seed": None, "boundary": [], "nonfinite": []}
    stub_roadrunner(lambda *a: finite)
    row = _run_worker(_spec(_sbml(tmp_path / "m.xml", "p*A/(g*A)")))
    assert row["status"] == "error" and row["nonfinite_rate"] == finite
    result = SA.case_to_job_result(row)
    assert result.outcome == "EXCEPTION"
    assert not SA.is_expected(result.outcome, result.subclass)
    assert "kept every rate finite on the same 30 seeds" in result.comment


@pytest.mark.parametrize(
    "failure",
    [
        "some other bngsim failure",
        "SSA: the total propensity is inf (every reaction's is finite; their sum "
        "overflowed) at t = 1. " + SS.NONFINITE_PROPENSITY,
    ],
    ids=["other_error", "total_propensity"],
)
def test_only_a_named_nonfinite_refusal_asks_roadrunner(
    tmp_path, monkeypatch, stub_roadrunner, failure
):
    def crash(*args, **kwargs):
        raise RuntimeError(failure)

    monkeypatch.setattr(SS.rc, "bn_ssa_replicates", crash)
    stub_roadrunner(_no_roadrunner)
    row = _run_worker(_spec(_sbml(tmp_path / "m.xml", "p*A/(g*A)")))
    assert row["status"] == "error" and "nonfinite_rate" not in row
    assert SA.case_to_job_result(row).outcome == "EXCEPTION"


# ── The reference archive (issue #805)


@pytest.mark.parametrize("rate_rule", [False, True], ids=["reaches_nan", "refused"])
def test_a_replay_gives_the_recorded_verdict_without_roadrunner(tmp_path, monkeypatch, rate_rule):
    pytest.importorskip("roadrunner")
    law = "P*A/(g*A)" if rate_rule else "p*A/(g*A)"
    xml = _sbml(tmp_path / "m.xml", law, rate_rule=rate_rule)
    arch = tmp_path / "arch"
    live = _run_worker(_spec(xml, write_reference_archive=str(arch)))
    assert live["status"] == "nonfinite_rate"

    monkeypatch.setattr(SS.rc, "rr_ssa_nonfinite_rate", _no_roadrunner)
    replay = _run_worker(_spec(xml, reference_archive=str(arch)))
    assert replay["status"] == "nonfinite_rate"
    assert replay["nonfinite_rate"] == live["nonfinite_rate"]


def test_a_record_without_the_rates_leaves_the_red_exception(tmp_path, monkeypatch):
    """An entry recorded before the screen asked for the rates: the row stays
    red, as an unattributed fail does, rather than going unreplayable."""
    pytest.importorskip("roadrunner")
    import json

    xml = _sbml(tmp_path / "m.xml", "p*A/(g*A)")
    arch = tmp_path / "arch"
    _run_worker(_spec(xml, write_reference_archive=str(arch)))
    path = SS._archive_file(arch, "zero_over_zero")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(bytes(arrays["meta"]).decode())
    del meta["nonfinite_rate"]
    arrays["meta"] = SS._json_bytes(meta)
    np.savez_compressed(path, **arrays)

    monkeypatch.setattr(SS.rc, "rr_ssa_nonfinite_rate", _no_roadrunner)
    row = _run_worker(_spec(xml, reference_archive=str(arch)))
    assert row["status"] == "error" and "nonfinite_rate" not in row
    assert SA.case_to_job_result(row).outcome == "EXCEPTION"
