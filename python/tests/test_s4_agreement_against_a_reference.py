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
    assert row["bngsim_vs_reference_converged"]["mre"] < 1e-8
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
    assert row["bngsim_vs_reference_converged"]["mre"] == pytest.approx(0.01 / 1.01, rel=1e-6)
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


# ── Negative controls (the review of #1250's fix) ───────────────────────────
#
# A broken engine must never be reported as agreeing. Each case below gave
# agree=True on the first version of the check: a NaN species was dropped
# from the comparison (its scale was NaN, and NaN > floor is false), two
# species' values swapped passed the order-independent comparison, a species
# under the 1e-6 floor was never compared at any error, and an engine
# reporting fewer species was compared over the shorter list.

# A -> B -> C from A = 100, and D = 1e-8 decaying: values from 1e-8 to 100.
_MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _species(sid, x):
    return (
        f'<species id="{sid}" compartment="c" initialConcentration="{x}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _reaction(rid, k, reactant, product=None):
    out = (
        f'<listOfProducts><speciesReference species="{product}" stoichiometry="1" '
        'constant="true"/></listOfProducts>'
        if product
        else ""
    )
    return (
        f'<reaction id="{rid}" reversible="false"><listOfReactants><speciesReference '
        f'species="{reactant}" stoichiometry="1" constant="true"/></listOfReactants>{out}'
        f"<kineticLaw><math {_MATH}><apply><times/><ci>{k}</ci><ci>{reactant}</ci>"
        "</apply></math></kineticLaw></reaction>"
    )


CHAIN_SBML = (
    '<?xml version="1.0" encoding="UTF-8"?><sbml xmlns="http://www.sbml.org/sbml/level3/'
    'version1/core" level="3" version="1"><model id="chain"><listOfCompartments>'
    '<compartment id="c" size="1" constant="true"/></listOfCompartments><listOfSpecies>'
    + "".join(_species(s, x) for s, x in (("A", 100), ("B", 0), ("C", 0), ("D", 1e-8)))
    + "</listOfSpecies><listOfParameters>"
    + "".join(
        f'<parameter id="{k}" value="{v}" constant="true"/>'
        for k, v in (("k1", 0.3), ("k2", 0.1), ("kd", 0.05))
    )
    + "</listOfParameters><listOfReactions>"
    + _reaction("r1", "k1", "A", "B")
    + _reaction("r2", "k2", "B", "C")
    + _reaction("r3", "kd", "D")
    + "</listOfReactions></model></sbml>"
)


def _chain_check(tmp_path, monkeypatch, broken=None):
    """The check on the chain, with bngsim's converged run passed through
    *broken* (names, trajectory) -> (names, trajectory)."""
    check = _check(tmp_path, monkeypatch)
    (tmp_path / "models" / "DECAY" / "decay.xml").write_text(CHAIN_SBML)
    if broken is not None:
        real = check._bngsim_final

        def bngsim(xml, span, rtol, atol):
            names, x = real(xml, span, rtol, atol)
            return broken(list(names), np.array(x, dtype=float)) if rtol < 1e-9 else (names, x)

        monkeypatch.setattr(check, "_bngsim_final", bngsim)
    return _row(check, tmp_path)


def _col(names, name):
    return names.index(name)


def test_the_chain_agrees_as_it_stands(tmp_path, monkeypatch):
    """The control: unbroken, the chain agrees, D included."""
    row = _chain_check(tmp_path, monkeypatch)
    assert row["agree"] is True
    assert row["bngsim_vs_reference_converged"]["compared"] == 4


@pytest.mark.parametrize("where", ["final", "mid"])
def test_a_nan_disagrees(tmp_path, monkeypatch, where):
    def nan(names, x):
        x[-1 if where == "final" else len(x) // 2, _col(names, "B")] = np.nan
        return names, x

    row = _chain_check(tmp_path, monkeypatch, nan)
    assert row["agree"] is False


def test_two_species_swapped_disagree(tmp_path, monkeypatch):
    """The multiset of values is unchanged; the species are not."""

    def swap(names, x):
        a, b = _col(names, "A"), _col(names, "C")
        x[:, [a, b]] = x[:, [b, a]]
        return names, x

    row = _chain_check(tmp_path, monkeypatch, swap)
    assert row["agree"] is False


def test_a_small_species_off_by_ten_percent_disagrees(tmp_path, monkeypatch):
    """D near 1e-8, well above what the reference resolves (1e-12)."""

    def small(names, x):
        x[:, _col(names, "D")] *= 1.1
        return names, x

    row = _chain_check(tmp_path, monkeypatch, small)
    assert row["agree"] is False


def test_a_species_off_mid_run_disagrees(tmp_path, monkeypatch):
    """Right at the end, wrong on the way: B 5% off at the middle row."""

    def mid(names, x):
        x[len(x) // 2, _col(names, "B")] *= 1.05
        return names, x

    row = _chain_check(tmp_path, monkeypatch, mid)
    assert row["agree"] is False


@pytest.mark.parametrize("change", ["missing", "renamed"])
def test_species_that_cannot_be_matched_are_not_agreement(tmp_path, monkeypatch, change):
    """An engine reporting a species fewer, or one under another id, is not
    compared over what is left: agreement is not established."""

    def species(names, x):
        if change == "missing":
            return names[:-1], x[:, :-1]
        return [*names[:-1], "Q"], x

    row = _chain_check(tmp_path, monkeypatch, species)
    assert row["agree"] is not True


def test_nothing_compared_is_not_agreement(tmp_path, monkeypatch):
    """Where every value is under what the reference resolves, nothing is
    compared, and that is no pass."""
    check = _check(tmp_path, monkeypatch)
    (tmp_path / "models" / "DECAY" / "decay.xml").write_text(
        DECAY_SBML.replace('initialAmount="100"', 'initialAmount="1e-14"')
    )
    row = _row(check, tmp_path)
    assert row["agree"] is None
