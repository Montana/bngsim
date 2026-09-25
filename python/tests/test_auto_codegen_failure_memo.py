"""GH #826: a failed automatic codegen build is not retried on every Simulator().

The automatic path (at or above BNGSIM_CODEGEN_THRESHOLD species) fell back to
the interpreted RHS on a failed build and forgot it, so every later construction
on the same model, or on a freshly loaded copy, regenerated the source and waited
out the compile timeout again. The failure is now remembered for the process,
keyed on the structural codegen key and the settings that decide the outcome.
"""

from __future__ import annotations

import logging

import bngsim
import bngsim._codegen as cg
import bngsim._simulator as simmod
import pytest
from bngsim._bngsim_core import ModelBuilder

N = 300  # above the default 256-species threshold


def _chain() -> bngsim.Model:
    b = ModelBuilder()
    for i in range(N):
        b.add_parameter(f"k{i}", 0.1 + 0.001 * i, "", False)
        b.add_species(f"S{i}", 1.0, False, 1.0)
    for i in range(N - 1):
        b.add_reaction([i], [i + 1], "elementary", f"k{i}", 1.0, True)
    return bngsim.Model(_core=b.build())


@pytest.fixture
def failing_build(monkeypatch):
    """Every model-path build fails the way a compile timeout does, and counts."""
    calls = []

    def fail(model):
        calls.append(model)
        cg._record_codegen_error(RuntimeError("Codegen compilation timed out after 600 s"))
        return None

    monkeypatch.setattr(cg, "prepare_model_codegen", fail)
    monkeypatch.setattr(cg, "prepare_model_codegen_source", fail)
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    monkeypatch.delenv("BNGSIM_CODEGEN_THRESHOLD", raising=False)
    monkeypatch.setenv("BNGSIM_CODEGEN_TIMEOUT", "600")
    monkeypatch.setattr(simmod, "_AUTO_CODEGEN_FAILURES", {})
    monkeypatch.setattr(simmod, "_AUTO_CODEGEN_SKIP_LOGGED", set())
    return calls


def test_a_failed_build_is_attempted_once_per_model(failing_build, caplog):
    m = _chain()
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        for _ in range(3):
            bngsim.Simulator(m, method="ode")
        # A freshly loaded copy has the same structural key.
        bngsim.Simulator(_chain(), method="ode")
    assert len(failing_build) == 1
    skips = [r for r in caplog.records if "Skipping automatic codegen" in r.getMessage()]
    assert len(skips) == 1  # said once, not per construction
    assert "timed out" in skips[0].getMessage()


def test_a_larger_budget_gets_a_fresh_attempt(failing_build, monkeypatch):
    m = _chain()
    bngsim.Simulator(m, method="ode")
    monkeypatch.setenv("BNGSIM_CODEGEN_TIMEOUT", "3600")
    bngsim.Simulator(m, method="ode")
    assert len(failing_build) == 2


def test_an_explicit_codegen_request_still_builds(failing_build):
    m = _chain()
    bngsim.Simulator(m, method="ode")
    with pytest.raises(RuntimeError):
        bngsim.Simulator(m, method="ode", codegen=True)
    assert len(failing_build) == 2


def test_a_decline_is_not_remembered(monkeypatch):
    calls = []

    def decline(model):
        calls.append(model)
        cg._record_codegen_error(None)
        return None

    monkeypatch.setattr(cg, "prepare_model_codegen", decline)
    monkeypatch.setattr(cg, "prepare_model_codegen_source", decline)
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    monkeypatch.delenv("BNGSIM_CODEGEN_THRESHOLD", raising=False)
    monkeypatch.setattr(simmod, "_AUTO_CODEGEN_FAILURES", {})
    m = _chain()
    bngsim.Simulator(m, method="ode")
    bngsim.Simulator(m, method="ode")
    assert len(calls) == 2
    assert simmod._AUTO_CODEGEN_FAILURES == {}
