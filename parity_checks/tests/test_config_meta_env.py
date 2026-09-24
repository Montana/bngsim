"""A report's ``_meta["config"]["env"]`` must say what the workers actually ran with.

Each ODE/sensitivity runner records the bngsim override knobs (``BNGSIM_NO_CODEGEN``
and friends) so a report states how its sweep ran. It used to record only the
knobs its own ``--config`` combo sets, so a caller that exported one -- the
nightly workflow runs the giant BioModels with ``BNGSIM_NO_CODEGEN=1`` -- got a
report claiming the knob was unset, although every spawned worker inherits the
caller's environment. The combo's own values still win, as they do in the worker.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pytest

_PC = Path(__file__).resolve().parent.parent
for _d in ("copasi_parity",):
    if str(_PC / _d) not in sys.path:
        sys.path.insert(0, str(_PC / _d))

ARGS = argparse.Namespace(
    config="auto",
    rtol=1e-9,
    atol=1e-12,
    param_cap=0,
    param_budget=20000,
    max_steps=100000,
)


@pytest.mark.parametrize("runner", ["rr_run", "amici_run", "amici_sens_run", "copasi_run"])
def test_inherited_knob_is_recorded(runner, monkeypatch):
    mod = importlib.import_module(runner)
    monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
    assert mod._bngsim_config_meta(ARGS)["env"]["BNGSIM_NO_CODEGEN"] == "1"
    monkeypatch.delenv("BNGSIM_NO_CODEGEN")
    assert mod._bngsim_config_meta(ARGS)["env"]["BNGSIM_NO_CODEGEN"] is None


def test_combo_value_overrides_the_inherited_one(monkeypatch):
    rr_run = importlib.import_module("rr_run")
    monkeypatch.setenv("BNGSIM_CODEGEN_JIT", "something-else")
    args = argparse.Namespace(**{**vars(ARGS), "config": "mir"})
    assert rr_run._bngsim_config_meta(args)["env"]["BNGSIM_CODEGEN_JIT"] == "mir"
