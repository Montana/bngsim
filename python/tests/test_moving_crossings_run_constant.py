"""GH #824 (part 2): a condition over run-constants is not a moving crossing.

A comparison written over primary parameters alone picks its branch before the
first step and holds it for the whole run, so it has no crossing time.
`model_moving_crossings` excluded only literal comparisons and fixed clock
thresholds, so whenever something else declined the analytic sensitivity RHS
the warning named `a<b` as a condition "whose crossing time(s) move" and denied
that the difference-quotient fallback was correct. It now excludes what
`condition_cannot_cross` already recognizes as run-constant.
"""

from __future__ import annotations

import logging

import bngsim
import pytest
from bngsim import _switch_sensitivity as sw

_NET = """begin parameters
    1 a 0.5
    2 b 2
    3 k1 0.5
    4 k2 0.1
end parameters
begin species
    1 A() 10
end species
begin functions
    1 f() if({cond}, k1, k2)*floor(Atot)
{extra}end functions
begin reactions
    1 1 0 f #_R1
{extra_rxn}end reactions
begin groups
    1 Atot 1
end groups
"""


def _model(tmp_path, cond: str, with_state_switch: bool = False) -> bngsim.Model:
    extra = "    2 g() if(Atot<5, k1, k2)\n" if with_state_switch else ""
    extra_rxn = "    2 1 0 g #_R2\n" if with_state_switch else ""
    path = tmp_path / "m.net"
    path.write_text(_NET.format(cond=cond, extra=extra, extra_rxn=extra_rxn), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize("cond", ["a<b", "(a==b)<1", "a*2>=b"])
def test_a_run_constant_condition_is_not_listed(tmp_path, cond):
    m = _model(tmp_path, cond, with_state_switch=True)
    assert sw.model_moving_crossings(m._core) == ("Atot<5",)


@pytest.mark.parametrize("cond", ["a<b", "(a==b)<1"])
def test_the_decline_warning_does_not_claim_a_moving_crossing(tmp_path, cond, caplog):
    # floor() declines the analytic RHS; the run-constant condition beside it
    # must not turn that into a claim that the fallback drops a jump.
    m = _model(tmp_path, cond)
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Simulator(m, method="ode", sensitivity_params=["k1"]).run(
            t_span=(0.0, 1.0), n_points=2
        )
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "could not be differentiated" in text
    assert "crossing time(s) move" not in text
    assert "correct, but slower" in text
