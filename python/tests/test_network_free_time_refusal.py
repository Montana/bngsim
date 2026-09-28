"""GH #862: a rate law that reads time() is refused on both network-free backends.

RuleMonkey treats a total propensity of 0 as absorbing, so A(s~0) -> A(s~1) at
f() = if(time()>=6, k, 0) never fired and A0 stayed at 20000 on every row, with
no warning (richardposner/RuleMonkey#86). NFsim cannot prepare f and failed at
setup with "Error preparing function f in class GlobalFunction". Both are now
refused before the backend is built, naming the function, and pointing at
ode/ssa, which read time() correctly.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import time_reading_functions

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
XML = DATA / "nfsim" / "time_switch.xml"


def _available(method: str) -> bool:
    if method == "nf":
        return bool(getattr(bngsim, "HAS_NFSIM", False))
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


# Literal reasons, the ones the other network-free tests give: the skip audit
# (test_skip_audit.py) reads each reason from the source and cannot read an
# f-string's.
METHODS = [
    pytest.param(
        "nf", marks=pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
    ),
    pytest.param(
        "rm",
        marks=pytest.mark.skipif(not _available("rm"), reason="RuleMonkey not compiled in"),
    ),
]


@pytest.mark.parametrize("method", METHODS)
def test_the_simulator_refuses_it(method):
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    with pytest.raises(ModelError, match=r"functions read time\(\): 'f'"):
        bngsim.Simulator(m, method=method, xml_path=str(XML))


@pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
def test_an_nfsim_session_refuses_it():
    with pytest.raises(ModelError, match=r"NFsim .* read time\(\): 'f'"):
        bngsim.NfsimSession(str(XML))


@pytest.mark.skipif(not _available("rm"), reason="RuleMonkey not compiled in")
def test_a_rulemonkey_session_refuses_it():
    with pytest.raises(ModelError, match=r"RuleMonkey .* read time\(\): 'f'"):
        bngsim.RuleMonkeySession(str(XML))


def test_the_scan_finds_the_function():
    assert time_reading_functions(XML) == ["f"]


def test_a_time_indexed_table_function_is_not_refused():
    # Its clock is the ctrName attribute and its Expression reads a placeholder,
    # and NFsim runs it.
    tfun = DATA / "nfsim" / "tfun_new_format" / "valid_inline_step_time.xml"
    assert time_reading_functions(tfun) == []


def test_a_model_with_no_time_reads_is_not_refused():
    assert time_reading_functions(DATA / "nfsim" / "first_order_switch.xml") == []
