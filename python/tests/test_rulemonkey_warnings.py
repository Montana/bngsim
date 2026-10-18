"""Issue #1022: RuleMonkey's warning-level load diagnostics reach the user.

RuleMonkey reports what it cannot honor in ``unsupported_features()``, each with
a severity. bngsim refused the ``Error`` ones and dropped every ``Warn`` one, so
a user never saw, for instance, that a rule whose products can never be what it
writes is never fired (richardposner/RuleMonkey#105, new in 3.11.0), or that a
Michaelis-Menten law over a pattern matching several species runs slower than
BNG2's per-species expansion. Each is now a :class:`bngsim.RuleMonkeyWarning`,
from ``Simulator(method="rm")`` and from :class:`bngsim.RuleMonkeySession`.

products_never_join.xml: rule 1, ``A(b,s~0) + B(a) -> A(b,s~1).B(a)``, writes a
free A and a free B into one product pattern that no bond joins, so BNG2
generates no reaction for it. Rule 2 binds A to B. RuleMonkey 3.10.1 fired rule
1, setting s~1 on A that BNG2 never sets.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

NF = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"
NEVER_JOIN = NF / "products_never_join"
NEVER_JOIN_WARNING = (
    r"RuleMonkey: ReactionRule: Rule 'RR1' \(_R1\) writes molecules into one product"
)


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_rulemonkey(), reason="RuleMonkey not compiled in")


def _simulator(xml: Path) -> bngsim.Simulator:
    m = bngsim.Model.from_net(str(NEVER_JOIN) + ".net")
    return bngsim.Simulator(m, method="rm", xml_path=str(xml))


def _assert_points_at(record, line: int) -> None:
    """Pointed at the line that built the simulator, not at bngsim's internals."""
    assert (record[0].filename, record[0].lineno) == (__file__, line)


def test_the_simulator_warns_of_a_rule_whose_products_never_form():
    m = bngsim.Model.from_net(f"{NEVER_JOIN}.net")
    with pytest.warns(bngsim.RuleMonkeyWarning, match=NEVER_JOIN_WARNING) as record:
        line = inspect.currentframe().f_lineno + 1
        bngsim.Simulator(m, method="rm", xml_path=f"{NEVER_JOIN}.xml")
    assert len(record) == 1
    _assert_points_at(record, line)


def test_a_session_warns_of_it_too():
    with pytest.warns(bngsim.RuleMonkeyWarning, match=NEVER_JOIN_WARNING) as record:
        line = inspect.currentframe().f_lineno + 1
        session = bngsim.RuleMonkeySession(f"{NEVER_JOIN}.xml")
    session.destroy()
    assert len(record) == 1
    _assert_points_at(record, line)


def test_the_rule_whose_products_never_form_never_fires():
    T = [0.0, 1.0, 5.0, 20.0]
    rm, ssa = [], []
    for seed in range(1, 41):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bngsim.RuleMonkeyWarning)
            r = _simulator(Path(f"{NEVER_JOIN}.xml")).run(sample_times=T, seed=seed)
        assert np.all(np.asarray(r.observables["A1"]) == 0)
        rm.append(r.observables["AB"])
        m = bngsim.Model.from_net(f"{NEVER_JOIN}.net")
        ssa.append(
            bngsim.Simulator(m, method="ssa").run(sample_times=T, seed=seed).observables["AB"]
        )
    rm, ssa = np.asarray(rm, float)[:, 1:], np.asarray(ssa, float)[:, 1:]
    se = np.sqrt(rm.var(axis=0, ddof=1) / len(rm) + ssa.var(axis=0, ddof=1) / len(ssa))
    assert np.all(np.abs(rm.mean(axis=0) - ssa.mean(axis=0)) < 4.5 * se)


def test_every_warning_is_one_python_warning():
    # TotalRate on two rules: RuleMonkey reports each.
    with pytest.warns(bngsim.RuleMonkeyWarning, match="TotalRate keyword") as record:
        bngsim.RuleMonkeySession(str(NF / "symmetry_factor_total_rate.xml")).destroy()
    assert [str(w.message).split(")")[0] for w in record] == [
        "RuleMonkey: RateLaw@totalrate: Rule 'RR1' (Tsym",
        "RuleMonkey: RateLaw@totalrate: Rule 'RR2' (Tasym",
    ]


def test_a_model_rulemonkey_reads_in_full_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.RuleMonkeyWarning)
        bngsim.RuleMonkeySession(str(NF / "first_order_switch.xml")).destroy()
        m = bngsim.Model.from_net(str(NF.parent / "simple_decay.net"))
        bngsim.Simulator(m, method="rm", xml_path=str(NF / "first_order_switch.xml"))
