"""amici_sens judges each parameter column on its own scale (issue #1261).

``sens_verdict`` on the whole (time, species, parameter) tensor hands
``_core.differ`` one array, whose significance gate counts a (species,
parameter) column only past 1e-3 of the array's peak. A parameter column's
sensitivities are in that parameter's units, so a column of dx/dk2 that is a
thousandth of dx/dk1 was never looked at: on the nightly's 82 compared
amici_sens cases, 2,702 of 4,272 parameter columns, in 56 cases, could be
zeroed, negated or multiplied by 1000 and still read PASS.

``_compare`` now judges each column with ``_amici_sens.column_verdicts``, the
rule fd_sens adopted in #1167 and now shares: a column scaled to its own peak,
judged where that peak clears its ``atol/|p|`` floor by ``JUDGE_FACTOR``, and
named on the row as not judged where it does not. On the same 82 cases that
judges 3,448 columns, every zeroed, negated or x1000 one among them reads DIFF,
and no case's outcome moves.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import _amici_sens as asens
import amici_sens_run as run
import numpy as np
import pytest

ATOL = 1e-12
T = np.linspace(0.0, 10.0, 11)
NAMES = ["A", "B"]
X = np.stack([np.exp(-0.1 * T), 1.0 - np.exp(-0.1 * T)], axis=1)


def _tensor(small_peak):
    """dx/dk1 peaks at 1e3, dx/dk2 at ``small_peak``: two parameters in their
    own units, which is what the tensor-wide gate cannot tell from noise."""
    shape = np.sin(0.3 * T + 0.2)[:, None] * np.array([1.0, -0.5])[None, :]
    return np.stack([1e3 * shape, small_peak * shape], axis=2)


def _compare(bn_sx, am_sx, param_values=(1.0, 1.0)):
    return run._compare(
        (T, X, bn_sx, NAMES),
        (T, X.copy(), am_sx, NAMES),
        param_values=list(param_values),
        atol=ATOL,
    )


@pytest.mark.parametrize("corrupt", ["zero", "negate", "x1000"])
def test_a_small_column_gone_wrong_reads_diff(corrupt):
    """dx/dk2 is 1e-7 of dx/dk1 (1e-4 even at x1000, still under the 1e-3
    gate) and 1e6 over its floor. Corrupted, it read PASS on the tensor as a
    whole; judged alone, it reads DIFF."""
    am = _tensor(1e-4)
    bn = am.copy()
    bn[..., 1] = {"zero": 0.0, "negate": -bn[..., 1], "x1000": 1000.0 * bn[..., 1]}[corrupt]
    assert asens.sens_verdict(bn, am, param_values=[1.0, 1.0], atol=ATOL)["passed"], (
        "precondition: the tensor-wide verdict cannot see it"
    )
    assert _compare(bn, am)["status"] == "diff"


def test_a_tensor_under_differs_absolute_floor_is_still_judged():
    """Every cell under differ's absolute 1e-12, as nanomolar_dimer's 6e-20
    tensor is in fd_sens, but far over its atol/|p| floor at atol 1e-30: each
    column is scaled to its own peak, so a negated one reads DIFF."""
    am = _tensor(1.0) * 1e-17
    bn = am.copy()
    bn[..., 1] = -bn[..., 1]
    res = run._compare(
        (T, X, bn, NAMES), (T, X.copy(), am, NAMES), param_values=[1.0, 1.0], atol=1e-30
    )
    assert res["status"] == "diff"


def test_the_failing_column_is_named():
    am = _tensor(0.1)
    bn = am.copy()
    bn[..., 1] = 0.0
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["columns_diff"] == ["k2"]
    assert res["n_columns_judged"] == 2


def test_agreeing_columns_pass():
    am = _tensor(0.1)
    res = _compare(am.copy(), am)
    assert res["status"] == "pass"
    assert res["columns_not_judged"] == [] and res["columns_diff"] == []


def test_a_column_under_its_floor_is_named_and_not_judged():
    """dx/dk2 peaks at 1e-10, under 20x its 1e-10 floor (100*atol/|p|): the
    solver does not resolve it, so the row names it rather than passing it.
    Zeroing it is then not the verdict's to see, by design."""
    am = _tensor(1e-10)
    bn = am.copy()
    bn[..., 1] = 0.0
    res = run._compare(
        (T, X, bn, NAMES),
        (T, X.copy(), am, NAMES),
        param_values=[1.0, 1.0],
        atol=ATOL,
        param_ids=["k1", "k2"],
    )
    assert res["status"] == "pass"
    assert res["columns_not_judged"] == ["k2"]
    assert res["n_columns_judged"] == 1


def test_a_row_with_no_column_to_judge_is_a_bad_test():
    """#328's vacuous pass, now refused rather than reported as one."""
    am = np.full((len(T), 2, 2), 1e-14)
    res = _compare(am.copy(), am)
    assert res["status"] == "bad_test"
    assert res["n_columns_judged"] == 0


def test_without_parameter_values_the_tensor_is_judged_whole():
    """No floor to judge a column against: the verdict sens_verdict gives
    without one, unchanged."""
    am = _tensor(0.1)
    res = run._compare((T, X, am.copy(), NAMES), (T, X.copy(), am, NAMES))
    assert res["status"] == "pass"
    assert res["n_columns_judged"] is None


def test_fd_sens_judges_by_the_same_rule():
    """One rule in two suites: fd_sens's factor and per-column verdict are
    _amici_sens's."""
    path = Path(__file__).resolve().parents[1] / "fd_sens" / "fd_sens_run.py"
    spec = importlib.util.spec_from_file_location("fd_sens_run", path)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)

    assert F.JUDGE_FACTOR == asens.JUDGE_FACTOR
    am = _tensor(0.1)
    bn = am.copy()
    bn[..., 1] = 0.0
    gap = np.zeros_like(am)
    ours = F.column_verdicts(bn, am, gap, 1.0, [1.0, 1.0], ATOL, [0, 1])
    shared = asens.column_verdicts(
        bn,
        am,
        [1.0, 1.0],
        ATOL,
        [0, 1],
        forgive_mask=F.reference_settled(bn, am, 1.0, [1.0, 1.0])
        | F.reference_uncertain(bn, am, gap),
    )
    assert {j: v["passed"] for j, v in ours.items()} == {j: v["passed"] for j, v in shared.items()}
    assert not ours[1]["passed"]
