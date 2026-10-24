"""Issue #506: every path that puts a model on the finite-difference Jacobian says
why — one INFO line on the ``bngsim`` logger at the decline site, and the same
reason readable as ``Model.analytical_jacobian_status``. Issue #538: each with a
stable kind beside it (``Model.analytical_jacobian_decline``), which a census
groups by.

Before, four of the decline paths — an un-inlinable function, a parse failure, a
symbol that survived inlining, a derivative no emitter prints — and the C++
attach gate went silent at every level; a harness could record nothing but the
boolean, and recovering a reason took a monkeypatch of ``differentiate_rate_law``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import bngsim
import pytest
import sympy as sp
from bngsim import _jacobian as J


def _decline_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "bngsim"
        and r.levelno == logging.INFO
        and "analytical Jacobian" in r.getMessage()
    ]


@pytest.fixture
def info(caplog):
    with caplog.at_level(logging.INFO, logger="bngsim"):
        yield caplog


# ─── The decline sites in differentiate_rate_law ─────────────────────────────


def test_function_cycle_is_logged(info):
    assert J.build_per_observable_terms("g*A", {"g": "g+1"}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "'g*A'" in line and "cycle" in line and "finite-difference" in line
    assert J.last_decline()["kind"] == "function_cycle"


def test_parse_failure_is_logged(info):
    assert J.build_per_observable_terms("k*A*", {}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "'k*A*'" in line and "could not be parsed" in line
    assert J.last_decline()["kind"] == "unparsed"


def test_surviving_symbol_is_named(info):
    assert J.build_per_observable_terms("k*A*Z", {}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "['Z']" in line and "hidden state" in line
    assert J.last_decline()["kind"] == "unresolved_symbol"


def test_underivable_function_is_named(info):
    # mratio has no closed form in its first argument (issue #457).
    assert J.build_per_observable_terms("k*mratio(A, 1, 2)", {}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "with respect to A carries an unevaluated Derivative" in line
    assert J.last_decline()["kind"] == "non_emittable"


def test_non_emittable_derivative_names_the_node_and_the_variable(info, monkeypatch):
    reason = "carries the function Heaviside, which no emitter spells"
    monkeypatch.setattr(J, "_non_emittable_reason", lambda e: reason)
    # differentiate_rate_law directly: the term builders try the native saturable
    # path first, and a rate law inside that family never reaches sympy.
    assert J.differentiate_rate_law("k*exp(A)", {}, {"A"}, {"k"}) is None
    (line,) = _decline_lines(info)
    assert "with respect to A carries the function Heaviside" in line
    decline = J.last_decline()
    assert decline["kind"] == "non_emittable"
    assert decline["reason"] in line


def test_an_unset_volume_in_a_chain_rule_factor_is_named(info):
    smeta = {0: (True, float("nan"))}  # amount-valued, its compartment size unset
    assert (
        J.build_per_species_terms("k*A_tot*A_tot", {}, {"A_tot": [(0, 1.0)]}, smeta, {"k"}) is None
    )
    (line,) = _decline_lines(info)
    assert "non-finite chain-rule factor (nan)" in line
    assert J.last_decline()["kind"] == "nonfinite_factor"


@pytest.mark.parametrize("path", ["per_species", "per_observable"])
def test_a_derivative_that_does_not_print_is_named(path, info, monkeypatch):
    monkeypatch.setattr(J, "sympy_to_exprtk", lambda *a, **k: None)
    if path == "per_species":
        terms = J.build_per_species_terms(
            "k*sin(A_tot)", {}, {"A_tot": [(0, 1.0)]}, {0: (False, 1.0)}, {"k"}
        )
    else:
        terms = J.build_per_observable_terms("k*sin(A)", {}, {"A"}, {"k"})
    assert terms is None
    (line,) = _decline_lines(info)
    assert "could not be printed as ExprTk" in line
    assert J.last_decline()["kind"] == "unprintable"


def test_a_derivative_that_folds_non_finite_is_its_own_kind():
    # 0^(A-1) over a state: a symbolic zoo the derivative cannot shed.
    assert J.differentiate_rate_law("k*if(B>1,B,0)^(A-1)", {}, {"A", "B"}, {"k"}) is None
    decline = J.last_decline()
    assert decline["kind"] == "nonfinite_derivative"
    assert "zoo / oo / nan" in decline["reason"]


def test_the_vocabulary_is_exactly_the_kinds_recorded(monkeypatch):
    """Every kind a site records is in DECLINE_KINDS, and every entry there is a
    kind something records: the Python sites' literals, whether passed to
    _declined, assigned first, or written into a dict, and the C++ gate's
    kind_name() tokens, which pass through under their own names."""
    import inspect
    import re

    from bngsim import _model

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    from _source_root import bngsim_source_root

    root = bngsim_source_root()
    header = None if root is None else root / "include" / "bngsim" / "types.hpp"
    if header is None or not header.exists():
        pytest.skip("include/bngsim/types.hpp is not in this checkout")
    literal = r'(?:\bkind\s*=\s*|"kind":\s*)"(\w+)"'
    python = {k for mod in (J, _model) for k in re.findall(literal, inspect.getsource(mod))}
    gate = set(re.findall(r'case Kind::\w+:\s*return "(\w+)";', header.read_text())) - {"none"}
    assert {"non_emittable", "unexpected_error", "gate_unrecorded"} <= python, python
    assert {"fd_mismatch", "no_sparsity"} <= gate, gate
    assert set(J._GATE_VERDICT_FIELDS) == gate  # which fields each gate kind sets
    assert python | gate == set(J.DECLINE_KINDS), (
        (python | gate) - set(J.DECLINE_KINDS),
        set(J.DECLINE_KINDS) - (python | gate),
    )


def test_a_rate_law_that_derives_logs_nothing(info):
    assert J.build_per_observable_terms("k*A*B", {}, {"A", "B"}, {"k"}) is not None
    assert _decline_lines(info) == []


def test_non_emittable_reasons_name_the_offending_node():
    x, y = sp.symbols("x y")
    assert "Heaviside" in J._non_emittable_reason(sp.Heaviside(x))
    assert "zoo" in J._non_emittable_reason(sp.zoo * x)
    assert "Derivative" in J._non_emittable_reason(sp.Derivative(sp.sign(x), x))
    assert "Xor" in J._non_emittable_reason(sp.Xor(x > 0, y > 0))
    assert J._non_emittable_reason(sp.exp(x) * sp.Piecewise((x, x > 0), (0, True))) is None
    assert J._is_emittable(sp.exp(x)) is True


# ─── Model.analytical_jacobian_status ────────────────────────────────────────

_NET_MRATIO = """\
begin parameters
    1 k 0.7
end parameters
begin species
    1 A() 1.0
    2 B() 0.0
end species
begin functions
    1 fMratio() k*mratio(A_tot, 1, 2)
end functions
begin reactions
    1 1 2 fMratio #A_to_B
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
end groups
"""

_NET_SQRT_AT_ZERO = (
    _NET_MRATIO.replace("1 A() 1.0", "1 A() 0.0")
    .replace("fMratio() k*mratio(A_tot, 1, 2)", "fMratio() k*sqrt(A_tot)")
    .replace("1 1 2 fMratio", "1 0 2 fMratio")
)


def _model(tmp_path: Path, text: str) -> bngsim.Model:
    p = tmp_path / "m.net"
    p.write_text(text)
    return bngsim.Model.from_net(str(p))


def test_status_is_complete_for_a_mass_action_model(data_dir: Path):
    m = bngsim.Model.from_net(str(data_dir / "simple_decay.net"))
    assert m.analytical_jacobian_status == "complete"  # closed form from the start
    assert m.analytical_jacobian_decline is None


def test_status_is_pending_until_derived_then_declined_with_the_reason(tmp_path, info):
    m = _model(tmp_path, _NET_MRATIO)
    assert m.analytical_jacobian_status == "pending"
    assert m.prepare_analytical_jacobian() is False
    status = m.analytical_jacobian_status
    assert status.startswith(
        "declined: the derivative of rate law 'k*mratio(A_tot, 1, 2)' with respect to A_tot"
    ), status
    (line,) = _decline_lines(info)
    assert status[len("declined: ") :] in line  # the same reason the log line carries
    assert m.analytical_jacobian_decline == {
        "kind": "non_emittable",
        "reason": status[len("declined: ") :],
    }


def test_status_after_the_cpp_gate_declines(tmp_path, info):
    m = _model(tmp_path, _NET_SQRT_AT_ZERO)
    assert m.prepare_analytical_jacobian() is False
    status = m.analytical_jacobian_status  # issue #534: the verdict names the entry
    assert status == (
        "declined: ∂f[B()]/∂[A()] is non-finite (inf) at probe 0 (the seed state) while "
        "the RHS is finite"
    ), status
    assert any(status[len("declined: ") :] in line for line in _decline_lines(info))
    decline = m.analytical_jacobian_decline  # issue #538: the gate's kind and fields
    assert decline["kind"] == "nonfinite_entry"
    assert decline["reason"] == status[len("declined: ") :]
    assert (decline["probe"], decline["row_species"], decline["col_species"]) == (0, "B()", "A()")


def test_status_is_complete_once_attached(data_dir: Path):
    m = bngsim.Model.from_net(str(data_dir / "jac_selfcheck_switching_surface.net"))
    assert m.analytical_jacobian_status == "pending"
    assert m.analytical_jacobian_decline is None
    assert m.prepare_analytical_jacobian() is True
    assert m.analytical_jacobian_status == "complete"
    assert m.analytical_jacobian_decline is None


def test_clone_carries_the_status(tmp_path):
    m = _model(tmp_path, _NET_MRATIO)
    m.prepare_analytical_jacobian()
    assert m.clone().analytical_jacobian_status == m.analytical_jacobian_status
    assert m.clone().analytical_jacobian_decline == m.analytical_jacobian_decline


def test_env_switch_is_a_named_decline(tmp_path, monkeypatch, info):
    monkeypatch.setenv("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", "0")
    m = _model(tmp_path, _NET_MRATIO)
    assert m.prepare_analytical_jacobian() is False
    expected = "declined: disabled by BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0"
    assert m.analytical_jacobian_status == expected
    assert m.analytical_jacobian_decline["kind"] == "disabled"


def test_a_decline_that_recorded_nothing_still_has_a_kind(tmp_path, monkeypatch):
    def silent(core):  # an attach that declines without saying why
        J._reset_decline()
        return False

    monkeypatch.setattr(J, "attach_functional_jacobian", silent)
    m = _model(tmp_path, _NET_MRATIO)
    assert m.prepare_analytical_jacobian() is False
    assert m.analytical_jacobian_status == "declined: no reason was recorded"
    assert m.analytical_jacobian_decline == {
        "kind": "unrecorded",
        "reason": "no reason was recorded",
    }


def test_an_unexpected_error_is_its_own_kind(tmp_path, monkeypatch):
    def boom(core):
        raise RuntimeError("boom")

    monkeypatch.setattr(J, "attach_functional_jacobian", boom)
    m = _model(tmp_path, _NET_MRATIO)
    assert m.prepare_analytical_jacobian() is False
    reason = "an unexpected error during derivation: boom"
    assert m.analytical_jacobian_status == f"declined: {reason}"
    assert m.analytical_jacobian_decline == {"kind": "unexpected_error", "reason": reason}


class _Core:
    """A model's core with one of its methods made to raise."""

    def __init__(self, core, raising: str):
        self._core, self._raising = core, raising

    def __getattr__(self, name):
        if name == self._raising:

            def fail(*args, **kwargs):
                raise RuntimeError(f"{name} failed")

            return fail
        return getattr(self._core, name)


@pytest.mark.parametrize(
    ("raising", "kind", "said"),
    [
        ("functional_jacobian_context", "no_context", "could not provide its functional-Jacobian"),
        ("set_functional_jacobian", "attach_raised", "the C++ attach raised RuntimeError"),
    ],
)
def test_a_core_that_raises_is_a_named_decline(tmp_path, info, raising, kind, said):
    core = _Core(_model(tmp_path, _NET_SQRT_AT_ZERO)._core, raising)
    assert J.attach_functional_jacobian(core) is False
    (line,) = _decline_lines(info)
    assert said in line and f"{raising} failed" in line
    assert J.last_decline()["kind"] == kind
