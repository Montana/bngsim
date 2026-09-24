"""A table function's index is part of the Jacobian sparsity pattern (issue #781).

`ModelBuilder::build` rewrites a tfun-bodied function to `tfun_<name>()` (and an
embedded call to `tfun_<f>__tfun<k>()`) before `build_jac_sparsity` scans the
function expressions. The index an observable- or function-indexed table reads
was therefore invisible to the scan, and every species behind it dropped out of
the pattern. Three things downstream trusted that pattern and were silently
wrong:

* ``Model.jacobian(sparse=True)`` returned 0 where the dense and
  finite-difference Jacobians have a nonzero;
* ``Model.pure_sink_species()`` reported the indexed species as an inert
  accumulator, although the rate law reads it;
* the steady-state masking recipe built on that list froze the species and
  reported a converged state the dynamics never reach.

The network below is A <-> B with A -> B at rate ``F * A``, where ``F`` is a
table on ``Ptot`` (= P) with slope 1/10, plus Q -> Q + P so that P accumulates.
At y = [A, B, Q, P] = [5, 5, 1, 50], ∂(dA/dt)/∂P = -A/10 = -0.5 and
∂(dB/dt)/∂P = +0.5.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest

pytest.importorskip("scipy")

Y = np.array([5.0, 5.0, 1.0, 50.0])
A, B, P = 0, 1, 3

#: Every way a .net can put an observable- or function-indexed table behind a
#: rate law: inline data, the documented file form, a call embedded in
#: arithmetic (synthetic table name), and an index that is itself a function,
#: whole-body and embedded.
INDEXED_FORMS = {
    "inline_observable": ["F() tfun([0,100],[0,10], Ptot)"],
    "file_observable": ["F() tfun('p.tfun', Ptot)"],
    "embedded_observable": ["F() 2*tfun([0,100],[0,5], Ptot)"],
    "function_index": ["G() Ptot*1", "F() tfun([0,100],[0,10], G)"],
    # F before G on purpose: the loader hands an embedded call over already
    # rewritten, so G was missing from the evaluation-order sort as well.
    "embedded_function_index": ["F() 2*tfun([0,100],[0,5], G)", "G() Ptot*1"],
}


def _write_net(tmp_path: Path, functions: list[str], name: str = "m") -> Path:
    (tmp_path / "p.tfun").write_text("# Ptot F\n0 0\n100 10\n")
    funcs = "\n".join(f"  {i} {f}" for i, f in enumerate(functions, start=1))
    text = textwrap.dedent(
        """
        begin parameters
          1 kb 1.0
          2 kp 1.0
          3 c  20.0
        end parameters
        begin species
          1 A() 5
          2 B() 5
          3 Q() 1
          4 P() 50
        end species
        begin functions
        FUNCS
        end functions
        begin reactions
          1 1 2 F
          2 2 1 kb
          3 3 3,4 kp
        end reactions
        begin groups
          1 Ptot 4
        end groups
        """
    ).strip()
    path = tmp_path / f"{name}.net"
    path.write_text(text.replace("FUNCS", funcs) + "\n")
    return path


def _dense_and_sparse(model: bngsim.Model) -> tuple[np.ndarray, np.ndarray]:
    dense = np.asarray(model.jacobian(Y, sparse=False))
    sparse = model.jacobian(Y, sparse=True)
    return dense, np.asarray(sparse.toarray())


@pytest.mark.parametrize("form", sorted(INDEXED_FORMS))
class TestIndexedTableReachesThePattern:
    def test_sparse_jacobian_matches_dense(self, tmp_path: Path, form: str) -> None:
        model = bngsim.Model.from_net(str(_write_net(tmp_path, INDEXED_FORMS[form])))
        dense, sparse = _dense_and_sparse(model)
        assert dense[A, P] == pytest.approx(-0.5)
        assert dense[B, P] == pytest.approx(0.5)
        np.testing.assert_allclose(sparse, dense, rtol=0, atol=1e-9)

    def test_indexed_species_is_not_a_pure_sink(self, tmp_path: Path, form: str) -> None:
        model = bngsim.Model.from_net(str(_write_net(tmp_path, INDEXED_FORMS[form])))
        assert "P()" not in model.pure_sink_species()


def test_function_index_declared_after_the_table(tmp_path: Path) -> None:
    """The function→function edge must not depend on declaration order."""
    functions = ["F() tfun([0,100],[0,10], G)", "G() Ptot*1"]
    model = bngsim.Model.from_net(str(_write_net(tmp_path, functions)))
    pairs = _pattern(model)
    assert (A, P) in pairs
    assert (B, P) in pairs


def test_an_embedded_call_orders_its_function_index_first(tmp_path: Path) -> None:
    """The RHS must be a function of y alone. An embedded call reaches the builder
    as `tfun_F__tfun1()`, so the sort that orders functions after what they read
    (GH #76) could not see G: F, declared first, read G from the previous RHS
    evaluation and gave dA/dt = -20 at P = 80 after a call at P = 50."""
    functions = ["F() 2*tfun([0,100],[0,5], G)", "G() Ptot*1"]
    model = bngsim.Model.from_net(str(_write_net(tmp_path, functions)))
    model.rhs(Y)
    y = Y.copy()
    y[P] = 80.0
    # F = 2 * (G / 20) = P / 10, so dA/dt = -(P / 10) * A + kb * B.
    assert np.asarray(model.rhs(y))[A] == pytest.approx(-(80.0 / 10.0) * 5.0 + 5.0)


def _pattern(model: bngsim.Model) -> set[tuple[int, int]]:
    """(row, col) pairs of the model's CSC sparsity pattern."""
    pat = model._core.jacobian_sparsity
    colptrs = list(pat["col_ptrs"])
    rowvals = list(pat["row_indices"])
    return {
        (int(rowvals[k]), j)
        for j in range(len(colptrs) - 1)
        for k in range(colptrs[j], colptrs[j + 1])
    }


@pytest.mark.parametrize(
    "fline",
    ["F() tfun([0,100],[0,10], time)", "F() tfun([0,100],[0,10], c)"],
    ids=["time_index", "parameter_index"],
)
def test_time_and_parameter_indices_add_no_species(tmp_path: Path, fline: str) -> None:
    """The fix resolves the index; it must not densify the pattern.

    A time- or parameter-indexed table reads no species, so column P stays
    empty for A and B, exactly as before the fix.
    """
    model = bngsim.Model.from_net(str(_write_net(tmp_path, [fline])))
    pairs = _pattern(model)
    assert (A, P) not in pairs
    assert (B, P) not in pairs
    dense, sparse = _dense_and_sparse(model)
    np.testing.assert_allclose(sparse, dense, rtol=0, atol=1e-9)
