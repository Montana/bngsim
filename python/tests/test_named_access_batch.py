"""GH #560 — named column access must work on a squeezed batch Result.

``_ObservableAccessor.__getitem__`` indexed a named column as ``[:, idx]``.
That is the right axis only for a single run's ``(n_times, n_cols)`` block. A
squeezed batch (``run_batch(..., squeeze=True)``) carries one more leading
axis, ``(n_sims, n_times, n_cols)``, so the same subscript took the TIME axis:
``batch.observables["X"]`` came back ``(n_sims, n_cols)`` — every replicate's
values at time row ``idx``, one column per observable — instead of ``(n_sims,
n_times)`` trajectories.

Nothing raised, because the result has the rank the caller expects and holds
real numbers from the run: for the common ``idx == 0`` it is each replicate's
initial values, which look like a plausible first column. The sibling
``Result.outputs("observable:X")`` indexes the last axis and was right
throughout, so the two APIs disagreed on the same result.

The column lives on the last axis in both layouts, so the lookup indexes there.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 k       0.3  # Constant
    2 k2      0.1  # Constant
end parameters
begin functions
    1 f() k*X
end functions
begin species
    1 X() 100
    2 Y() 40
end species
begin reactions
    1 1 0 k #_R1
    2 2 0 k2 #_R2
end reactions
begin groups
    1 X                    1
    2 Y                    2
end groups
"""

N_POINTS = 11
PARAMS = [{"k": 0.3}, {"k": 0.5}, {"k": 0.7}]


@pytest.fixture
def net(tmp_path):
    p = tmp_path / "decay.net"
    p.write_text(NET)
    return str(p)


def _model(net):
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(net)


@pytest.fixture
def batch(net):
    """A squeezed batch: (n_sims, n_times, n_cols) on every block."""
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Simulator(_model(net), method="ode").run_batch(
            t_span=(0.0, 10.0), n_points=N_POINTS, params=PARAMS, squeeze=True
        )


@pytest.fixture
def single(net):
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Simulator(_model(net), method="ode").run(
            t_span=(0.0, 10.0), n_points=N_POINTS
        )


# ── The batch layout ─────────────────────────────────────────────────────────


def test_the_batch_block_really_is_three_dimensional(batch):
    """Guards the premise: without the extra axis this file proves nothing."""
    assert np.asarray(batch.observables).shape == (len(PARAMS), N_POINTS, 2)


@pytest.mark.parametrize("block, name", [("observables", "X"), ("expressions", "f")])
def test_a_named_column_is_the_trajectory_not_a_time_slice(batch, block, name):
    col = getattr(batch, block)[name]
    assert col.shape == (len(PARAMS), N_POINTS)
    # Each row moves: the pre-fix answer was one constant per replicate.
    assert all(row[-1] < row[0] for row in col)


def _names_of(result, block):
    return result.expression_names if block == "expressions" else result.observable_names


@pytest.mark.parametrize("block, name", [("observables", "X"), ("expressions", "f")])
def test_named_access_agrees_with_the_raw_block(batch, block, name):
    data = np.asarray(getattr(batch, block))
    idx = _names_of(batch, block).index(name)
    assert np.array_equal(getattr(batch, block)[name], data[..., idx])


def test_named_access_agrees_with_outputs(batch):
    """The two APIs read the same column; they used to disagree on a batch."""
    assert np.array_equal(batch.observables["X"], batch.outputs("observable:X")[..., 0])


def test_two_names_do_not_return_the_same_column(batch):
    """Pre-fix these differed too — as two TIME slices rather than two columns —
    so this is a standing property, not the signature of the bug."""
    assert not np.array_equal(batch.observables["X"], batch.observables["Y"])


# ── The single-run layout is untouched ───────────────────────────────────────


@pytest.mark.parametrize("block, name", [("observables", "X"), ("expressions", "f")])
def test_a_single_run_is_unchanged(single, block, name):
    col = getattr(single, block)[name]
    assert col.shape == (N_POINTS,)
    data = np.asarray(getattr(single, block))
    assert np.array_equal(col, data[:, _names_of(single, block).index(name)])


def test_an_integer_key_still_indexes_the_leading_axis(batch, single):
    """Unchanged by the fix: a positional key means what it means on the array
    itself — a time row on a single run, a replicate on a batch."""
    assert np.array_equal(single.observables[0], np.asarray(single.observables)[0])
    assert np.array_equal(batch.observables[0], np.asarray(batch.observables)[0])


def test_an_unknown_name_still_raises(batch):
    with pytest.raises(KeyError, match="nope"):
        batch.observables["nope"]
