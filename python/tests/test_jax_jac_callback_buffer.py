"""GH #566 — the JAX Jacobian callback's array must be checked before it is copied.

``set_jax_jac_fn``'s bridge cast the callback's return to a bare
``py::array_t<double>`` and memcpy'd ``ns * ns`` doubles out of it. Nothing
checked that it held that many. A callback returning a shorter array — a stub,
a half-built matrix, a model whose species count moved under it — read foreign
heap past the end of that array straight into CVODE's Newton matrix.

The corruption does not announce itself. It surfaces as a convergence failure,
or as non-finite concentrations out of an integration that looked ordinary,
with nothing naming the callback. Two smaller hazards sat beside it: a
non-contiguous array made the copy read the wrong strides and silently integrate
the wrong Jacobian, and ``ns * ns`` was computed in ``int``, which overflows
above ns = 46340 before widening to ``size_t``. (A float32 array was already
converted — ``array_t``'s default flags carry ``forcecast`` — and is pinned here
so it stays that way.)

These go through ``bngsim._bngsim_core`` directly. The callback path is C++ and
pybind11 — JAX itself is not involved, so none of this needs JAX installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from bngsim._bngsim_core import CvodeSimulator, NetworkModel, SolverOptions, TimeSpec

N_SPECIES = 3

NET = """begin parameters
    1 k 0.1
end parameters
begin species
    1 A() 10
    2 B() 20
    3 C() 30
end species
begin reactions
    1 1 0 k #_R1
    2 2 0 k #_R2
    3 3 0 k #_R3
end reactions
begin groups
    1 A_t 1
    2 B_t 2
    3 C_t 3
end groups
"""

# The true Jacobian of this decay model: dx_i/dt = -k x_i.
JAC = np.diag(np.full(N_SPECIES, -0.1)).astype(np.float64)
FLAT = JAC.flatten(order="F")


@pytest.fixture
def net(tmp_path):
    p = tmp_path / "decay.net"
    p.write_text(NET)
    return str(p)


def _run(net, make_jac, t_end=5.0, n_points=6):
    """Integrate with a JAX-style Jacobian callback returning ``make_jac()``."""
    model = NetworkModel.from_net(net)
    opts = SolverOptions()
    opts.jacobian = "jax"
    opts.set_jax_jac_fn(lambda t, y: make_jac())
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_points
    return np.asarray(CvodeSimulator(model).run(ts, opts).species_data)


# ── The size check ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [0, 1, N_SPECIES, N_SPECIES * N_SPECIES - 1])
def test_a_short_array_is_refused(net, n):
    """The defect: these were copied over anyway, reading past the end."""
    with pytest.raises(ValueError, match="JAX Jacobian callback returned"):
        _run(net, lambda: np.zeros(n, dtype=np.float64))


def test_a_long_array_is_refused_too(net):
    """Not a memory-safety problem, but the same mistake — say so rather than
    silently using the first ns*ns elements of something else."""
    with pytest.raises(ValueError, match="JAX Jacobian callback returned"):
        _run(net, lambda: np.zeros(N_SPECIES * N_SPECIES + 1, dtype=np.float64))


def test_the_message_names_both_counts(net):
    """What the caller needs is what was returned and what was wanted."""
    with pytest.raises(ValueError) as exc:
        _run(net, lambda: np.zeros(1, dtype=np.float64))
    msg = str(exc.value)
    assert "1 element" in msg
    assert str(N_SPECIES * N_SPECIES) in msg
    assert f"{N_SPECIES}x{N_SPECIES}" in msg


# ── The layouts that must keep working ───────────────────────────────────────


def test_a_flat_column_major_array_works(net):
    """The documented shape — the baseline every other case is compared to."""
    out = _run(net, lambda: FLAT.copy())
    assert np.isfinite(out).all()
    assert out[-1, 0] < out[0, 0]


def test_a_two_dimensional_array_of_the_right_size_works(net):
    assert np.array_equal(_run(net, lambda: JAC.copy()), _run(net, lambda: FLAT.copy()))


# A -> B -> 0: dA/dt = -k*A, dB/dt = k*A - k2*B. Unlike the diagonal JAC above,
# this Jacobian is not its own transpose, so it can tell the two apart.
NET_AB = """begin parameters
    1 k 50
    2 k2 0.1
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 k #_R1
    2 2 0 k2 #_R2
end reactions
begin groups
    1 A_t 1
    2 B_t 2
end groups
"""
JAC_AB = np.array([[-50.0, 0.0], [50.0, -0.1]])  # J[i, j] = df_i/dy_j


@pytest.mark.parametrize("order", ["C", "F"])
def test_a_square_matrix_is_read_as_the_jacobian_not_its_transpose(tmp_path, order):
    """An (ns, ns) return means J[i, j] = df_i/dy_j, the layout jax.jacfwd
    produces. Read in C order it reached CVODE as J^T, and nothing said so:
    Newton tolerates an inexact matrix, so the run still finished."""
    net = tmp_path / "ab.net"
    net.write_text(NET_AB)
    want = _run(str(net), lambda: JAC_AB.flatten(order="F"))
    assert np.array_equal(_run(str(net), lambda: np.array(JAC_AB, order=order)), want)
    # Not vacuous: on this model the transpose really does integrate differently.
    assert not np.array_equal(_run(str(net), lambda: JAC_AB.T.flatten(order="F")), want)


def test_a_non_contiguous_array_is_read_correctly(net):
    """It used to be read at the wrong strides — the values the callback meant
    were skipped, and the run integrated a different matrix without saying so."""

    def strided():
        base = np.zeros(FLAT.size * 2, dtype=np.float64)
        base[1::2] = FLAT  # the real values sit on the odd slots
        return base[1::2]  # a non-contiguous view of exactly those

    assert np.array_equal(_run(net, strided), _run(net, lambda: FLAT.copy()))


def test_a_float32_array_is_converted_not_misread(net):
    """Already true before GH #566 (forcecast is array_t's default); kept so a
    change of cast flags cannot quietly undo it."""
    out32 = _run(net, lambda: FLAT.astype(np.float32))
    assert np.isfinite(out32).all()
    assert out32 == pytest.approx(_run(net, lambda: FLAT.copy()), rel=1e-6, abs=1e-8)


# ── What the callback throws reaches the caller ──────────────────────────────


def test_an_exception_from_the_callback_arrives_intact(net):
    """It is parked and rethrown once CVode has returned, rather than unwinding
    through SUNDIALS' C frames, and the caller still sees their own error —
    not the convergence failure it would otherwise become."""

    def boom():
        raise RuntimeError("callback exploded")

    with pytest.raises(RuntimeError, match="callback exploded"):
        _run(net, boom)


def test_a_non_array_return_is_refused(net):
    with pytest.raises((TypeError, ValueError)):
        _run(net, lambda: "not an array")
