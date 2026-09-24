"""GH #576 — the state handed to the JAX Jacobian callback must own its data.

``set_jax_jac_fn`` wrapped CVODE's ``N_Vector`` storage as a zero-copy numpy
view whose base was ``py::cast(0)`` — an integer, which owns nothing and keeps
nothing alive. A callback that retained the array (a trace, a debug log, a
closure over it) was left holding a view into memory CVODE frees at teardown.
Reading it afterwards returned whatever had since been allocated there:

    kept[0] after teardown: [8.53184889e-316, 8.71335754e-316]

for a model seeded ``[100.0, 0.0]`` — plausible-looking floats, never an error,
so nothing downstream had cause to notice.

The view was also writeable over a ``const double *``. Writing into it did not
move the trajectory in the cases probed here, since CVODE does not read that
vector again after the Jacobian call, but the binding was casting away a const
it had no business casting away.

The callback now gets a copy: ``ns`` doubles, next to the ``ns²`` work it is
about to do. It survives teardown, and writing into it is the caller's own
business.

These go through ``bngsim._bngsim_core`` directly — the defect is in the
C++/pybind11 bridge, so no JAX is needed.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
from bngsim._bngsim_core import CvodeSimulator, NetworkModel, SolverOptions, TimeSpec

N_SPECIES = 2
A0, B0 = 100.0, 0.0

NET = f"""begin parameters
    1 k 0.1
end parameters
begin species
    1 A() {A0}
    2 B() {B0}
end species
begin reactions
    1 1 2 k #_R1
end reactions
begin groups
    1 A_t 1
    2 B_t 2
end groups
"""


@pytest.fixture
def net(tmp_path):
    p = tmp_path / "decay.net"
    p.write_text(NET)
    return str(p)


def _run(net, callback, *, t_end=50.0, n_points=20):
    model = NetworkModel.from_net(net)
    opts = SolverOptions()
    opts.jacobian = "jax"
    opts.set_jax_jac_fn(callback)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_points
    return np.asarray(CvodeSimulator(model).run(ts, opts).species_data)


def _zero_jac(t, y):
    return np.zeros(N_SPECIES * N_SPECIES, dtype=np.float64)


# ── The array the callback receives ──────────────────────────────────────────


def test_it_owns_its_data(net):
    """`base is None` is the property that makes the rest of this file true:
    the array is not a view into somebody else's buffer. It used to be the
    integer 0, which owns nothing."""
    seen = []

    def cb(t, y):
        seen.append((y.base, y.shape, y.dtype))
        return _zero_jac(t, y)

    _run(net, cb)
    assert seen, "the Jacobian callback was never called"
    base, shape, dtype = seen[0]
    assert base is None
    assert shape == (N_SPECIES,)
    assert dtype == np.float64


def test_a_retained_array_survives_solver_teardown(net):
    """The defect: after the run is gone and the heap has churned, this read
    freed memory and returned plausible numbers."""
    kept = []

    def cb(t, y):
        kept.append(y)
        return _zero_jac(t, y)

    _run(net, cb)
    gc.collect()
    for _ in range(200):  # churn the heap so a freed buffer is reused
        np.zeros(1 << 16)

    first = np.asarray(kept[0])
    assert np.isfinite(first).all()
    # The state at the first Jacobian evaluation: A has barely moved from its
    # initial value, and every entry is a real concentration rather than a
    # denormal left over from whatever took that memory next.
    assert first[0] == pytest.approx(A0, rel=1e-3)
    assert first[1] == pytest.approx(B0, abs=1e-3)


def test_every_retained_row_is_a_distinct_array(net):
    """Each call gets its own copy, so a callback collecting the state over a
    run ends up with the trajectory rather than N aliases of one buffer."""
    kept = []

    def cb(t, y):
        kept.append(y)
        return _zero_jac(t, y)

    _run(net, cb)
    assert len(kept) > 1
    assert len({id(a) for a in kept}) == len(kept)
    stacked = np.asarray([np.asarray(a) for a in kept])
    assert np.isfinite(stacked).all()
    # A decays, so the collected states are not all the same row.
    assert stacked[:, 0].min() < stacked[:, 0].max()


def test_writing_into_it_does_not_reach_the_integrator(net):
    """It is the caller's array now. The trajectory must be the one the model
    describes: A(t) = A0·exp(-k·t)."""

    def mutating(t, y):
        y[:] = -1.0e9
        return _zero_jac(t, y)

    species = _run(net, mutating)
    assert np.isfinite(species).all()
    assert species[0, 0] == pytest.approx(A0, rel=0, abs=0)
    assert species[-1, 0] == pytest.approx(A0 * np.exp(-0.1 * 50.0), rel=1e-6)


def test_the_state_it_carries_is_the_live_one(net):
    """A copy is only useful if it copies the right thing: the values must
    track the run, not stay pinned at the initial condition."""
    seen = []

    def cb(t, y):
        seen.append((t, float(y[0])))
        return _zero_jac(t, y)

    _run(net, cb)
    late = [a for t, a in seen if t > 25.0]
    assert late, "no Jacobian evaluation in the second half of the run"
    assert min(late) < A0 * 0.5
