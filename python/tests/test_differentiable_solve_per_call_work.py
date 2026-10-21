"""Issue #919 — work ``differentiable_solve`` repeated on every call.

Since #902/#913 every call solves through ``Simulator`` on a fresh clone of the
model, which is what makes it correct. It also made each call pay again for
work that depends on the model's structure only:

- the clone derived the analytical Functional Jacobian with SymPy, because the
  caller's model never had (a clone takes a warmed model's terms without SymPy,
  GH #145); the bridge now derives it once, on the caller's model, as a first
  ``Simulator(model).run()`` would;
- ``NetworkModel::clone`` threw and caught a C++ exception for each species
  whose name is no ExprTk symbol, which is every ``.net`` species (``A()``):
  2 ms of a clone of a 300-species model, now 0.15 ms.

What must not move is the answer: each call's primal and sensitivities are the
ones an unwarmed clone gives, bit for bit.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from bngsim._jax_bridge import _write_params, differentiable_solve  # noqa: E402

# Functional rate laws, a derived parameter one of them reads, and a primary
# that reaches a species through a derived initial condition.
FUNCTIONAL = """\
begin parameters
    1 k1  1.0
    2 k2  0.5
    3 K   2.0
    4 A0  10
    5 kd  k1*K
    6 Ai  2*A0
end parameters
begin functions
    1 r1() k1*Aobs/(K+Aobs)
    2 r2() k2*Bobs^2/(1+Bobs)
    3 r3() kd*Cobs
end functions
begin species
    1 A() Ai
    2 B() 0
    3 C() 0
end species
begin reactions
    1 1 2 r1
    2 2 3 r2
    3 3 0 r3
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
    3 Cobs 3
end groups
"""

# Its derivative switches on model state: roots, jumps and crossing stops.
STATE_WINDOW = """\
begin parameters
    1 k   10.0
    2 w   0.01
    3 kd  0.5
end parameters
begin functions
    1 onset() 4*Sobs
    2 r() if((time()>=onset())&&(time()<(onset()+w)),k,0)
end functions
begin species
    1 S() 1
    2 Y() 0
end species
begin reactions
    1 1 0 kd
    2 0 2 r
end reactions
begin groups
    1 Sobs 1
end groups
"""

T_SPAN = (0.0, 5.0)
N_POINTS = 11


def _model(tmp_path, text, name):
    p = tmp_path / name
    p.write_text(text)
    return bngsim.Model.from_net(p)


def _names(model, flat):
    if flat:
        internal = model._internal_param_names()
        return tuple(n for n in model.param_names if n not in internal)
    return tuple(model.primary_param_names)


def _p0(model, flat):
    # Off the nominal point, so that every value written is a write.
    return jnp.array([1.1 * model.get_param(n) for n in _names(model, flat)])


def _unwarmed(path, names, values, flat, sens):
    """What a call ran before #919: a clone of a model that has derived
    nothing, the values written, and its own Simulator."""
    clone = bngsim.Model.from_net(path).clone()
    _write_params(clone, names, np.asarray(values), flat)
    clone.reset()
    kwargs = {"sensitivity_params": list(names)} if sens else {}
    return bngsim.Simulator(clone, method="ode", **kwargs).run(
        t_span=T_SPAN, n_points=N_POINTS, rtol=1e-8, atol=1e-8, max_steps=10000
    )


@pytest.mark.parametrize("flat", [False, True], ids=["primary", "flat"])
@pytest.mark.parametrize("text", [FUNCTIONAL, STATE_WINDOW], ids=["functional", "state-window"])
def test_each_call_is_the_unwarmed_clone_bit_for_bit(tmp_path, text, flat):
    model = _model(tmp_path, text, "m.net")
    path = tmp_path / "m.net"
    names = _names(model, flat)
    p = _p0(model, flat)

    def solve(q):
        return differentiable_solve(model, q, T_SPAN, N_POINTS, flat=flat)

    for _ in range(2):  # the first call warms the model, the second finds it warm
        primal = np.asarray(solve(p))
        np.testing.assert_array_equal(
            primal, np.asarray(_unwarmed(path, names, p, flat, sens=False).species)
        )
        jac = np.asarray(jax.jacfwd(solve)(p))
        ref = _unwarmed(path, names, p, flat, sens=True)
        np.testing.assert_array_equal(jac, np.asarray(ref.sensitivities))


def test_the_jacobian_is_derived_once_not_per_call(tmp_path, monkeypatch):
    import bngsim._jacobian as jacobian

    calls = []
    attach = jacobian.attach_functional_jacobian

    def counting(core):
        calls.append(core)
        return attach(core)

    monkeypatch.setattr(jacobian, "attach_functional_jacobian", counting)

    model = _model(tmp_path, FUNCTIONAL, "m.net")
    assert model.analytical_jacobian_status == "pending"
    p = _p0(model, False)

    def loss(q):
        return jnp.sum(differentiable_solve(model, q, T_SPAN, N_POINTS) ** 2)

    for _ in range(3):
        differentiable_solve(model, p, T_SPAN, N_POINTS)
        jax.grad(loss)(p)
        differentiable_solve(model, p, T_SPAN, N_POINTS, chunk_size=1)
    assert len(calls) == 1
    assert model.analytical_jacobian_status == "complete"


def test_a_call_while_another_thread_derives_waits_for_the_terms(tmp_path, monkeypatch):
    """The derivation marks the model attempted before it starts. A call from a
    second thread that read that mark unlocked, while the first thread was still
    deriving, cloned a model with no terms, marked attempted, so its clone never
    derived and solved on the finite-difference Jacobian: a gradient that was not
    the one the same call gives alone. It now waits for the terms.
    """
    import threading
    import time

    import bngsim._jacobian as jacobian

    model = _model(tmp_path, FUNCTIONAL, "m.net")
    names = _names(model, False)
    p = _p0(model, False)
    ref = np.asarray(_unwarmed(tmp_path / "m.net", names, p, False, sens=True).sensitivities)

    attach = jacobian.attach_functional_jacobian
    second_calling = threading.Event()

    def slow(core):
        # Hold the derivation open until the second call is on its way in.
        second_calling.wait(timeout=10)
        time.sleep(0.5)
        return attach(core)

    monkeypatch.setattr(jacobian, "attach_functional_jacobian", slow)

    def solve(q):
        return differentiable_solve(model, q, T_SPAN, N_POINTS)

    out = {}

    def call(tag):
        if tag == "second":
            second_calling.set()
        out[tag] = np.asarray(jax.jacfwd(solve)(p))

    first = threading.Thread(target=call, args=("first",))
    first.start()
    while not model._jac_attempted:  # the first call is inside the derivation
        time.sleep(0.001)
    second = threading.Thread(target=call, args=("second",))
    second.start()
    first.join()
    second.join()

    assert set(out) == {"first", "second"}
    for tag in ("first", "second"):
        np.testing.assert_array_equal(out[tag], ref, err_msg=tag)
