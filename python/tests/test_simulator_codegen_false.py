"""``Simulator(codegen=False)`` runs the interpreter on a model that carries compiled
code (issue #877).

An auto-compiling or ``codegen=True`` Simulator writes its artifact onto the Model,
and ``Model.clone()`` copies it. The reuse block in ``Simulator.__init__`` handed
that artifact to every later Simulator on the model, so a ``codegen=False`` one ran
``cc`` (or the MIR JIT) although it compiled nothing itself, and a ``codegen=False``
sensitivity run skipped its GH #214 refusal.

``BNGSIM_CODEGEN_THRESHOLD=1`` makes the two-species fixture auto-compile, so the
auto-codegen case needs no 256-species network. Each test first asserts that the
model really carries the artifact, so none of them can pass by never having had
one to inherit.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest


def _has_cc() -> bool:
    try:
        from bngsim._codegen import _find_c_compiler

        _find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


@pytest.fixture
def net(data_dir, monkeypatch):
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)
    return str(data_dir / "derived_rate_const.net")


def _species(model, sim):
    model.reset()  # run() advances the shared core; start each from the IC
    r = sim.run(t_span=(0.0, 5.0), n_points=11, rtol=1e-10, atol=1e-12)
    return np.asarray(r.species)


@requires_cc
@pytest.mark.parametrize("first_codegen", [None, True], ids=["after-auto", "after-codegen-true"])
def test_codegen_false_ignores_the_models_so(net, first_codegen):
    m = bngsim.Model.from_net(net)
    compiled = bngsim.Simulator(m, method="ode", codegen=first_codegen)
    assert compiled.codegen_backend == "cc"
    assert m._codegen_so_path

    interp = bngsim.Simulator(m, method="ode", codegen=False)
    assert interp.codegen_backend == "exprtk"
    np.testing.assert_allclose(_species(m, interp), _species(m, compiled), rtol=1e-7, atol=1e-12)


@requires_cc
def test_codegen_false_ignores_the_so_a_clone_copied(net):
    m = bngsim.Model.from_net(net)
    bngsim.Simulator(m, method="ode")
    c = m.clone()
    assert c._codegen_so_path

    assert bngsim.Simulator(c, method="ode", codegen=False).codegen_backend == "exprtk"


def test_codegen_false_ignores_the_models_jit_source(net, monkeypatch):
    # Building a Simulator under the JIT backend only generates C source; nothing
    # is compiled or run, so this holds on a build without MIR too.
    monkeypatch.setenv("BNGSIM_CODEGEN_JIT", "mir")
    m = bngsim.Model.from_net(net)
    assert bngsim.Simulator(m, method="ode").codegen_backend == "mir"
    assert m._codegen_c_source

    assert bngsim.Simulator(m, method="ode", codegen=False).codegen_backend == "exprtk"


@requires_cc
def test_codegen_false_sensitivity_run_refuses_despite_the_models_artifact(net):
    m = bngsim.Model.from_net(net)
    bngsim.Simulator(m, method="ode", sensitivity_params=["kon"])
    assert m._codegen_so_path

    with pytest.raises(ValueError, match="codegen=False was passed"):
        bngsim.Simulator(m, method="ode", codegen=False, sensitivity_params=["kon"])


@requires_cc
def test_default_simulator_still_reuses_the_models_so(net, monkeypatch):
    # Above the threshold the default Simulator would not compile this model, so
    # 'cc' here can only come from the artifact the codegen=True one left, and the
    # codegen=False Simulator in between must not have taken it away.
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "256")
    m = bngsim.Model.from_net(net)
    compiled = bngsim.Simulator(m, method="ode", codegen=True)
    bngsim.Simulator(m, method="ode", codegen=False)

    reused = bngsim.Simulator(m, method="ode")
    assert reused.codegen_backend == "cc"
    assert reused._codegen_so_path == compiled._codegen_so_path
