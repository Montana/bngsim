"""``Simulator(method="ssa", codegen=False)`` runs the interpreted propensities
on a MIR build too (issue #1213), and with codegen left on, every engine a
Simulator builds takes the backend its plain run takes.

``codegen=False`` kept Python from building the cc propensity library and went
no further. The C++ selector read only ``BNGSIM_SSA_NO_CODEGEN``, so with no
library to load, its default fell through to ``MirJit::available()`` and
JIT'd the propensities in-process: the branch for an install with no compiler
(GH #139/#140) could not tell that apart from a caller who asked for no
codegen, and reported ``propensity_backend == "mir"``. Every place an SSA
engine is built now passes the caller's choice down: a plain run, the engine
rebuilt after ``intervene``, ``run_replicates``' per-thread engines and
``run_batch``'s per-row ones.

The same three later engines never got the library ``__init__`` built either,
so with codegen left on they ran interpreted on a build without MIR (``mir`` on
one with it) where the plain run ran ``cc``. That half fails on any build with
a C compiler; the ``codegen=False`` half passed on a build without MIR before
the fix, and is run on the MIR leg (``mir.yml``) for that reason.
"""

from __future__ import annotations

import bngsim
import pytest
from bngsim._bngsim_core import HAS_MIR, SsaSimulator, TimeSpec

NET = """begin parameters
    1 k 0.01
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1,1 2 k
end reactions
begin groups
    1 Aobs 1
end groups
"""
RUN = {"t_span": (0.0, 1.0), "n_points": 3, "seed": 1}
FLAGS = ("BNGSIM_SSA_NO_CODEGEN", "BNGSIM_SSA_PROP_CC", "BNGSIM_SSA_PROP_JIT")


@pytest.fixture
def model(tmp_path, monkeypatch):
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    path = tmp_path / "dimer.net"
    path.write_text(NET)
    return bngsim.Model.from_net(str(path))


def _backend(result):
    return result.ssa_diagnostics["propensity_backend"]


def _plain_run(sim):
    return [sim.run(**RUN)]


def _after_intervene(sim):
    sim.run(**RUN)
    sim.intervene({"k": 0.02})
    return [sim.run(**RUN)]


def _replicates_on_threads(sim):
    return sim.run_replicates(4, num_processors=2, **RUN)


def _batch_rows(sim):
    return sim.run_batch((0.0, 1.0), 3, params=[{"k": 0.01}, {"k": 0.03}], seed=1)


LATER_ENGINES = [_after_intervene, _replicates_on_threads, _batch_rows]


@pytest.mark.parametrize("entry", [_plain_run, *LATER_ENGINES], ids=lambda f: f.__name__[1:])
def test_codegen_false_is_interpreted(model, entry):
    sim = bngsim.Simulator(model, method="ssa", codegen=False)
    assert {_backend(r) for r in entry(sim)} == {"interpreted"}


@pytest.mark.parametrize("entry", LATER_ENGINES, ids=lambda f: f.__name__[1:])
def test_left_on_a_later_engine_takes_the_plain_runs_backend(model, entry):
    """The library is structure-specialized and reads every rate constant at
    run time, so a rebuilt engine, a clone's and a changed parameter's are all
    served by the one ``__init__`` built."""
    want = _backend(bngsim.Simulator(model, method="ssa").run(**RUN))
    sim = bngsim.Simulator(model, method="ssa")
    assert {_backend(r) for r in entry(sim)} == {want}


@pytest.mark.parametrize("flag", FLAGS)
def test_a_flag_set_to_zero_is_off(model, monkeypatch, flag):
    """The C++ selector reads ``0`` as off. Python read any set value as on and
    skipped the library, which the run, with no override on, then lacked."""
    want = _backend(bngsim.Simulator(model, method="ssa").run(**RUN))
    monkeypatch.setenv(flag, "0")
    assert _backend(bngsim.Simulator(model, method="ssa").run(**RUN)) == want


@pytest.mark.parametrize(("enabled", "want"), [(False, "interpreted"), (True, "mir")])
def test_the_core_engine_keeps_to_set_codegen(model, enabled, want):
    """``SsaSimulator.set_codegen`` with no library to load, what the Python
    layer passes for ``codegen=False``. Left on, a MIR build JITs: the fallback
    for an install with no compiler stays."""
    if want == "mir" and not HAS_MIR:
        want = "interpreted"
    sim = SsaSimulator(model._core)
    sim.set_codegen(enabled)
    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, 1.0, 3
    assert sim.run(times, 1).ssa_diagnostics.propensity_backend == want
