"""Each bng_ode parity arm runs the backend it is named for (issue #872).

Since #825 bngsim compiles a ``.net`` model with at least
``BNGSIM_CODEGEN_THRESHOLD`` (256) species by default. The nightly interpreter arm
passed no ``codegen`` argument, so 26 of its 592 jobs ran compiled: the arm
stopped testing ExprTk on the largest models and its wall time grew x5.8 from
C compiles. ``codegen=False`` now pins ExprTk, ``True`` pins the compiled RHS, and
``None`` leaves bngsim's own choice for the timing benchmarks that want it.

Each case runs at the ``BNGSIM_CODEGEN_THRESHOLD`` where bngsim's own choice
would pick the other backend, so only an honoured pin passes. At a threshold of 1
the two-species fixture auto-compiles, and ``False`` has to stop that. At the
default 256 it runs ExprTk, and ``True`` has to force a compile. ``None`` runs at
both and follows the threshold. Ambient ``BNGSIM_NO_CODEGEN`` and
``BNGSIM_CODEGEN_JIT`` are cleared, since either one changes the backend by
itself. Both harness entry points are covered: ``bn_ode_net`` (single segment)
and ``multi_segment_replay`` (the dirty-carryover protocol arm).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_BNG_PARITY = Path(__file__).resolve().parent.parent / "bng_parity"
_NET = Path(__file__).resolve().parents[2] / "tests" / "data" / "derived_rate_const.net"


def _load_bng_common():
    spec = importlib.util.spec_from_file_location("_bng_common", _BNG_PARITY / "_bng_common.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_bng_common()


def _have_cc() -> bool:
    try:
        from bngsim._codegen import _find_c_compiler

        _find_c_compiler()
        return True
    except Exception:
        return False


_NEEDS_CC = pytest.mark.skipif(not _have_cc(), reason="needs bngsim and a C compiler for codegen")

# (codegen, BNGSIM_CODEGEN_THRESHOLD, backend that must run)
_CASES = [
    pytest.param(False, "1", "exprtk", id="false-pins-exprtk"),
    pytest.param(True, "256", "cc", id="true-pins-compiled", marks=_NEEDS_CC),
    pytest.param(None, "1", "cc", id="none-auto-compiles", marks=_NEEDS_CC),
    pytest.param(None, "256", "exprtk", id="none-auto-interprets"),
]


@pytest.fixture
def codegen_env(monkeypatch):
    pytest.importorskip("bngsim")
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)
    return monkeypatch


@pytest.mark.parametrize(("codegen", "threshold", "backend"), _CASES)
def test_bn_ode_net_runs_the_pinned_backend(codegen_env, codegen, threshold, backend):
    codegen_env.setenv("BNGSIM_CODEGEN_THRESHOLD", threshold)
    _t, _v, _n, timing = C.bn_ode_net(_NET, 0.0, 1.0, 11, 1e-8, 1e-8, codegen=codegen)
    assert timing["config"]["codegen"] == backend


@pytest.mark.parametrize(("codegen", "threshold", "backend"), _CASES)
def test_multi_segment_replay_runs_the_pinned_backend(codegen_env, codegen, threshold, backend):
    codegen_env.setenv("BNGSIM_CODEGEN_THRESHOLD", threshold)
    step = {
        "kind": "sim",
        "method": "ode",
        "t_start": 0.0,
        "t_end": 1.0,
        "n_steps": 10,
        "rtol": None,
        "atol": None,
    }
    _result, info = C.multi_segment_replay(
        _NET,
        [step],
        0,
        track="ode",
        atol=1e-8,
        rtol=1e-8,
        seed=None,
        poplevel=0.0,
        codegen=codegen,
    )
    assert info["backend"] == backend
