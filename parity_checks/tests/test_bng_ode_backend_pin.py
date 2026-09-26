"""Each bng_ode parity arm runs the backend it is named for (issue #872).

Since #825 bngsim compiles a ``.net`` model with at least
``BNGSIM_CODEGEN_THRESHOLD`` (256) species by default. The nightly interpreter arm
passed no ``codegen`` argument, so 26 of its 592 jobs ran compiled: the arm
stopped testing ExprTk on the largest models and its wall time grew x5.8 from
C compiles. ``codegen=False`` now pins ExprTk, ``True`` pins the compiled RHS, and
``None`` leaves bngsim's own choice for the timing benchmarks that want it.

Lowering the threshold to 1 makes a two-species fixture take the auto-compile
branch, so the pin is observable without a 256-species network. Both harness
entry points are covered: ``bn_ode_net`` (single segment) and
``multi_segment_replay`` (the dirty-carryover protocol arm).
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


_CASES = [
    pytest.param(False, "exprtk", id="false-pins-exprtk"),
    pytest.param(
        True,
        "cc",
        id="true-pins-compiled",
        marks=pytest.mark.skipif(
            not _have_cc(), reason="needs bngsim and a C compiler for codegen"
        ),
    ),
    pytest.param(
        None,
        "cc",
        id="none-keeps-auto",
        marks=pytest.mark.skipif(
            not _have_cc(), reason="needs bngsim and a C compiler for codegen"
        ),
    ),
]


@pytest.fixture
def auto_compiles_everything(monkeypatch):
    pytest.importorskip("bngsim")
    monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")


@pytest.mark.parametrize(("codegen", "backend"), _CASES)
def test_bn_ode_net_runs_the_pinned_backend(auto_compiles_everything, codegen, backend):
    _t, _v, _n, timing = C.bn_ode_net(_NET, 0.0, 1.0, 11, 1e-8, 1e-8, codegen=codegen)
    assert timing["config"]["codegen"] == backend


@pytest.mark.parametrize(("codegen", "backend"), _CASES)
def test_multi_segment_replay_runs_the_pinned_backend(auto_compiles_everything, codegen, backend):
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
