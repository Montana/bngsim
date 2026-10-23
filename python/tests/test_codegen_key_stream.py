"""The structural codegen key's bytes come from the core's own walk (issue #820).

``compute_model_codegen_hash`` (#174) hashed ``codegen_data()``,
``codegen_jacobian_plan()`` and ``functional_jacobian_context()`` by walking
their Python dicts a value at a time (``_codegen._canon_update``): ~40 values a
reaction, 2 s at 58k reactions, on every ``Simulator`` construction that
compiles. The core now writes the same bytes (``codegen_key_stream``) from the
walk that builds those dicts, so a field the generator reads cannot be left out
of the key.

The stream is the Python walk's byte for byte, not merely another injective
encoding: a key is the same whichever of the two makes it. (A ``.so`` cached by
an earlier release is not kept by that: the code generator's own source is
part of the key, and this change edits it.) The walk is kept
(``_codegen_key_stream_reference``), and these tests hold the two equal on every
model they load, before and after the analytical Jacobian is attached, since
that fills the plan.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import bngsim
import pytest
from bngsim import _codegen as cg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _net_corpus  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_ROOT = _net_corpus.ROOT
_FCERI_GAMMA = _ROOT / "benchmarks" / "models" / "net" / "ode" / "fceri_gamma.net"

#: SBML the loader refuses, so with no key to compare.
_REFUSED = frozenset({"MODEL2205030001.xml", "MODEL2402030002.xml"})


def _tracked_models() -> list[Path]:
    """The tracked ``.net`` corpus and SBML, below a size at which the
    reference walk takes long."""
    sbml = [
        *sorted((_ROOT / "harness" / "sbml_test_suite" / "dsmts" / "cases").rglob("*.xml")),
        *sorted((_ROOT / "benchmarks" / "sbml_events").glob("*.xml")),
    ]
    found = [*_net_corpus.TRACKED, *(p for p in sbml if p.name not in _REFUSED)]
    return [p for p in found if p.stat().st_size < 1_000_000]


def _load(path: Path) -> bngsim.Model:
    if path.suffix == ".net":
        return bngsim.Model.from_net(str(path))
    return bngsim.Model.from_sbml(str(path))


def _assert_streams_equal(model) -> None:
    core = model._core if hasattr(model, "_core") else model
    stream = core.codegen_key_stream()
    reference = cg._codegen_key_stream_reference(core)
    if stream != reference:
        at = next(
            (i for i, (a, b) in enumerate(zip(stream, reference, strict=False)) if a != b),
            min(len(stream), len(reference)),
        )
        raise AssertionError(
            f"the core's key stream leaves the reference at byte {at} of "
            f"{len(stream)}/{len(reference)}:\n  core      {stream[max(0, at - 40) : at + 40]!r}"
            f"\n  reference {reference[max(0, at - 40) : at + 40]!r}"
        )


_MODELS = _tracked_models()


def test_the_corpus_is_there():
    assert len(_MODELS) > 150


@pytest.mark.parametrize("path", _MODELS, ids=lambda p: str(p.relative_to(_ROOT)))
def test_the_stream_is_the_walk_of_the_dicts(path):
    model = _load(path)
    _assert_streams_equal(model)
    # Attaching the analytical Jacobian fills the plan's scatter blocks.
    model.prepare_analytical_jacobian()
    _assert_streams_equal(model)


# Fields no tracked model above reaches.


def test_a_falling_factorial():
    model = bngsim.Model.from_antimony_string(
        "compartment C=2; species A in C=10; species B in C=0; J: 2A -> B; k*A*A/C; k=0.1"
    )
    assert any("ssa_falling_factorial" in r for r in model._core.codegen_data()["reactions"])
    _assert_streams_equal(model)


def test_a_live_volume_term():
    model = bngsim.Model.from_antimony_string(
        "compartment C1=1, C2=2; C1' = 0.1; species A in C1 = 10; species B in C2 = 1; "
        "J: A + B -> B; k*A*B; k=0.1"
    )
    assert any(r["ssa_live_volume_terms"] for r in model._core.codegen_data()["reactions"])
    _assert_streams_equal(model)


def test_a_nested_rate_of():
    model = bngsim.Model.from_antimony_string(
        "A' = 1; B' = rateOf(A); C' = rateOf(B); A=0; B=0; C=0"
    )
    assert model._core.codegen_data()["rateof_nesting"] > 0
    _assert_streams_equal(model)


def test_a_ring_of_rates():
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.enable_rateof()
    x = b.add_species("X", 0.0)
    y = b.add_species("Y", 0.0)
    b.add_function("fX", "0.5*rate_of__Y + 1")
    b.add_function("fY", "0.5*rate_of__X + 1")
    b.add_reaction([], [x], "functional", "fX")
    b.add_reaction([], [y], "functional", "fY")
    model = bngsim.Model(_core=b.build())
    assert model._core.codegen_data()["rateof_ring_refusal"]
    _assert_streams_equal(model)


def test_a_name_beyond_ascii_and_a_negative_zero():
    """A string's length is its UTF-8 bytes', and -0.0 is not 0.0."""
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    z = b.add_species("Zé", 1.0)
    w = b.add_species("W", 0.0)
    b.add_parameter("k", 1.0)
    b.add_reaction([z], [w], "elementary", "k", -0.0)
    model = bngsim.Model(_core=b.build())
    data = model._core.codegen_data()
    assert data["species"][0]["name"] == "Zé"
    assert str(data["reactions"][0]["stat_factor"]) == "-0.0"
    _assert_streams_equal(model)


def test_a_value_does_not_reach_the_stream():
    """A parameter's value and a species' declared amount are left out, by the
    core as by the reference."""
    model = bngsim.Model.from_antimony_string(
        "compartment C=5; species A in C; A = 10; species P in C = 0; J: A -> P; k*A; k=0.1"
    )
    before = model._core.codegen_key_stream()
    model.set_param("k", 0.7)
    params = {p["name"]: p for p in model._core.codegen_data()["parameters"]}
    assert params["k"]["value"] == 0.7
    assert model._core.codegen_key_stream() == before
    _assert_streams_equal(model)


def test_a_core_without_the_stream_gets_the_same_key():
    """The key walks the dicts on a core that cannot write the stream."""

    class _Without:
        def __init__(self, core):
            self._c = core

        def __getattr__(self, name):
            if name == "codegen_key_stream":
                raise AttributeError(name)
            return getattr(self._c, name)

    model = bngsim.Model.from_net(str(_ROOT / "tests" / "data" / "mm_tqssa_stiff.net"))
    assert cg.compute_model_codegen_hash(_Without(model._core)) == cg.compute_model_codegen_hash(
        model
    )


def test_the_key_of_a_large_model_is_cheap():
    """About 2 s at 58k reactions walking the dicts; about 0.12 s from the core,
    on Apple silicon. The bound has room for a slow runner and none for the
    walk."""
    model = bngsim.Model.from_net(str(_FCERI_GAMMA))
    assert model._core.n_reactions > 50_000
    cg.compute_model_codegen_hash(model)
    best = min(
        (lambda t0: (cg.compute_model_codegen_hash(model), time.perf_counter() - t0)[1])(
            time.perf_counter()
        )
        for _ in range(3)
    )
    assert best < 1.0
