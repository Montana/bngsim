"""An event delay that is NaN, infinite or negative is refused (issue #762).

SBML L3 §4.11.3 requires a delay to be a non-negative number. The ODE event loop
clamped a negative *expression* delay to 0 and never checked a constant one, and
NaN failed its ``delay > 0`` test, so every invalid delay fired at trigger time
exactly as a zero delay would, with no error. A constant delay is now refused
when the model is built; a delay expression is refused when it is evaluated, at
the fire. A negative within rounding of zero is still read as 0.
"""

from __future__ import annotations

import logging
import math

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

_REFUSAL = "an event delay must be a finite, non-negative number"


def _builder_model(**event_kw) -> bngsim.Model:
    b = ModelBuilder()
    x = b.add_species("X", 0.0)
    b.add_event("E", "time()>1", [(x, "1")], **event_kw)
    return bngsim.Model(_core=b.build())


def _x(model: bngsim.Model) -> list[float]:
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 3.0), n_points=7)
    names = list(r.species_names)
    i = names.index("X") if "X" in names else names.index("x")
    return np.asarray(r.species)[:, i].tolist()


_FIRES_AT_1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
_FIRES_AT_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


# ─── Controls ───────────────────────────────────────────────────────────────


def test_a_valid_delay_still_fires_late():
    assert _x(_builder_model(delay=1.0)) == _FIRES_AT_2
    assert _x(_builder_model(delay_expr="0.5 + 0.5")) == _FIRES_AT_2


def test_a_delay_that_is_zero_up_to_rounding_is_still_zero():
    """``1 - time()`` at the crossing is a hair below 0 after the root find."""
    assert _x(_builder_model(delay_expr="1 - time()")) == _FIRES_AT_1


# ─── Constant delays: refused when the model is built ───────────────────────


@pytest.mark.parametrize("delay", [-1.0, math.nan, math.inf, -math.inf])
def test_a_constant_invalid_delay_is_refused_at_build(delay):
    """Each of these fired at t = 1, as a zero delay, before the fix."""
    with pytest.raises(RuntimeError, match=_REFUSAL):
        _builder_model(delay=delay)


def test_a_negative_sbml_delay_is_refused_at_load():
    with pytest.raises(bngsim.ModelError, match=_REFUSAL):
        bngsim.Model.from_antimony_string("x=0; x'=0; E: at -1 after (time > 1): x = 1;")


# ─── Delay expressions: refused when evaluated, at the fire ─────────────────


@pytest.mark.parametrize(
    "make",
    [
        lambda: _builder_model(delay_expr="sqrt(-1)"),
        lambda: _builder_model(delay_expr="-1"),
        lambda: bngsim.Model.from_antimony_string("x=0; x'=0; E: at 0/0 after (time > 1): x = 1;"),
        lambda: bngsim.Model.from_antimony_string(
            "q=-1; x=0; x'=0; E: at sqrt(q) after (time > 1): x = 1;"
        ),
    ],
    ids=["builder-sqrt(-1)", "builder-(-1)", "sbml-0/0", "sbml-sqrt(q)"],
)
def test_an_invalid_delay_expression_is_refused_at_the_fire(make, caplog):
    model = make()
    with (
        caplog.at_level(logging.WARNING, logger="bngsim"),
        pytest.raises(bngsim.SimulationError, match=_REFUSAL) as info,
    ):
        _x(model)
    assert "'E'" in str(info.value) and "t=1" in str(info.value)
    # Not a Jacobian failure, so the jacobian="auto" FD retry must not run
    # (it would blame the Jacobian and fail the same way a second time).
    assert not [r for r in caplog.records if "GH#176" in r.getMessage()]
