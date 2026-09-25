"""An SBML event delay or priority that reads a parameter follows set_param.

Both are evaluated when the trigger fires (SBML L3v2 §4.11.3/§4.11.4). GH #558
stopped the loader folding one that reads a parameter the model writes, but it
still folded every other parameter to its value in the file, and those are the
parameters a caller changes with ``set_param``. So ``d = 1; at d after (...)``
fired at ``t_trigger + 1`` whatever ``d`` was set to, and priorities read from
parameters kept their load-time order, with no warning. Only a literal is folded
now; anything reading a parameter is evaluated at the fire against its live
value, so a write and a reload of the edited file agree.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_DELAY = "d={d}; x=0; x'=0; E: at d after (time > 1): x = 1;"
_PRIORITY = (
    "p1={p1}; p2={p2}; x=0; x'=0; "
    "E1: at (time > 1), priority=p1: x = 1; "
    "E2: at (time > 1), priority=p2: x = 2;"
)


def _x(model: bngsim.Model) -> list[float]:
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 3.0), n_points=7)
    return np.asarray(r.species)[:, list(r.species_names).index("x")].tolist()


@pytest.mark.parametrize("d", [0.5, 1.5])
def test_set_param_on_a_delay_parameter_moves_the_fire_time(d):
    """Before, both values fired at t = 2, the delay in the file."""
    written = bngsim.Model.from_antimony_string(_DELAY.format(d=1))
    written.set_param("d", d)
    written.reset()
    reloaded = bngsim.Model.from_antimony_string(_DELAY.format(d=d))
    got = _x(written)
    assert got == _x(reloaded)
    t = np.linspace(0.0, 3.0, 7)
    assert got == [1.0 if ti > 1 + d else 0.0 for ti in t]


def test_set_param_on_priority_parameters_changes_the_order():
    """E1 and E2 fire together; the lower priority runs last and its write
    stays. Swapping the priorities used to change nothing."""
    m = bngsim.Model.from_antimony_string(_PRIORITY.format(p1=2, p2=1))
    assert _x(m)[-1] == 2.0
    m.set_param("p1", 1)
    m.set_param("p2", 2)
    m.reset()
    assert _x(m)[-1] == 1.0
    assert _x(bngsim.Model.from_antimony_string(_PRIORITY.format(p1=1, p2=2)))[-1] == 1.0


def test_a_literal_delay_still_folds():
    """Nothing to keep live: a plain number stays a constant delay."""
    m = bngsim.Model.from_antimony_string("x=0; x'=0; E: at 0.5 after (time > 1): x = 1;")
    assert _x(m) == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
