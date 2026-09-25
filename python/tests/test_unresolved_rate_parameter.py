"""GH #589: a reaction whose rate parameter never resolved is refused at build.

compute_rxn_rate answers an unresolved rate-parameter index with a propensity
of 0.0, an ordinary value, so the reaction was dead for the whole run with
nothing reporting it. It cannot throw there (it runs inside CVODE's C
right-hand-side callback), so ModelBuilder.build() now checks every reaction's
resolved index once. An empty rate name is one way to get there: validation
checks a name only when one is given.
"""

from __future__ import annotations

import pytest
from bngsim._bngsim_core import ModelBuilder


def _builder() -> ModelBuilder:
    b = ModelBuilder()
    b.add_parameter("k", 1.0, "", False)
    b.add_species("A", 1.0, False, 1.0)
    b.add_species("B", 0.0, False, 1.0)
    return b


def test_a_reaction_with_no_rate_parameter_is_refused():
    b = _builder()
    b.add_reaction([0], [1], "elementary", "", 1.0, True)
    with pytest.raises(RuntimeError, match="no resolvable rate parameter"):
        b.build()


def test_a_resolved_reaction_still_builds():
    b = _builder()
    b.add_reaction([0], [1], "elementary", "k", 1.0, True)
    assert b.build() is not None


# ── A refusal inside a SUNDIALS callback surfaces as itself ──────────────────
#
# A rate law may refuse rather than answer: mratio does for arguments its
# continued fraction cannot be trusted with, and the rate kernel now does for an
# unresolved rate parameter. Those are C++ exceptions raised inside CVODE's and
# KINSOL's callbacks, which are C, so every callback that evaluates the model
# now parks the exception and returns a failure code, and the solve rethrows it
# afterwards. mratio(2, 2.5, -30) is a refusal (see test_mratio_trust_region).

_REFUSING_NET = """begin parameters
    1 a 2.0
    2 b 2.5
    3 zz -30.0
end parameters
begin functions
    1 law() mratio(a,b,zz)
end functions
begin species
    1 A() 1
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 Atot 1
end groups
"""


@pytest.mark.parametrize(
    "solve",
    [
        pytest.param(lambda s: s.run(t_span=(0, 1), n_points=3), id="ode"),
        pytest.param(lambda s: s.steady_state(), id="steady_state"),
    ],
)
@pytest.mark.parametrize("jacobian", ["auto", "fd"])
def test_a_refusal_in_a_solver_callback_is_raised_as_itself(tmp_path, solve, jacobian):
    import bngsim

    path = tmp_path / "refusing.net"
    path.write_text(_REFUSING_NET, encoding="utf-8")
    m = bngsim.Model.from_net(str(path))
    sim = bngsim.Simulator(m, method="ode", codegen=False, jacobian=jacobian)
    with pytest.raises(bngsim.SimulationError, match="mratio.*not reliable"):
        solve(sim)
