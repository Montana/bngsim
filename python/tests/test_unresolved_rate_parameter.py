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
