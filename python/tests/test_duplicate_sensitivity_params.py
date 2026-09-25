"""A parameter named twice in ``sensitivity_params`` gets the same column twice
(issue #759).

``setup_forward_sensitivities`` mapped each parameter to the FIRST column that
named it, and both initial-condition seeding paths looked columns up through
that map. The RHS forcing df/dp reached every column, but ``yS(0) = dx(0)/dp``
landed only in the first, so a repeated IC parameter's column came back all
zeros, and one that also sits in a rate law was off by exactly the missing seed.
A list built by concatenation names a parameter twice easily. Every column is
now seeded, so a repeat is the same column again.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest

_PURE = "species A = A0; A0 = 10; R1: A -> B; k1*A; k1 = 0.5; B = 0"
# A0 is also in the rate law, so dA/dA0 is the seed plus a forcing term.
_MIXED = "species A = A0; A0 = 10; R1: A -> B; k1*A*A0/10; k1 = 0.5; B = 0"

_NET = """
begin parameters
    1 A0 10
    2 k1 0.5
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 k1
end reactions
begin groups
    1 Atot 1
end groups
"""


def _a_sens(model: bngsim.Model, params: list[str]) -> np.ndarray:
    r = bngsim.Simulator(model, sensitivity_params=params).run((0.0, 2.0), 5)
    return np.asarray(r.sensitivities)[:, list(r.species_names).index("A"), :]


def _fd_a0(text: str, h: float = 1e-6) -> np.ndarray:
    def a(v: float) -> np.ndarray:
        m = bngsim.Model.from_antimony_string(text)
        m.set_param("A0", v)
        m.reset()
        r = bngsim.Simulator(m).run((0.0, 2.0), 5, rtol=1e-12, atol=1e-12)
        return np.asarray(r.species)[:, list(r.species_names).index("A")]

    return (a(10.0 + h) - a(10.0 - h)) / (2 * h)


@pytest.mark.parametrize("text", [_PURE, _MIXED], ids=["pure-ic", "ic-and-rate"])
def test_a_repeated_ic_parameter_gets_its_seed_in_every_column(text):
    """Before the fix the pure repeat was [0, 0, 0, 0, 0] and the mixed one
    was off by the missing seed (max error 1.0)."""
    sens = _a_sens(bngsim.Model.from_antimony_string(text), ["A0", "k1", "A0"])
    np.testing.assert_array_equal(sens[:, 2], sens[:, 0])
    np.testing.assert_allclose(sens[:, 0], _fd_a0(text), rtol=1e-5, atol=1e-7)


def test_a_list_of_only_the_repeat_is_seeded_twice():
    sens = _a_sens(bngsim.Model.from_antimony_string(_PURE), ["A0", "A0"])
    np.testing.assert_array_equal(sens[:, 1], sens[:, 0])
    assert sens[0, 1] == pytest.approx(1.0)


def test_a_repeated_ic_parameter_on_a_net_model_and_its_observable(tmp_path: Path):
    net = tmp_path / "ic.net"
    net.write_text(textwrap.dedent(_NET).strip() + "\n")
    r = bngsim.Simulator(
        bngsim.Model.from_net(str(net)), sensitivity_params=["A0", "k1", "A0"]
    ).run((0.0, 2.0), 3)
    species = np.asarray(r.sensitivities)[:, 0, :]
    obs = np.asarray(r.sensitivities_observables)[:, 0, :]
    expected = np.exp(-0.5 * np.array([0.0, 1.0, 2.0]))  # dA/dA0 = exp(-k1 t)
    for cols in (species, obs):
        np.testing.assert_array_equal(cols[:, 2], cols[:, 0])
        np.testing.assert_allclose(cols[:, 2], expected, rtol=1e-6)
