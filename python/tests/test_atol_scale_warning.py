"""Issue #1361: an ODE run says when the default atol is above species that matter.

The default ``atol`` is the scalar ``1e-8``, BNG2.pl's, for species held as
molecule counts. A species that stays below it all run has no error control but
the tolerance: the issue's binding model, species near 1e-9 M, returned A =
-8.7e-10 for 4.1e-10 at t = 5000, and BIOMD0000000002 (ligand at 1e-5 M,
complexes at 3e-10 to 7e-9 M) errors of 1e-2, both with no word. ``run()`` now
raises :class:`bngsim.AtolScaleWarning`, naming the species, where the caller
gave no atol and a species stays below it while reaching at least 1e-6 of the
run's largest species. The default itself is unchanged.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

NET = """begin parameters
 1 kf 1e6
 2 kr 1e-3
end parameters
begin species
 1 A() {a}
 2 B() {b}
 3 C() 0
end species
begin reactions
 1 1,2 3 kf
 2 3 1,2 kr
end reactions
"""


def _sim(tmp_path, a=1e-9, b=2e-9):
    path = tmp_path / "binding.net"
    path.write_text(NET.format(a=a, b=b))
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")


def _warned(sim, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(t_span=(0, 5000), n_points=51, **kw)
    return result, [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]


def test_a_molar_model_at_the_default_atol_is_warned(tmp_path):
    """The issue's model: at the default it ends at a negative A, and says so."""
    result, caught = _warned(_sim(tmp_path))
    assert np.asarray(result.species)[-1, 0] < 0.0
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "atol=1e-08" in message and "'A()'" in message and "atol='auto'" in message


def test_the_remedy_it_names_is_right_and_quiet(tmp_path):
    """Control: atol='auto' gives the reference to 1e-5 of the peak, with no warning."""
    reference = _sim(tmp_path).run(t_span=(0, 5000), n_points=51, rtol=1e-10, atol=1e-16)
    result, caught = _warned(_sim(tmp_path), atol="auto")
    assert caught == []
    ref = np.asarray(reference.species)
    err = np.abs(np.asarray(result.species) - ref).max(axis=0) / np.abs(ref).max(axis=0)
    assert err.max() < 1e-5


@pytest.mark.parametrize("how", ["run", "set_tolerances"])
def test_an_atol_the_caller_chose_is_not_second_guessed(tmp_path, how):
    sim = _sim(tmp_path)
    if how == "run":
        _, caught = _warned(sim, atol=1e-8)
    else:
        sim.set_tolerances(atol=1e-8)
        _, caught = _warned(sim)
    assert caught == []


def test_a_count_model_is_not_warned(tmp_path):
    """Control: species of a thousand molecules, with C starting at 0."""
    _, caught = _warned(_sim(tmp_path, a=1000, b=2000))
    assert caught == []


def test_a_species_at_the_noise_of_a_count_model_is_not_warned(tmp_path):
    """Control: a species that stays below atol, but at under 1e-6 of the
    largest species, is rounding in a model of counts, not a scale."""
    path = tmp_path / "noise.net"
    path.write_text(
        "begin parameters\n 1 k 1\n 2 kn 1e-15\nend parameters\n"
        "begin species\n 1 A() 1000\n 2 N() 0\nend species\n"
        "begin reactions\n 1 1 0 k\n 2 0 2 kn\nend reactions\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    result, caught = _warned(sim)
    assert 0.0 < np.abs(np.asarray(result.species)[:, 1]).max() < 1e-8
    assert caught == []


def test_a_mixed_scale_biomodel_is_warned():
    """BIOMD0000000002: L at 1e-5 is above atol; complexes at 3e-10 to 7e-9
    are below it and are part of the model. A whole-run test would pass it."""
    pytest.importorskip("libsbml")
    sim = bngsim.Simulator(bngsim.Model.from_sbml(str(DATA / "BIOMD0000000002.xml")), method="ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(t_span=(0, 500), n_points=51)
    hits = [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]
    assert len(hits) == 1
    assert "'BL'" in str(hits[0].message)


def test_ssa_is_not_warned():
    """SSA has no atol: a concentration of 1e-9 in a compartment of 1e10 is
    ten molecules, run exactly, and below 1e-8 all run."""
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "compartment c = 1e10;\nspecies X in c; X = 1e-9;\nk = 0.1;\nR: X -> ; k*X;\n"
    )
    sim = bngsim.Simulator(model, method="ssa")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(t_span=(0, 1), n_points=3, seed=1)
    assert 0.0 < np.abs(np.asarray(result.species)).max() < 1e-8
    assert not [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]
