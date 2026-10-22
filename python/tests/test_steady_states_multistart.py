"""Enumerating steady states by multistart Newton (issue #524).

``Simulator.steady_states()`` samples starting states on the conservation
manifold, runs KINSOL from each with no integration burst
(``steady_state_batch(states=..., newton_burst=False)``), and clusters the
roots, each with the stability certificate's verdict. The Gardner toggle at
``alpha_2 = 53.53`` has two attractors and a saddle between them (GH #78's
dose); the bivalent ligand / receptor model has two conservation laws.
"""

from __future__ import annotations

import logging
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._simulator import _sample_on_conservation_manifold

_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"
_SADDLE_DOSE = 53.526315789473685
# Newton on f(y) = 0 to full precision (test_steady_state_gh78.py).
_SADDLE = np.array([28.245215769071677, 1.8302588776274484])
_ATTRACTOR_R2 = np.array([0.0075962079094201856, 53.122784076898327])


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not _NETS_DIR.is_dir():
        pytest.skip(f"benchmark net corpus not available: {_NETS_DIR}")
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
    return str(path)


def _toggle(dose=_SADDLE_DOSE):
    model = bngsim.Model.from_net(_net("genetic_switch.net"))
    model.set_param("alpha_2", dose)
    return bngsim.Simulator(model, method="ode"), model


def _row(roots, near, rtol=1e-6):
    hits = [k for k, y in enumerate(roots.states) if np.allclose(y, near, rtol=rtol)]
    assert len(hits) == 1, (near, roots.states)
    return hits[0]


def test_the_toggle_has_two_attractors_and_a_saddle():
    sim, model = _toggle()
    roots = sim.steady_states(n_starts=200, seed=0)
    assert len(roots) == 3, roots
    saddle = _row(roots, _SADDLE)
    low = _row(roots, _ATTRACTOR_R2)
    high = ({0, 1, 2} - {saddle, low}).pop()
    assert roots.states[high][0] > 100 and roots.states[high][1] < 1
    assert roots.stability[saddle] == "unstable"
    assert roots.stability[low] == roots.stability[high] == "stable"
    # The certificate's spectrum says why: a positive eigenvalue at the saddle.
    assert max(np.asarray(roots.eigenvalues[saddle]).real) > 0
    assert max(np.asarray(roots.eigenvalues[low]).real) < 0
    # Every start is accounted for, and the model is as it was.
    assert roots.multiplicity.sum() + roots.n_failed == 200
    for k in range(3):
        assert np.sum(roots.basins == k) == roots.multiplicity[k]
    assert np.all(roots.residuals < 1e-9)
    np.testing.assert_array_equal(model.get_state(), [0.0, 0.0])


def test_a_monostable_dose_has_one_stable_root():
    sim, _ = _toggle(dose=5.0)
    roots = sim.steady_states(n_starts=60, seed=1)
    assert len(roots) == 1 and roots.stability == ["stable"]
    np.testing.assert_allclose(
        roots.states[0], sim.steady_state(method="newton").concentrations, rtol=1e-6
    )


def test_the_same_seed_gives_the_same_roots():
    sim, _ = _toggle()
    a = sim.steady_states(n_starts=40, seed=7)
    b = sim.steady_states(n_starts=40, seed=7)
    np.testing.assert_array_equal(a.basins, b.basins)
    np.testing.assert_array_equal(a.states, b.states)


def test_no_burst_keeps_the_saddle_the_two_tier_solver_turns_down():
    """From next to the saddle: burst-free Newton lands on it and says it is
    unstable; the default two-tier solver rejects it (#78) and returns an
    attractor."""
    sim, _ = _toggle()
    start = np.array([_SADDLE * 1.01])
    (free,) = sim.steady_state_batch(states=start, method="newton", newton_burst=False)
    assert free.converged and free.root_stability == "unstable"
    np.testing.assert_allclose(free.concentrations, _SADDLE, rtol=1e-8)
    (burst,) = sim.steady_state_batch(states=start, method="newton")
    assert burst.converged and not np.allclose(burst.concentrations, _SADDLE, rtol=1e-3)


def test_samples_lie_on_the_conservation_manifold():
    model = bngsim.Model.from_net(_net("blbr.net"))
    cl = model.conservation_laws
    L = np.asarray(cl["coefficients"], float).reshape(cl["n_laws"], model.n_species)
    starts = _sample_on_conservation_manifold(model, 50, np.random.default_rng(0), None, None)
    assert starts.shape == (50, model.n_species)
    assert np.all(starts >= 0)
    np.testing.assert_allclose(starts @ L.T, np.tile(cl["constants"], (50, 1)), rtol=1e-9)
    assert np.unique(starts[:, cl["independent"][0]]).size == 50  # actually sampled


def test_a_one_root_model_with_laws_gives_that_root():
    model = bngsim.Model.from_net(_net("blbr.net"))
    sim = bngsim.Simulator(model, method="ode")
    roots = sim.steady_states(n_starts=60, seed=0)
    assert len(roots) == 1 and roots.stability == ["stable"]
    np.testing.assert_allclose(
        roots.states[0], sim.steady_state(method="newton").concentrations, rtol=1e-5, atol=1e-9
    )


def test_a_masked_species_keeps_its_initial_value_in_every_start():
    model = bngsim.Model.from_net(_net("genetic_switch.net"))
    model.set_state([3.0, 0.0])
    starts = _sample_on_conservation_manifold(model, 20, np.random.default_rng(0), 100.0, ["R2()"])
    np.testing.assert_array_equal(starts[:, 0], 3.0)
    assert np.unique(starts[:, 1]).size == 20


def test_states_must_have_one_row_per_entry_ordered_like_the_species():
    sim, _ = _toggle()
    with pytest.raises(ValueError, match=r"shape \(n_entries, 2\)"):
        sim.steady_state_batch(states=np.zeros((3, 5)))
    with pytest.raises(ValueError, match="pair up row by row"):
        sim.steady_state_batch(params=[{}, {}], states=np.zeros((3, 2)))


def test_params_and_states_pair_up_row_by_row():
    """Row 0 runs dose 5 from beside its one root, row 1 the saddle dose from
    beside the saddle: each solve sees its own row's parameters and state."""
    low_dose_root = _toggle(dose=5.0)[0].steady_state(method="newton").concentrations
    sim, _ = _toggle()
    out = sim.steady_state_batch(
        params=[{"alpha_2": 5.0}, {"alpha_2": _SADDLE_DOSE}],
        states=np.array([np.asarray(low_dose_root) * 1.05, _SADDLE * 1.01]),
        method="newton",
        newton_burst=False,
    )
    assert out[0].root_stability == "stable"
    assert out[1].root_stability == "unstable"


# ── Review of PR #1165 ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "kinetic_proofreading.net",  # a law of 8 members
        "metapop_sir_100.net",  # one law of 300
        "egfr_net.net",  # 5 overlapping laws
        "Haugh2b.net",  # a row-reduced law with a negative total
        "nfkb.net",  # laws of one member at a total of 0
        "catalysis.net",  # $-fixed species
    ],
)
def test_samples_hold_the_laws_of_a_network(name):
    """Drawing each member below its law's total and redrawing while a
    dependent came out negative ran out of draws on 17 of the 53 tracked ODE
    nets, the first three here among them. It pinned at 0 a member of a law
    whose row-reduced total was not positive, and it drew the boundary species
    a solve holds fixed."""
    model = bngsim.Model.from_net(_net(name))
    y0 = np.asarray(model.get_state(), float)
    cl = model.conservation_laws
    L = np.asarray(cl["coefficients"], float).reshape(cl["n_laws"], model.n_species)
    fixed = np.asarray(model._core.species_is_fixed, bool)
    starts = _sample_on_conservation_manifold(model, 30, np.random.default_rng(0), None, None)
    totals = L @ y0
    np.testing.assert_allclose(
        starts @ L.T, np.tile(totals, (30, 1)), rtol=1e-9, atol=1e-9 * np.abs(totals).max()
    )
    assert np.all(starts >= 0)
    np.testing.assert_array_equal(starts[:, fixed], np.tile(y0[fixed], (30, 1)))
    # A law of non-negative coefficients at a total of 0 holds its members at 0;
    # every other species a start moves is drawn.
    zero_laws = np.all(L >= 0, axis=1) & (np.abs(totals) == 0)
    held_at_zero = (L[zero_laws] != 0).any(axis=0)
    for i in np.flatnonzero(~fixed):
        column = starts[:, i]
        if held_at_zero[i]:
            assert np.all(column == 0), model.species_names[i]
        else:
            assert np.unique(column).size == 30, model.species_names[i]


def test_samples_keep_the_totals_of_the_state_the_model_holds():
    """The solve recomputes a law's total from the state it starts in. The
    samples kept the totals the model was loaded with, so after a set_state
    they lay on another manifold than the one steady_state() solves on."""
    model = bngsim.Model.from_net(_net("blbr.net"))
    y = np.asarray(model.get_state(), float)
    y[0] *= 2.0
    model.set_state(y)
    cl = model.conservation_laws
    L = np.asarray(cl["coefficients"], float).reshape(cl["n_laws"], model.n_species)
    assert not np.allclose(L @ y, cl["constants"])
    starts = _sample_on_conservation_manifold(model, 20, np.random.default_rng(0), None, None)
    np.testing.assert_allclose(starts @ L.T, np.tile(L @ y, (20, 1)), rtol=1e-9)


@pytest.mark.parametrize(
    "name", ["Haugh2b.net", "nfkb.net", "wofsy_goldstein.net", "egfr_path.net"]
)
def test_a_one_root_network_comes_back_as_one_root(name):
    """KINSOL stops at the first iterate under tol, which along a slow mode sat
    1e-5 to 0.26 off the root in some species. A law's dependent species
    carries the law's round-off: egfr_path's EGF, 5e-8 in a law over 1.8e5.
    From 40 starts these four came back as 4, 5, 3 and 34 stable roots."""
    model = bngsim.Model.from_net(_net(name))
    sim = bngsim.Simulator(model, method="ode")
    roots = sim.steady_states(n_starts=40, seed=0)
    assert len(roots) == 1, roots
    ref = sim.steady_state(method="newton")
    assert ref.converged
    ref_y = np.asarray(ref.concentrations)
    np.testing.assert_allclose(roots.states[0], ref_y, rtol=1e-5, atol=1e-6 * np.abs(ref_y).max())


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


def test_a_refusal_in_a_burst_free_solve_is_raised(tmp_path):
    """A rate law that refuses (issue #589; mratio(2, 2.5, -30) does). The
    burst-free solve caught the refusal and returned a result holding no state,
    and find_steady_state read past its end: the interpreter died."""
    path = tmp_path / "refusing.net"
    path.write_text(_REFUSING_NET, encoding="utf-8")
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=False)
    with pytest.raises(bngsim.SimulationError, match="mratio.*not reliable"):
        sim.steady_state_batch(states=[[1.0]], method="newton", newton_burst=False)


_SINK_TOGGLE_NET = """begin parameters
    1 alpha_1     156.25
    2 alpha_2     53.526315789473685
    3 beta        2.5
    4 gamma       1.0
    5 k_deg       1.0
    6 k_s         1.0
end parameters
begin functions
    1 syn_R1() alpha_1/(1+(Obs_Tot_R2^beta))
    2 syn_R2() alpha_2/(1+(Obs_Tot_R1^gamma))
end functions
begin species
    1 R1() 0
    2 R2() 0
    3 S() 0
end species
begin reactions
    1 0 1 syn_R1 #_R1
    2 0 2 syn_R2 #_R2
    3 1 0 k_deg #_R3
    4 2 0 k_deg #_R4
    5 0 3 k_s #_R5
end reactions
begin groups
    1 Obs_Tot_R1           1
    2 Obs_Tot_R2           2
end groups
"""


def test_an_accumulator_sets_neither_the_scale_nor_the_log(tmp_path, caplog):
    """The toggle at the saddle dose with an accumulator S beside it. Its
    remedy is mask=~model.is_pure_sink(), but the scale probe ignored the mask,
    integrated S to max_time and drew every start below 1e7: one root of three.
    Unmasked, each failed start logged the same warning, 201 lines for 200."""
    path = tmp_path / "sink_toggle.net"
    path.write_text(_SINK_TOGGLE_NET, encoding="utf-8")
    model = bngsim.Model.from_net(str(path))
    sim = bngsim.Simulator(model, method="ode")
    roots = sim.steady_states(n_starts=200, seed=0, mask=~model.is_pure_sink())
    assert len(roots) == 3, roots
    _row(roots, np.append(_SADDLE, 0.0))
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        none = sim.steady_states(n_starts=50, seed=0)
    assert len(none) == 0 and none.n_failed == 50
    # The probe's own solve, and one line for the batch.
    assert caplog.text.count("structural reason") == 2
    assert caplog.text.count("in 50 of 50 batch entries") == 1
