"""A concentration carried far below 0 is not clamped to 0 to read a rate law.

Where the right-hand side comes back non-finite, GH #135 evaluates it again on
the state with every concentration below 0 set to 0. That is the boundary value
for a species the solver's own error put a hair below 0. Over the rr corpus the
states it rescues sit 80 to 1.3e4 of the run's atol below 0. It also answered
for states nowhere near 0.

BIOMD0000000338 (Wajima 2009, BioModels, CC0) has XIIa and K activating each
other through ``v*S*E/(k + E)`` with k = 1, so each has a pole at -1, and XIIa =
K = 0 is an exact, unstable solution. The dense LU's rounding can tip XIIa below
0, and the model's own laws then carry it down to the pole. There the RHS is
non-finite, and clamped it read 0 for XIIa and for K, which held both at -1. On
main (d1b27d5d, arm64) the dense run over t 0..0.08 returned XIIa = -1 and K =
-0.997 with no error, 1e9 atols below 0. Before #1302 the x86-64 CI's run over
0..2 did the same, and its agreement with KLU (XIIa = 0, the exact solution)
passed the path-agreement check.

A clamp now leaves alone a component more than 1e6 atols below 0. Where the run
itself has gone there (the state the solver's step started from is as far below
0), the run is refused by name. Where only a predictor or Newton iterate has
(BIOMD0000000404's closed-form attempt reached -1.6e147; a Hill law's predictor
overshoots its saturated phase to 1e9 atols), the RHS is CVODE's recoverable
failure, as any non-finite RHS is, and a run that cannot then get past the band
fails as CVODE fails, naming the species below 0 and the law.

The oracle is the property itself: a run either raises, naming the
concentration, or returns no concentration carried 1e6 atols below 0 through a
law that has no value there. The constant-sink model below is exact: X = 1 - t,
and Y = 2/3 at t = 1, after which sqrt(X) has no value. Main returned X(2) = -1,
Y(2) = 2/3 with no error.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest

BIOMD338 = Path(__file__).resolve().parents[2] / "tests" / "data" / "BIOMD0000000338.xml"

# X is emptied by a constant sink, so X = 1 - t goes below 0 at t = 1, where
# sqrt(X) stops having a value. No event: this is run()'s warm path.
SQRT_SINK = """
species X = 1; species Y = 0
R1: X -> ; 1
R2: -> Y; sqrt(X)
"""

PATHS = {
    "fd": {"jacobian": "fd"},
    "auto": {},
    "analytical": {"jacobian": "analytical"},
    "compiled": {"codegen": True},
    "klu fd": {"jacobian": "fd", "force_sparse_linear_solver": True},
    "klu auto": {"force_sparse_linear_solver": True},
}


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _run_338(**kw):
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml(str(BIOMD338)), method="ode", force_dense_linear_solver=True, **kw
    )
    return sim.run(t_span=(0.0, 0.08), n_points=81, rtol=1e-6, atol=1e-9, max_steps=100000)


@pytest.mark.parametrize("jacobian", ["fd", "auto", "analytical"])
def test_338_carried_to_its_pole_raises_rather_than_returning_it(jacobian):
    # Where rounding tips XIIa below 0 depends on the platform's last bits: on
    # arm64 this window takes it to the pole, on x86-64 XIIa bursts positive.
    # Either way no concentration comes back 1e6 atols below 0.
    try:
        result = _run_338(jacobian=jacobian)
    except bngsim.SimulationError as exc:
        msg = str(exc)
        assert "carried concentration 'XIIa'" in msg or "carried concentration 'K'" in msg, msg
        assert "times its atol below 0" in msg, msg
        return
    species = np.asarray(result.species, dtype=float)
    lowest = species.min(axis=0)
    names = list(result.species_names)
    worst = int(np.argmin(lowest))
    assert lowest[worst] >= -1e6 * 1e-9, f"{names[worst]} returned at {lowest[worst]}"


def test_338_on_klu_keeps_the_exact_equilibrium_at_0():
    # The default run (KLU: 54 species) never tips XIIa: XIIa = K = 0 is the exact
    # solution, and nothing here touches a run whose RHS stays finite.
    sim = bngsim.Simulator(bngsim.Model.from_sbml(str(BIOMD338)), method="ode")
    result = sim.run(t_span=(0.0, 2.0), n_points=101, rtol=1e-6, atol=1e-9)
    species = np.asarray(result.species, dtype=float)
    names = list(result.species_names)
    assert np.all(species[:, names.index("XIIa")] == 0.0)
    assert np.all(species[:, names.index("K")] == 0.0)


@pytest.mark.parametrize("kw", PATHS.values(), ids=PATHS.keys())
def test_a_law_read_past_its_domain_fails_naming_it(kw):
    # X is held at the band's edge, 1e6 atols below 0, where the clamp ends; past
    # it sqrt(X) has no value and the step cannot be taken. The run fails as
    # CVODE fails, its message naming X below 0 and the law.
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(SQRT_SINK), "ode", **kw)
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run(t_span=(0.0, 2.0), n_points=21, rtol=1e-6, atol=1e-9)
    msg = str(info.value)
    assert "Species below zero there: X = -" in msg, msg
    assert "X^(1/2)" in msg, msg


def test_up_to_its_domain_edge_the_answer_is_exact():
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(SQRT_SINK), "ode")
    result = sim.run(t_span=(0.0, 1.0), n_points=11, rtol=1e-10, atol=1e-12)
    species = np.asarray(result.species, dtype=float)
    names = list(result.species_names)
    t = np.asarray(result.time)
    np.testing.assert_allclose(species[:, names.index("X")], 1.0 - t, atol=1e-8)
    y_exact = (2.0 / 3.0) * (1.0 - (1.0 - t) ** 1.5)
    np.testing.assert_allclose(species[:, names.index("Y")], y_exact, atol=1e-6)


# Two corpus models whose runs met a far-negative state the clamp used to read
# (BioModels, CC0; each at its rr-parity job's own window and tolerances).
BIOMD404 = BIOMD338.parent / "BIOMD0000000404.xml"  # Bray 1993, chemotaxis
BIOMD810 = BIOMD338.parent / "BIOMD0000000810.xml"  # Ganguli 2018, tumour immunology


def _run_404(**kw):
    sim = bngsim.Simulator(bngsim.Model.from_sbml(str(BIOMD404)), method="ode", **kw)
    result = sim.run(t_span=(0.0, 10.0), n_points=1001, rtol=1e-6, atol=1e-9, max_steps=100000)
    return sim, np.asarray(result.species, dtype=float)


def test_404_a_failed_closed_form_attempt_still_retries_on_the_difference_quotient():
    # The closed-form attempt collapses its step to 1e-156 with MYpYpYp at
    # -1.6e147 and is refused there; the refusal is a CVODE failure, so the
    # GH #176 retry runs, and its answer is the difference quotient's.
    sim, species = _run_404()
    assert sim.jacobian_strategy == "fd"
    _, fd = _run_404(jacobian="fd")
    assert np.array_equal(species, fd)
    assert np.all(np.isfinite(species)) and species.min() > -1e-6


def _run_810(**kw):
    sim = bngsim.Simulator(bngsim.Model.from_sbml(str(BIOMD810)), method="ode", **kw)
    return sim.run(t_span=(0.0, 50.0), n_points=101, rtol=1e-5, atol=1e-9, max_steps=100000)


def test_810_on_the_difference_quotient_is_refused_not_returned_as_3e26():
    # Main returned a cell of 3.3e26 here, where the default run (which agrees
    # with RoadRunner) has 3.4: the run went through T-helper cells at -5.6e9,
    # read as 0 by the clamp. Refused there now, naming the species.
    with pytest.raises(bngsim.SimulationError, match="carried concentration '"):
        _run_810(jacobian="fd")


def test_810_by_default_is_untouched():
    species = np.asarray(_run_810().species, dtype=float)
    assert np.all(np.isfinite(species))
    assert species.min() >= 0.0 and species.max() < 1e10


# The band is judged where the step started, not at the state CVODE asks about:
# a step that starts near 0 is clamped however far its predictor or Newton
# iterate overshoots, as on main (review round 1 refused these on the predicted
# state).
HILL = "species S = 100; species P = 0\nR1: S -> P; 50*S^2.5/(0.01^2.5 + S^2.5)\n"


def test_a_predictor_overshooting_from_a_step_near_0_is_clamped_as_before():
    # Predicted to S = -9.7 (1e9 atols) on the straight line of its saturated
    # phase; S itself never leaves 0. bngsim's defaults (rtol = atol = 1e-8).
    from scipy.integrate import solve_ivp

    def g(s):
        s = max(s, 0.0)
        return 50.0 * s**2.5 / (0.01**2.5 + s**2.5)

    t = np.linspace(0.0, 20.0, 101)
    ref = solve_ivp(
        lambda _t, y: [-g(y[0]), g(y[0])],
        (0.0, 20.0),
        [100.0, 0.0],
        method="LSODA",
        t_eval=t,
        rtol=1e-12,
        atol=1e-14,
    ).y.T
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(HILL), "ode")
    result = sim.run(t_span=(0.0, 20.0), n_points=101)
    species = np.asarray(result.species, dtype=float)
    names = list(result.species_names)
    assert np.abs(species[:, names.index("S")] - ref[:, 0]).max() < 1e-6
    assert np.abs(species[:, names.index("P")] - ref[:, 1]).max() < 1e-6


def _extinction(x0, a, rtol, atol, **kw):
    # X' = -k X^a empties X at t = 2 exactly; X = 0 from there on.
    k = x0 ** (1.0 - a) / ((1.0 - a) * 2.0)
    text = f"species X = {x0}; species Y = 0\nR1: X -> ; {k!r}*X^{a}\nR2: -> Y; X^{a}\n"
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), "ode", **kw)
    result = sim.run(t_span=(0.0, 5.0), n_points=51, rtol=rtol, atol=atol, max_steps=100000)
    t = np.asarray(result.time)
    exact = np.maximum(0.0, x0 ** (1.0 - a) - (1.0 - a) * k * t) ** (1.0 / (1.0 - a))
    x = np.asarray(result.species, dtype=float)[:, list(result.species_names).index("X")]
    return np.abs(x - exact).max() / x0, x.min()


def test_an_extinction_that_stops_rtols_of_its_start_below_0_is_clamped():
    # Where X fell from 1 it stops 8.2 rtols of that below 0, 8e7 atols at atol
    # 1e-12: inside the run's tolerance, which an atol-only band would refuse.
    err, lowest = _extinction(1.0, 0.05, 1e-5, 1e-12)
    assert err < 10 * 1e-5, err
    assert lowest < -1e6 * 1e-12  # the depth this guards


@pytest.mark.parametrize("kw", [{}, {"force_sparse_linear_solver": True}], ids=["auto", "klu"])
def test_an_extinction_main_froze_far_below_0_is_right_or_refused(kw):
    # Main froze X at -0.125 from X0 = 10 (1.25e4 rtols off) with no error: the
    # clamp read it at 0. Either the run is now right, or it raises.
    try:
        err, lowest = _extinction(10.0, 0.05, 1e-6, 1e-9, **kw)
    except bngsim.SimulationError:
        return
    assert err < 10 * 1e-6, err


def test_the_refusal_names_the_species_whose_law_has_no_value():
    # A goes below 0 through a constant sink, a finite law: no cause, not named.
    # B is read by sqrt(B) and is the one the run is refused for.
    text = (
        "species A = 0.1; species B = 0.5; species Y = 0\n"
        "R1: A -> ; 1\nR2: B -> ; 1\nR3: -> Y; sqrt(B)\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), "ode", jacobian="fd")
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run(t_span=(0.0, 1.0), n_points=11, rtol=1e-6, atol=1e-9)
    msg = str(info.value)
    assert "carried concentration 'B'" in msg, msg
    assert "'A' to" not in msg, msg
