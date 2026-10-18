"""Issue #728: as_roadrunner's bare-id amount in a compartment whose size changes.

``as_roadrunner(["X"])`` is the amount of X, its concentration times the
compartment's live size. Three kinds of species were reported at the size the
model loaded with instead, so off by V_live/V_static, with ``[X]`` right:

- an amount species (``substanceOnly``) in a compartment an assignment rule
  sizes, under ODE and SSA: its column is rescaled to amount/V_live, and nothing
  recorded V_live for the amount, so a conserved 5 read 0.83 at t = 10;
- a concentration species a rate rule or an assignment rule sets, in a
  compartment a rate rule sizes (SBML suite 01198: S1 = 0.2 for 0.28);
- a concentration species a rate rule or an assignment rule sets, in a
  compartment an assignment rule sizes.

Each is checked against its closed form. The 20 SBML suite cases the issue
names (01000, 01121, 01198, 01208, 01223, 01462, 01478, 01498, 01504 to 01514
and 01823) now match their results files in every amount column.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

T = np.array([0.0, 5.0, 10.0])

CASES = {
    # V = 0.5 + 0.2t, [S] = 0.4t (the shape of 01198)
    "rate-rule target, rate-rule compartment": (
        "compartment c = 0.5; c' = 0.2; species S in c; S = 0; S' = 0.4;",
        "S",
        0.4 * T * (0.5 + 0.2 * T),
        0.4 * T,
    ),
    # [S] := 3 in V = 0.5 + 0.2t
    "assignment-rule target, rate-rule compartment": (
        "compartment c = 0.5; c' = 0.2; species S in c; S := 3;",
        "S",
        3.0 * (0.5 + 0.2 * T),
        np.full_like(T, 3.0),
    ),
    # a conserved amount
    "amount species, assignment-rule compartment": (
        "compartment c; c := 2 + time; substanceOnly species H in c = 5;",
        "H",
        np.full_like(T, 5.0),
        5.0 / (2.0 + T),
    ),
    # [S] := 3 in V = 2 + t
    "assignment-rule target, assignment-rule compartment": (
        "compartment c; c := 2 + time; species S in c; S := 3;",
        "S",
        3.0 * (2.0 + T),
        np.full_like(T, 3.0),
    ),
    # [S] = 1 + t in V = 2 + t
    "rate-rule target, assignment-rule compartment": (
        "compartment c; c := 2 + time; species S in c; S = 1; S' = 1;",
        "S",
        (1.0 + T) * (2.0 + T),
        1.0 + T,
    ),
}


def _report(text: str, sid: str, method: str = "ode") -> np.ndarray:
    model = bngsim.Model.from_antimony_string(text)
    kw = {"seed": 1} if method == "ssa" else {"rtol": 1e-10, "atol": 1e-12}
    run = bngsim.Simulator(model, method=method).run(sample_times=list(T), **kw)
    return np.asarray(run.as_roadrunner([sid, f"[{sid}]"])).reshape(len(T), 2)


@pytest.mark.parametrize("case", list(CASES))
def test_the_bare_id_is_the_amount_at_the_live_size(case):
    text, sid, amount, conc = CASES[case]
    got = _report(text, sid)
    np.testing.assert_allclose(got[:, 0], amount, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(got[:, 1], conc, rtol=1e-7, atol=1e-12)


def test_an_amount_species_in_an_assignment_rule_compartment_under_ssa():
    """The concentration pass rescales the SSA column too: the amount is the
    count, 50, at every row."""
    got = _report("compartment c; c := 2 + time; substanceOnly species H in c = 50;", "H", "ssa")
    np.testing.assert_allclose(got[:, 0], 50.0, rtol=1e-12)
    np.testing.assert_allclose(got[:, 1], 50.0 / (2.0 + T), rtol=1e-12)


def test_an_amount_species_keeps_its_amount_where_the_size_is_zero():
    """At t = 2 the size, 2 - t, is 0. The concentration pass does not divide by
    it there and leaves the row at amount/V_static, so the amount is that row
    times V_static: the conserved 5, as at every other row, and not 0."""
    model = bngsim.Model.from_antimony_string(
        "compartment c; c := 2 - time; substanceOnly species H in c = 5;"
    )
    run = bngsim.Simulator(model).run(sample_times=[0.0, 1.0, 2.0, 3.0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(run.as_roadrunner(["H"]))[:, 0], 5.0, rtol=1e-12)


@pytest.mark.parametrize("route", ["run_batch", "make_subset_model"])
def test_a_copy_of_the_model_reports_the_amount_at_the_live_size(route):
    """A run_batch row runs on a clone of the model, and coupling builds a subset
    model from it. Each reports from the map it carries, not the loader's."""
    text, sid, amount, _conc = CASES["rate-rule target, assignment-rule compartment"]
    model = bngsim.Model.from_antimony_string(text)
    kw = {"rtol": 1e-10, "atol": 1e-12}
    if route == "run_batch":
        rows = bngsim.Simulator(model).run_batch(t_span=(0.0, 10.0), n_points=3, params=[{}], **kw)
        run = rows[0]
    else:
        run = bngsim.Simulator(bngsim.make_subset_model(model)).run(sample_times=list(T), **kw)
    np.testing.assert_allclose(np.asarray(run.as_roadrunner([sid]))[:, 0], amount, rtol=1e-7)
