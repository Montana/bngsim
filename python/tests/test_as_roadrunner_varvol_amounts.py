"""Issue #728: as_roadrunner's bare-id amount in a compartment whose size changes.

``as_roadrunner(["X"])`` is the amount of X, its concentration times the
compartment's live size. Three kinds of species were reported at the size the
model loaded with instead, so off by V_live/V_static, with ``[X]`` right:

- an amount species (``substanceOnly``) in a compartment an assignment rule
  sizes, under ODE and SSA: its column is rescaled to amount/V_live, and nothing
  recorded V_live for the amount, so a conserved 5 read 0.83 at t = 10;
- a concentration species a rate rule sets, in a compartment a rate rule sizes
  (SBML suite 01198: S1 = 0.2 for 0.28);
- a concentration species a rate rule or an assignment rule sets, in a
  compartment an assignment rule sizes.

Each is checked against its closed form. SBML suite cases 01198, 01223, 01000,
01208, 01462 and 01823 now match their results files in every amount column.
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
