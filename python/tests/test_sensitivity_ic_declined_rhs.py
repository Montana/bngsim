"""``sensitivity_ic`` on a model whose analytic sensitivity RHS codegen
declined is refused with the reason (issue #1124, item 7).

``C := 1 + P``, with P a species and S a concentration in C: S's dilution
term reads P's rate, ``rate_of__P``, which codegen does not differentiate, so
the analytic sensitivity RHS is declined and a parameter column runs on
CVODES' difference quotient. A column of an initial value has no parameter for
that quotient to perturb, and the run failed with "sensitivity_ic requires
codegen sensitivity RHS, but no codegen .so is loaded. Build the model with
codegen enabled": codegen was enabled, and had built the model. It is now
refused at construction, naming what was declined.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_READS_A_RATE = """
compartment C0 = 1; compartment C; C := 1 + P;
species P in C0 = 1; species S in C = 5;
J1: P -> ; 0.5*P*C0; J2: S -> ; 0.2*S*C;
"""

# Control: the size reads a parameter, so no rate, and the RHS builds.
_READS_NO_RATE = """
compartment C; C := 1 + q; q = 1;
species S in C = 5;
J2: S -> ; 0.2*S*C;
"""


@pytest.mark.filterwarnings("ignore")
def test_the_refusal_names_the_declined_rate_law():
    model = bngsim.Model.from_antimony_string(_READS_A_RATE)
    with pytest.raises(
        bngsim.SensitivityUnsupportedError,
        match=r"sensitivity_ic needs the analytic sensitivity RHS.*rate_of__P",
    ):
        bngsim.Simulator(model, method="ode", sensitivity_ic=["S"])


@pytest.mark.filterwarnings("ignore")
def test_a_parameter_column_still_runs_on_the_difference_quotient():
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(_READS_A_RATE),
        method="ode",
        sensitivity_params=["C0"],
    )
    assert not sim.has_analytic_sens_rhs
    sim.run(t_span=(0.0, 2.0), n_points=3)


def test_an_initial_value_column_where_the_rhs_builds():
    """The law is an amount rate, so d[S]/dt = -0.2·[S], and dS(t)/dS0 = e^(-0.2·t)."""
    result = bngsim.Simulator(
        bngsim.Model.from_antimony_string(_READS_NO_RATE), method="ode", sensitivity_ic=["S"]
    ).run(t_span=(0.0, 2.0), n_points=3, rtol=1e-10, atol=1e-12)
    t = np.asarray(result.time)
    col = np.asarray(result.sensitivities_ic)[:, 0, 0]
    np.testing.assert_allclose(col, np.exp(-0.2 * t), rtol=1e-6)


def test_the_refusal_is_asked_again_after_a_codegen_rebuild(monkeypatch):
    """The check ran at construction only. A rebuild of the sensitivity code
    (issue #708) that declines the analytic RHS has to refuse the same way,
    not reach the C++ backstop at run time."""
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(_READS_NO_RATE), method="ode", sensitivity_ic=["S"]
    )
    assert sim.has_analytic_sens_rhs
    # The rebuilt artifact declines the RHS: the question is whether it is asked.
    monkeypatch.setattr(bngsim.Simulator, "_codegen_provides_sens_rhs", lambda self: False)
    with pytest.raises(
        bngsim.SensitivityUnsupportedError,
        match=r"sensitivity_ic needs the analytic sensitivity RHS",
    ):
        sim._build_codegen_again()
