"""``sensitivity_ic`` on a model whose analytic sensitivity RHS codegen
declined is refused with the reason (issue #1124, item 7).

A column of an initial value has no parameter for CVODES' difference quotient
to perturb, and the run failed with "sensitivity_ic requires codegen
sensitivity RHS, but no codegen .so is loaded. Build the model with codegen
enabled": codegen was enabled, and had built the model. It is now refused at
construction, naming what was declined.

The model here was ``C := 1 + P``, S a concentration in C whose dilution term
reads P's rate, with P leaving for X in a compartment of another size: a rate
the derivations did not write out, so the RHS declined. Since issue #1266 they
divide such a reaction's rate by each species' own volume, the RHS builds, and
that model's initial-value column is checked against its closed form below.
The refusal is checked on a law the analytic path still declines, a clamp
whose upper bound is a parameter.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_READS_A_RATE = """
compartment C0 = 1; compartment C2 = 3; compartment C; C := 1 + P;
species P in C0 = 1; species X in C2 = 0; species S in C = 5;
J1: P -> X; 0.5*P*C0; J2: S -> ; 0.2*S*C;
"""

# A law the analytic sensitivity RHS still declines, with nothing in it that
# crosses: a clamp whose upper bound is a parameter is not known to bend (#1227),
# and its value is a*1, so A' = -a·A and dA/da = -10·t·e^(-a·t).
_DECLINED_NET = """begin parameters
    1 a   0.5
    2 big 1e300
end parameters
begin species
    1 A() 10
end species
begin functions
    1 f() a*clamp(1,0*Aobs,big)
end functions
begin reactions
    1 1 0 f
end reactions
begin groups
    1 Aobs 1
end groups
"""


def _declined(tmp_path):
    path = tmp_path / "declined.net"
    path.write_text(_DECLINED_NET)
    return bngsim.Model.from_net(str(path))


# Control: the size reads a parameter, so no rate, and the RHS builds.
_READS_NO_RATE = """
compartment C; C := 1 + q; q = 1;
species S in C = 5;
J2: S -> ; 0.2*S*C;
"""


@pytest.mark.filterwarnings("ignore")
def test_the_refusal_names_the_declined_rate_law(tmp_path):
    with pytest.raises(
        bngsim.SensitivityUnsupportedError,
        match=r"sensitivity_ic needs the analytic sensitivity RHS.*clamp",
    ):
        bngsim.Simulator(_declined(tmp_path), method="ode", sensitivity_ic=["A()"])


@pytest.mark.filterwarnings("ignore")
def test_a_parameter_column_still_runs_on_the_difference_quotient(tmp_path):
    sim = bngsim.Simulator(_declined(tmp_path), method="ode", sensitivity_params=["a"])
    assert not sim.has_analytic_sens_rhs
    result = sim.run(sample_times=[0.0, 1.0, 2.0], rtol=1e-10, atol=1e-12)
    t = np.asarray(result.time)
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[:, 0, 0], -10 * t * np.exp(-0.5 * t), rtol=1e-5, atol=1e-8
    )


def test_an_initial_value_column_through_a_rate_across_compartments():
    """Issue #1266: P's rate is written out, divided by P's volume C0, so the
    RHS builds. S's amount S·C decays at 0.2 and C = 1 + P with P = e^(-t/2),
    so S(t) = S0·2·e^(-0.2·t)/(1 + e^(-t/2)) and dS(t)/dS0 is that over S0."""
    result = bngsim.Simulator(
        bngsim.Model.from_antimony_string(_READS_A_RATE), method="ode", sensitivity_ic=["S"]
    ).run(t_span=(0.0, 2.0), n_points=5, rtol=1e-10, atol=1e-12)
    t = np.asarray(result.time)
    col = np.asarray(result.sensitivities_ic)[:, list(result.species_names).index("S"), 0]
    np.testing.assert_allclose(col, 2 * np.exp(-0.2 * t) / (1 + np.exp(-0.5 * t)), rtol=1e-7)


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
