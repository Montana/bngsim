"""An exponent that calls a function is one operand (issue #657)."""

import numpy as np
import pytest
from bngsim import Model, Simulator
from bngsim._codegen import _expr_to_c, _extract_exp_right, _replace_power_op


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("A^sqrt(n)", "pow(A, sqrt(n))"),
        ("A^-sqrt(n)", "pow(A, -sqrt(n))"),
        ("A^sqrt(n^2)", "pow(A, sqrt(pow(n, 2)))"),
        ("A^sqrt(n)^p", "pow(A, pow(sqrt(n), p))"),
        ("A^max(n, sqrt(p))", "pow(A, max(n, sqrt(p)))"),
        ("sqrt(A)^sqrt(n)", "pow(sqrt(A), sqrt(n))"),
        ("A^n+sqrt(p)", "pow(A, n)+sqrt(p)"),
    ],
)
def test_function_call_is_complete_exponent(expr, expected):
    assert _replace_power_op(expr) == expected


def test_exponent_scanner_consumes_balanced_call():
    assert _extract_exp_right("A^sqrt(max(n, p)) + 1", 2) == ("sqrt(max(n, p))", 17)


def test_identifier_mapping_keeps_call_inside_power():
    assert _expr_to_c("Atot^sqrt(n)", ["n"], [], ["Atot"], []) == "pow(obs[0], sqrt(p[0]))"


def test_compiled_ode_matches_interpreter(tmp_path):
    net = tmp_path / "call_exponent.net"
    net.write_text(
        """begin parameters
    1 k 0.1
    2 n 2.0
end parameters
begin functions
    1 law() k*Atot^sqrt(n)
end functions
begin species
    1 A() 5.0
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 Atot 1
end groups
"""
    )
    times = (0.0, 2.0)
    interpreted = Simulator(Model.from_net(str(net)), method="ode", codegen=False).run(
        t_span=times, n_points=3
    )
    compiled = Simulator(Model.from_net(str(net)), method="ode", codegen=True).run(
        t_span=times, n_points=3
    )
    np.testing.assert_allclose(compiled.species, interpreted.species, rtol=1e-7, atol=1e-9)

    sensitivity = Simulator(Model.from_net(str(net)), method="ode", sensitivity_params=["n"]).run(
        t_span=times, n_points=3
    )
    legs = []
    for offset in (1e-4, -1e-4):
        perturbed = Model.from_net(str(net))
        perturbed.set_param("n", 2.0 + offset)
        legs.append(
            np.asarray(Simulator(perturbed, method="ode").run(t_span=times, n_points=3).species)
        )
    np.testing.assert_allclose(
        np.asarray(sensitivity.sensitivities)[:, :, 0],
        (legs[0] - legs[1]) / 2e-4,
        rtol=3e-3,
        atol=1e-7,
    )
