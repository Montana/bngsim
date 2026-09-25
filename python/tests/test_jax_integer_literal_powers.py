"""GH #837 — literal-only integer powers must not overflow JAX's int64 parsing."""

from __future__ import annotations

import numpy as np
import pytest
from bngsim._jax_rhs import _jax_numeric_literals_as_floats, _translate_expr_jax


def test_integer_literal_powers_translate_to_float_arithmetic():
    translated = _translate_expr_jax(
        "k*((1500^6)+Atot)/(1500^6)", {"k": 0}, {"Atot": 0}, set(), []
    )
    assert translated == "params[0]*((1500.0**6.0)+obs[0])/(1500.0**6.0)"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1e-3^2", "1e-3**2.0"),
        ("1.25+2", "1.25+2.0"),
    ],
)
def test_integer_rewrite_preserves_other_numeric_forms_and_strings(expr, expected):
    assert _translate_expr_jax(expr, {}, {}, set(), []) == expected


def test_integer_rewrite_does_not_touch_digits_inside_quoted_strings():
    assert _jax_numeric_literals_as_floats("tfun('dose1500.dat', x)+2") == (
        "tfun('dose1500.dat', x)+2.0"
    )


NET = """begin parameters
    1 k 0.4
end parameters
begin species
    1 A() 3
    2 B() 0
end species
begin reactions
    1 1 2 f #_R1
end reactions
begin groups
    1 Atot 1
end groups
begin functions
    1 f() k*((1500^6)+Atot)/(1500^6)
end functions
"""


def test_reported_net_runs_through_diffrax_and_jax_jacobian(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("diffrax")
    import bngsim
    from bngsim._diffrax_solver import run_diffrax

    net = tmp_path / "integer_power.net"
    net.write_text(NET)

    diffrax_result = run_diffrax(
        str(net), {"k": 0.4}, t_end=2.0, n_points=3, rtol=1e-10, atol=1e-12
    )
    expected_a = 3.0 * np.exp(-0.4 * np.asarray(diffrax_result["time"]))
    np.testing.assert_allclose(diffrax_result["species"][:, 0], expected_a, rtol=1e-8)

    model = bngsim.Model.from_net(str(net))
    result = bngsim.Simulator(model, method="ode", jacobian="jax", net_path=str(net)).run(
        t_span=(0.0, 2.0), n_points=3, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(result.species[:, 0], expected_a, rtol=1e-8)
