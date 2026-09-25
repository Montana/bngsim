"""GH #720: the derived-parameter chain rule is guarded at a zero power base.

``_direct_derived_partials`` printed ``sp.diff`` with bare ``sp.ccode``, which
bypassed the emitters' zero-base guards: d(E^n)/dn came out as
``pow(E,n)*log(E)`` and d(E^n)/dE as ``n*pow(E,n)/E``, both NaN at E = 0. Every
sensitivity run on such a model was refused (CV_FIRST_SRHSFUNC_ERR), and an
output sensitivity through the numeric twin came back NaN with no warning.

kr = E^n/(E^n + h^n) + kb with E = 0 is kb, so A = A0 exp(-kb t). At E = 0,
d(kr)/dn = 0 for every n > 0, and d(kr)/dE = 0 for n > 1 and 1/h for n = 1.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_NET = """begin parameters
    1 E 0
    2 n {n}
    3 h 1.5
    4 kb 0.5
    5 kr (E^n)/(E^n+h^n)+kb
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 {rate} #_R1
end reactions
begin groups
    1 Ao 1
end groups
begin functions
    1 g() kr*Ao
end functions
"""
H, KB = 1.5, 0.5


def _load(tmp_path, n, rate="kr"):
    path = tmp_path / f"zero_base_{n}_{rate}.net"
    path.write_text(_NET.format(n=n, rate=rate), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize(("n", "dkr_dE"), [(2, 0.0), (1, 1.0 / H)])
def test_state_sensitivities_at_a_zero_base(tmp_path, n, dkr_dE):
    m = _load(tmp_path, n)
    r = bngsim.Simulator(m, method="ode", sensitivity_params=["n", "E"]).run(
        t_span=(0.0, 4.0), n_points=3
    )
    t = np.asarray(r.time)
    a = 10.0 * np.exp(-KB * t)
    np.testing.assert_allclose(r.species[:, 0], a, rtol=1e-6)
    # dA/dp = -t A d(kr)/dp for a constant rate.
    np.testing.assert_allclose(r.sensitivities[:, 0, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(r.sensitivities[:, 0, 1], -t * a * dkr_dE, rtol=1e-5, atol=1e-9)


def test_output_sensitivity_at_a_zero_base_is_finite(tmp_path):
    # The rate is kb, so A does not depend on n or E; g = kr*A does only
    # through kr, whose partials at E = 0 (n = 2) are both zero.
    m = _load(tmp_path, 2, rate="kb")
    r = bngsim.Simulator(m, method="ode", sensitivity_params=["n", "E", "kb"]).run(
        t_span=(0.0, 4.0), n_points=3
    )
    out = np.asarray(r.output_sensitivities(["g"]))[:, 0, :]
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, :2], 0.0, atol=1e-9)
    t = np.asarray(r.time)
    a = 10.0 * np.exp(-KB * t)
    # dg/dkb = A + kr dA/dkb = A (1 - kb t).
    np.testing.assert_allclose(out[:, 2], a * (1.0 - KB * t), rtol=1e-5, atol=1e-8)


def test_an_infinite_threshold_partial_is_refused_not_zeroed():
    """d(E^n)/dE at E = 0 with n = 0.5 is infinite, so the event's crossing time
    has no derivative in E. The guards leave it non-finite; dropping it read as
    dt*/dp = 0 in the event-threshold detector, which asks for no warnings, and
    the run returned dX/dE = 0 where the one-sided difference grows without
    bound. Kept, it reaches the jump, which refuses it."""
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("kin", 2.0)
    b.add_parameter("kout", 0.5)
    b.add_parameter("E", 0.0)
    b.add_parameter("n", 0.5)
    b.add_parameter("T", 1.0, expression="1 + E^n", is_expression=True)
    x = b.add_species("X", 0.0)
    on = b.add_species("on", 0.0)
    b.add_reaction([on], [on, x], "elementary", "kin")
    b.add_reaction([x], [], "elementary", "kout")
    b.add_observable("Xobs", [(x, 1.0)])
    b.add_event("onset", "time() >= T", [(on, "1.0")])
    sim = bngsim.Simulator(bngsim.Model(_core=b.build()), method="ode", sensitivity_params=["E"])
    with pytest.raises(bngsim.SimulationError, match="non-finite"):
        sim.run(t_span=(0.0, 5.0), n_points=6)


@pytest.mark.parametrize(("n", "dkr_dE"), [(0.5, np.inf), (1.0, 1.0 / H), (2.0, 0.0)])
def test_the_numeric_twin_takes_the_limit_the_c_twin_takes(n, dkr_dE):
    """The guarded partials divide through by the base, so at E = 0 they read
    pow(0, -k): C takes that to inf and 1/inf to 0. The numeric twin evaluates
    the same way, so d(E^n/(E^n+h^n))/dE is 0 at n = 2 (sympy's zoo arithmetic
    made it NaN) and inf at n = 0.5, kept rather than dropped as a zero."""
    from bngsim._codegen import _derived_expr_partials_numeric

    names = ["E", "n", "h"]
    walked = _derived_expr_partials_numeric(
        "E^n/(E^n+h^n)", set(names), {p: i for i, p in enumerate(names)}, [0.0, n, H], {}
    )
    assert walked.get("E", 0.0) == pytest.approx(dkr_dE)
    assert walked.get("n", 0.0) == 0.0
