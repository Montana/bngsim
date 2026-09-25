"""Tests for JAX AD Jacobian (Session 22).

Tests the JAX RHS generator, expression translator, discontinuity
screening, and Jacobian correctness vs finite differences.

These tests require JAX: ``pip install jax jaxlib``
If JAX is not installed, all tests are skipped.
"""

import os

import numpy as np
import pytest

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh, which copies
# tests to a temp dir (breaking __file__-relative resolution).
DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)
ODE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "models", "net", "ode")

# Skip entire module if JAX is not available
try:
    import jax  # noqa: F401
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


class TestJaxAvailability:
    """Test JAX availability detection."""

    def test_jax_available(self):
        from bngsim._jax_rhs import jax_available

        assert jax_available() is True


class TestDiscontinuityScreening:
    """Test screening for discontinuous functions."""

    def test_clean_model(self):
        from bngsim._jax_rhs import screen_for_discontinuities

        path = os.path.join(DATA, "simple_decay.net")
        problems = screen_for_discontinuities(path)
        assert problems == []

    def test_clean_functional_model(self):
        from bngsim._jax_rhs import screen_for_discontinuities

        path = os.path.join(ODE_DIR, "CaOscillate_functional.net")
        if not os.path.exists(path):
            pytest.skip("CaOscillate_functional.net not found")
        problems = screen_for_discontinuities(path)
        assert problems == []


class TestJaxRhsGeneration:
    """Test JAX RHS function generation."""

    def test_simple_decay_rhs(self):
        """JAX RHS matches analytical: A -> B, k=0.1, A0=100."""
        from bngsim._jax_rhs import generate_jax_rhs

        path = os.path.join(DATA, "simple_decay.net")
        rhs = generate_jax_rhs(path)

        assert rhs.n_species == 2
        assert rhs.n_params == 1

        # Initial state
        y = jnp.array([100.0, 0.0])
        params = jnp.array([0.1])  # k1 = 0.1

        dydt = rhs(y, 0.0, params)
        dydt_np = np.asarray(dydt)

        # dA/dt = -k*A = -0.1*100 = -10
        # dB/dt = +k*A = +0.1*100 = +10
        assert abs(dydt_np[0] - (-10.0)) < 1e-10
        assert abs(dydt_np[1] - 10.0) < 1e-10

    def test_reversible_rhs(self):
        """JAX RHS for A + B <-> C."""
        from bngsim._jax_rhs import generate_jax_rhs

        path = os.path.join(DATA, "two_species_reversible.net")
        rhs = generate_jax_rhs(path)

        assert rhs.n_species == 3
        assert rhs.n_params == 2


class TestJaxJacobian:
    """Test JAX AD Jacobian computation."""

    def test_simple_decay_jacobian(self):
        """Jacobian of simple decay: J = [[-k, 0], [k, 0]]."""
        from bngsim._jax_rhs import generate_jax_jacobian

        path = os.path.join(DATA, "simple_decay.net")
        jac_fn = generate_jax_jacobian(path)

        y = jnp.array([100.0, 0.0])
        params = jnp.array([0.1])

        J = jac_fn(y, 0.0, params)
        J_np = np.asarray(J)

        # dA/dt = -k*A → ∂(dA/dt)/∂A = -k = -0.1
        # dB/dt = +k*A → ∂(dB/dt)/∂A = +k = +0.1
        assert abs(J_np[0, 0] - (-0.1)) < 1e-10
        assert abs(J_np[1, 0] - 0.1) < 1e-10
        # No dependence on B
        assert abs(J_np[0, 1]) < 1e-10
        assert abs(J_np[1, 1]) < 1e-10

    def test_reversible_jacobian_vs_fd(self):
        """Compare JAX Jacobian vs finite differences for A+B<->C."""
        from bngsim._jax_rhs import generate_jax_jacobian

        path = os.path.join(DATA, "two_species_reversible.net")
        jac_fn = generate_jax_jacobian(path)
        rhs = jac_fn.rhs

        y = jnp.array([50.0, 30.0, 20.0])
        params = jnp.array([0.001, 0.1])

        J_jax = np.asarray(jac_fn(y, 0.0, params))

        # Finite difference Jacobian
        n = len(y)
        J_fd = np.zeros((n, n))
        h = 1e-7
        f0 = np.asarray(rhs(y, 0.0, params))
        for j in range(n):
            y_pert = np.array(y)
            y_pert[j] += h
            f1 = np.asarray(rhs(jnp.array(y_pert), 0.0, params))
            J_fd[:, j] = (f1 - f0) / h

        # Compare
        max_err = np.max(np.abs(J_jax - J_fd))
        assert max_err < 1e-5, f"JAX vs FD max error: {max_err}"

    def test_functional_rate_jacobian(self):
        """JAX Jacobian for CaOscillate (Functional rates)."""
        from bngsim._jax_rhs import generate_jax_jacobian

        path = os.path.join(ODE_DIR, "CaOscillate_functional.net")
        if not os.path.exists(path):
            pytest.skip("CaOscillate_functional.net not found")

        import bngsim

        m = bngsim.Model.from_net(path)
        jac_fn = generate_jax_jacobian(m)
        n_sp = m.n_species

        # The model's own parameter values, derived ones evaluated (the .net
        # reader this used to take them from left every expression at 1.0).
        core = m._core
        params = jnp.array([core.get_param(n) for n in core.param_names], dtype=jnp.float64)
        y = jnp.ones(n_sp, dtype=jnp.float64) * 1000.0

        J_jax = np.asarray(jac_fn(y, 0.0, params))
        assert J_jax.shape == (n_sp, n_sp)
        assert np.any(J_jax != 0.0)  # a coupled system
        # ...and it is the engine's Jacobian, not merely a nonzero one.
        J_eng = m.jacobian(np.asarray(y))
        np.testing.assert_allclose(
            J_jax, np.asarray(J_eng), rtol=1e-8, atol=1e-10 * np.abs(J_eng).max()
        )


class TestPrepareJaxJacobian:
    """Test the prepare_jax_jacobian entry point."""

    def test_prepare_returns_callable(self):
        from bngsim._jax_rhs import prepare_jax_jacobian

        path = os.path.join(DATA, "simple_decay.net")
        eval_fn, n_sp = prepare_jax_jacobian(path)

        assert n_sp == 2
        assert callable(eval_fn)

    def test_prepare_column_major_output(self):
        """Verify output is column-major flat array."""
        from bngsim._jax_rhs import prepare_jax_jacobian

        path = os.path.join(DATA, "simple_decay.net")
        eval_fn, n_sp = prepare_jax_jacobian(path)

        y = np.array([100.0, 0.0])
        params = np.array([0.1])

        flat_jac = eval_fn(y, 0.0, params)
        assert flat_jac.shape == (4,)  # 2x2 flattened

        # Column-major: col 0 = [J[0,0], J[1,0]], col 1 = [J[0,1], J[1,1]]
        # J[0,0] = -k = -0.1, J[1,0] = +k = +0.1
        assert abs(flat_jac[0] - (-0.1)) < 1e-10  # col 0, row 0
        assert abs(flat_jac[1] - 0.1) < 1e-10  # col 0, row 1
        assert abs(flat_jac[2]) < 1e-10  # col 1, row 0
        assert abs(flat_jac[3]) < 1e-10  # col 1, row 1

    def test_saturated_hill_jacobian_is_finite_and_the_solve_completes(self, tmp_path):
        """Issue #838: AD of x**n used to produce inf/inf in the Hill tangent."""
        import bngsim
        from bngsim._jax_rhs import generate_jax_jacobian

        net = tmp_path / "saturated_hill.net"
        net.write_text(
            """begin parameters
 1 k 2.0
 2 K 1.0
 3 n 999.898
 4 ks 1.0
end parameters
begin species
 1 A() 5
 2 B() 0
end species
begin reactions
 1 0 2 f #_R1
 2 1 0 ks #_R2
end reactions
begin groups
 1 Atot 1
end groups
begin functions
 1 f() k/(1+(Atot/K)^n)
end functions
""",
            encoding="utf-8",
        )

        params = np.array([2.0, 1.0, 999.898, 1.0])
        jac = np.asarray(generate_jax_jacobian(str(net))(np.array([5.0, 1.0]), 0.0, params))
        assert np.isfinite(jac).all()
        np.testing.assert_allclose(jac, [[-1.0, 0.0], [0.0, 0.0]], atol=1e-14)

        model = bngsim.Model.from_net(str(net))
        with pytest.warns(UserWarning, match="jacobian='jax'"):
            sim = bngsim.Simulator(model, method="ode", jacobian="jax", net_path=str(net))
        result = sim.run(t_span=(0.0, 2.0), n_points=3)
        np.testing.assert_allclose(result.species[-1, 0], 5.0 * np.exp(-2.0), rtol=1e-6)


class TestExpressionTranslation:
    """Test .net expression -> JAX translation."""

    def test_translate_simple(self):
        from bngsim._jax_rhs import _translate_expr_jax

        expr = "k3/(K4+G)"
        param_names = {"k3": 0, "K4": 1}
        obs_names = {"G": 0}

        result = _translate_expr_jax(expr, param_names, obs_names, set(), [])
        assert "params[0]" in result
        assert "params[1]" in result
        assert "obs[0]" in result

    def test_translate_time(self):
        from bngsim._jax_rhs import _translate_expr_jax

        expr = "time()+1"
        result = _translate_expr_jax(expr, {}, {}, set(), [])
        assert "t+1" in result or "t +1" in result

    def test_translate_exponentiation(self):
        from bngsim._jax_rhs import _translate_expr_jax

        expr = "k*(x^2)"
        result = _translate_expr_jax(expr, {"k": 0, "x": 1}, {}, set(), [])
        assert "**" in result

    def test_translates_saturating_hill_denominator_stably(self):
        from bngsim._jax_rhs import _translate_expr_jax

        result = _translate_expr_jax(
            "k/(1+(Atot/K)^n)", {"k": 0, "K": 1, "n": 2}, {"Atot": 0}, set(), []
        )
        assert "__bngsim_inv_hill_power__" in result

    @pytest.mark.parametrize(
        ("x", "n"),
        [(1e-300, 0.5), (1e-8, 2.0), (0.5, 4.0), (1.0, 50.0), (2.0, -3.0), (-2.0, 3.0)],
    )
    def test_stable_hill_tangent_matches_the_direct_form(self, x, n):
        """The stable form keeps the direct form's derivative where that one is
        finite, in both AD modes. sigmoid(-n*log(x)) has the same value, but its
        tangent s*(1 - s) cancels to 0 once x**n < ~1e-16 (x = 1e-300, n = 0.5:
        0 against -5e149), and an unmasked n leaked 0**n = inf from the branch
        not taken into the reverse-mode dH/dn for a negative n (NaN)."""
        from bngsim._jax_rhs import _jax_inv_hill_power, jax_available

        assert jax_available()  # enables x64

        def direct(x, n):
            return 1.0 / (1.0 + jnp.power(x, n))

        args = (jnp.float64(x), jnp.float64(n))
        np.testing.assert_allclose(_jax_inv_hill_power(*args), direct(*args), rtol=1e-14)
        for argnum in (0, 1) if x > 0 else (0,):
            want = jax.grad(direct, argnums=argnum)(*args)
            for mode in (jax.grad, jax.jacfwd):
                got = mode(_jax_inv_hill_power, argnums=argnum)(*args)
                np.testing.assert_allclose(got, want, rtol=1e-12)
