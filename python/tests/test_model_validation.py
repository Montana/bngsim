# bngsim/python/tests/test_model_validation.py
# Session 65: P9 — Model validation on construction
#
# Tests that ModelBuilder.build() rejects malformed models with clear errors.

import pytest
from bngsim._bngsim_core import ModelBuilder


def _builder():
    """Create a fresh ModelBuilder."""
    return ModelBuilder()


class TestDuplicateNames:
    """Duplicate species/parameter names are rejected."""

    def test_duplicate_species(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("A", 20.0)  # duplicate
        b.add_reaction([0], [1], "elementary", "k")
        with pytest.raises(RuntimeError, match="duplicate species"):
            b.build()

    def test_duplicate_parameter(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_parameter("k", 0.2)  # duplicate
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_reaction([0], [1], "elementary", "k")
        with pytest.raises(RuntimeError, match="duplicate parameter"):
            b.build()


class TestSpeciesIndexRange:
    """Reaction species indices must be in range."""

    def test_bad_reactant_index(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_reaction([5], [0], "elementary", "k")
        with pytest.raises(RuntimeError, match="reactant species index"):
            b.build()

    def test_bad_product_index(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_reaction([0], [3], "elementary", "k")
        with pytest.raises(RuntimeError, match="product species index"):
            b.build()


class TestObservableIndexRange:
    """Observable group species indices must be in range."""

    def test_bad_observable_species(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_observable("bad", [(5, 1.0)])
        b.add_reaction([0], [1], "elementary", "k")
        with pytest.raises(RuntimeError, match="observable.*out of range"):
            b.build()


class TestRateLawResolution:
    """Rate law parameter/function refs must resolve."""

    def test_unknown_elementary_param(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_reaction([0], [1], "elementary", "k_missing")
        with pytest.raises(RuntimeError, match="unknown parameter"):
            b.build()

    def test_unknown_functional_func(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_reaction([0], [1], "functional", "ghost")
        with pytest.raises(RuntimeError, match="unknown function"):
            b.build()

    @pytest.mark.parametrize("rate_type", ["elementary", "functional"])
    def test_empty_rate_name(self, rate_type):
        # Issue #863: "" passes the unknown-name check; issue #589's
        # build-time check is what refuses it.
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_reaction([0], [1], rate_type, "")
        with pytest.raises(RuntimeError, match="reaction 0 has no resolvable rate parameter"):
            b.build()


def _a_to_b(rate_type, rate_law, ic_ref=None, apply_species_factor=True, a_seed=100.0):
    """A -> B at k = 2 from A(0) = 100, so dA/dt = -200 at the initial state."""
    from bngsim._model import Model

    b = _builder()
    b.add_parameter("k", 2.0, "2.0", False)
    b.add_parameter("A0", 100.0, "100", False)
    b.add_species("A()", a_seed, False)
    b.add_species("B()", 0.0, False)
    if ic_ref is not None:
        b.add_species_param_ref(*ic_ref)
    b.add_observable("Atot", [(0, 1.0)])
    b.add_reaction([0], [1], rate_type, rate_law, 1.0, apply_species_factor)
    return Model(_core=b.build())


class TestFunctionalNamingParameter:
    """A functional reaction whose rate names only a parameter fires (issue #863).

    The mirror of an elementary reaction naming a function, which build()
    reclassifies as functional. Before #863 it resolved to no rate: dA/dt was
    0, and since #589 the build refused it.
    """

    @pytest.mark.parametrize("rate_type", ["elementary", "functional"])
    def test_fires_at_the_parameter_rate(self, rate_type):
        m = _a_to_b(rate_type, "k")
        assert m.rhs([100.0, 0.0])[0] == pytest.approx(-200.0)

    def test_follows_set_param(self):
        m = _a_to_b("functional", "k")
        m.set_param("k", 3.0)
        assert m.rhs([100.0, 0.0])[0] == pytest.approx(-300.0)

    def test_generated_rhs_agrees(self):
        import bngsim
        import numpy as np

        finals = []
        for codegen in (False, True):
            sim = bngsim.Simulator(_a_to_b("functional", "k"), method="ode", codegen=codegen)
            finals.append(np.asarray(sim.run(t_span=(0, 0.1), n_points=2).species)[-1, 0])
        assert finals == pytest.approx([100.0 * np.exp(-0.2)] * 2, rel=1e-6)

    @pytest.mark.parametrize("rate_type", ["elementary", "functional"])
    def test_jacobian_keeps_the_per_species_volume_divide(self, rate_type):
        # Cross-compartment accumulation divides each row by its species'
        # volume, dA/dt = -k*A/2 and dB/dt = k*A/5. The elementary analytical
        # Jacobian had no such divide and returned -k and k.
        import numpy as np
        from bngsim._model import Model

        b = _builder()
        b.add_parameter("k", 2.0, "2.0", False)
        b.add_species("A()", 100.0, False, 2.0)
        b.add_species("B()", 0.0, False, 5.0)
        b.add_observable("Atot", [(0, 1.0)])
        b.add_reaction([0], [1], rate_type, "k", 1.0, True, 1.0, True)
        m = Model(_core=b.build())
        y = np.array([100.0, 3.0])
        np.testing.assert_allclose(m.rhs(y), [-100.0, 40.0])
        np.testing.assert_allclose(m.jacobian(y), [[-1.0, 0.0], [0.4, 0.0]], atol=1e-6)

    @pytest.mark.parametrize("rate_type", ["elementary", "functional"])
    def test_parameter_rate_without_species_factor_refused(self, rate_type):
        # The interpreted kernel read this as dA/dt = -k, while the generated
        # RHS and the analytical Jacobian applied the reactant factor, -k*A.
        with pytest.raises(
            RuntimeError, match="takes its rate from parameter 'k' with apply_species_factor=false"
        ):
            _a_to_b(rate_type, "k", apply_species_factor=False)


class TestSpeciesICReference:
    """A species IC reference must name a declared parameter and species (issue #863)."""

    def test_declared_reference_is_kept(self):
        m = _a_to_b("elementary", "k", ic_ref=(0, "A0"), a_seed=0.0)
        assert list(m._core.species_ic_param_refs) == [(0, 1)]
        # The IC comes from A0, not the number add_species was given.
        assert m.rhs(m._core.get_state())[0] == pytest.approx(-200.0)

    def test_undeclared_parameter(self):
        # Was dropped: the species kept add_species' number and no reference.
        with pytest.raises(
            RuntimeError,
            match="species 'A\\(\\)' takes its initial value from unknown parameter 'nope'",
        ):
            _a_to_b("elementary", "k", ic_ref=(0, "nope"))

    @pytest.mark.parametrize("idx", [-1, 2])
    def test_species_index_out_of_range(self, idx):
        # Was an out-of-bounds write in build().
        with pytest.raises(RuntimeError, match="names species index .* out of range"):
            _a_to_b("elementary", "k", ic_ref=(idx, "A0"))


class TestParameterExpressionResolution:
    """A parameter expression must compile, the way a function body must (issue #602).

    `build()` used to swallow a failed parameter compile with `continue`, leaving
    the slot holding whatever partial number the front end had scraped off the
    front of the text. `run_network` refuses such a file outright ("Could not
    find parameter ... Exiting."), and a function body that will not compile
    already threw a few lines further down — only the parameter was silent.
    """

    def _model(self, expr):
        b = _builder()
        b.add_parameter("kbase", 0.5)
        b.add_parameter("k", 0.0, expr, True)
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_reaction([0], [1], "elementary", "k")
        return b

    def test_unknown_symbol_is_refused(self):
        with pytest.raises(RuntimeError, match="failed to compile parameter 'k'"):
            self._model("2*kbse").build()

    def test_numeric_prefix_is_not_kept_as_the_value(self):
        """`2*kbse` used to survive as 2.0 — a plausible rate, not an obvious zero."""
        with pytest.raises(RuntimeError, match=r"2\*kbse"):
            self._model("2*kbse").build()

    def test_syntax_error_is_refused(self):
        with pytest.raises(RuntimeError, match="failed to compile parameter 'k'"):
            self._model("kbase +").build()

    def test_valid_expression_still_compiles(self):
        """The refusal must not catch an expression that resolves."""
        m = self._model("2*kbase").build()
        assert m.get_param("k") == 1.0


class TestValidModelOK:
    """Well-formed models build and simulate."""

    def test_simple_model(self):
        from bngsim._bngsim_core import (
            CvodeSimulator,
            TimeSpec,
        )

        b = _builder()
        b.add_parameter("k", 0.1)
        a = b.add_species("A", 100.0)
        bb = b.add_species("B", 0.0)
        b.add_observable("A_tot", [(a, 1.0)])
        b.add_reaction([a], [bb], "elementary", "k")
        model = b.build()
        assert model.n_species == 2
        assert model.n_reactions == 1

        sim = CvodeSimulator(model)
        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = 10.0
        ts.n_points = 11
        result = sim.run(ts)
        assert result.n_times == 11
