"""The JAX RHS and the diffrax solver are built from the built model (#803 step 4).

Until step 4 both re-read the ``.net`` file with codegen's private parser, and a
corpus sweep against the engine found the JAX RHS wrong on 249 of 767 networks
and crashing on 5 more. The biggest cause: a synthesis reaction's null reactant
(index 0 in the file) multiplied the rate by ``y[-1]``, the *last* species. It
also inherited the #784 family, and a model with no observables crashed on an
empty weight matrix. ``jacobian="jax"`` only feeds CVODE a Jacobian, so there
those defects cost Newton convergence; ``run_diffrax`` integrates the JAX RHS
itself, so there they were wrong trajectories.

Every expectation here is a closed form or the engine's own interpreted RHS,
``Model.rhs`` (ExprTk; no code shared with the JAX translator), and, for the
Jacobian, the engine's analytical Jacobian.
"""

from __future__ import annotations

import textwrap
import warnings

import bngsim
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import bngsim._jax_rhs as jr  # noqa: E402

jr.jax_available()  # enables float64


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text).lstrip())
    return p


def _params(m):
    core = m._core
    return np.array([core.get_param(n) for n in core.param_names], dtype=np.float64)


def _positive_state(m, seed=7):
    """A strictly positive state, so no rate term vanishes into a zero species."""
    y0 = np.abs(np.asarray(m._core.get_initial_state(), dtype=np.float64))
    rng = np.random.default_rng(seed)
    return y0 * (1 + 0.3 * rng.random(y0.size)) + 0.5 + rng.random(y0.size)


def _assert_rhs_matches_engine(m, t=0.0):
    rhs = jr.generate_jax_rhs(m)
    y = _positive_state(m)
    f_jax = np.asarray(rhs(jnp.asarray(y), t, jnp.asarray(_params(m))))
    f_eng = np.asarray(m.rhs(y, t))
    np.testing.assert_allclose(f_jax, f_eng, rtol=1e-12, atol=1e-12 * np.abs(f_eng).max())
    return rhs, y


SYNTH = """\
begin parameters
    1 ksyn 2.0
    2 kdeg 0.5
end parameters
begin species
    1 A() 0
    2 B() 3
end species
begin reactions
    1 0 1 ksyn #_R1
    2 1 0 kdeg #_R2
end reactions
begin groups
    1 Atot 1
end groups
"""


class TestTheRhsIsTheModels:
    def test_a_synthesis_rate_is_not_scaled_by_the_last_species(self, tmp_path):
        """0 -> A at ksyn, A -> 0 at kdeg, B inert: dA/dt = ksyn - kdeg A exactly.
        The .net path read the null reactant as index 0 and multiplied ksyn by
        y[-1], which is B here."""
        m = bngsim.Model.from_net(_write(tmp_path, "s.net", SYNTH))
        rhs, y = _assert_rhs_matches_engine(m)
        f = np.asarray(rhs(jnp.asarray(y), 0.0, jnp.asarray(_params(m))))
        assert f[0] == pytest.approx(2.0 - 0.5 * y[0], rel=1e-14)
        assert f[1] == 0.0

    def test_a_model_without_observables_builds(self, tmp_path):
        """An empty groups block gave a (0,) weight array the matmul rejected."""
        net = _write(
            tmp_path,
            "noobs.net",
            """\
            begin parameters
                1 k 0.4
            end parameters
            begin species
                1 A() 5
                2 B() 0
            end species
            begin reactions
                1 1 2 k #_R1
            end reactions
            begin groups
            end groups
            """,
        )
        _assert_rhs_matches_engine(bngsim.Model.from_net(net))

    @pytest.mark.parametrize(
        "tok", [".5*k", "5e-01*k", "1.6605779e-09*k", "1e+24*k"],
        ids=["leading-dot", "exponent", "cbngl-volume", "1e24"],
    )  # fmt: skip
    def test_689_stat_factor_spellings(self, tmp_path, tok):
        sf = float(tok.split("*")[0])
        net = _write(
            tmp_path,
            "d.net",
            f"begin parameters\n    1 k {1.0 / sf!r}\nend parameters\n"
            "begin species\n    1 A() 10\nend species\n"
            f"begin reactions\n    1 1 0 {tok}\nend reactions\n",
        )
        rhs, y = _assert_rhs_matches_engine(bngsim.Model.from_net(net))
        # sf * k = 1: dA/dt = -A
        f = np.asarray(rhs(jnp.asarray(y), 0.0, jnp.asarray([1.0 / sf])))
        assert f[0] == pytest.approx(-y[0], rel=1e-12)

    @pytest.mark.filterwarnings("ignore:Legacy/deprecated BioNetGen")
    @pytest.mark.parametrize(
        ("text", "dsdt"),
        [
            (
                "begin parameters\n    1 k3  1.52\n    2 K4  114.418\nend parameters\n"
                "begin species\n    1 S() 100\n    2 P() 0\nend species\n"
                "begin reactions\n    1 1 2 Sat k3 K4\nend reactions\n",
                lambda S: -1.52 * S / (114.418 + S),
            ),
            (
                "begin parameters\n    1 V   1.0\n    2 K   50.0\n    3 n   2.0\nend parameters\n"
                "begin species\n    1 S() 100\n    2 P() 0\nend species\n"
                "begin reactions\n    1 1 2 Hill V K n\nend reactions\n",
                lambda S: -1.0 * S**2 / (50.0**2 + S**2),
            ),
        ],
        ids=["Sat", "Hill"],
    )
    def test_689_legacy_sat_and_hill(self, tmp_path, text, dsdt):
        m = bngsim.Model.from_net(_write(tmp_path, "l.net", text))
        rhs, y = _assert_rhs_matches_engine(m)
        f = np.asarray(rhs(jnp.asarray(y), 0.0, jnp.asarray(_params(m))))
        assert f[0] == pytest.approx(dsdt(y[0]), rel=1e-12)

    def test_699_a_function_reading_one_declared_after_it(self, tmp_path):
        """y() = 3 x(), x() = 0.5 k, declared in that order: rate 1.5 k."""
        net = _write(
            tmp_path,
            "fwd.net",
            """\
            begin parameters
                1 k 2
            end parameters
            begin species
                1 A() 10
            end species
            begin reactions
                1 1 0 y
            end reactions
            begin groups
                1 Atot 1
            end groups
            begin functions
                1 y() x*3
                2 x() k*0.5
            end functions
            """,
        )
        rhs, y = _assert_rhs_matches_engine(bngsim.Model.from_net(net))
        f = np.asarray(rhs(jnp.asarray(y), 0.0, jnp.asarray([2.0])))
        assert f[0] == pytest.approx(-3.0 * y[0], rel=1e-14)

    def test_608_unindexed_blocks(self, tmp_path):
        """The .net path's parser died on an unindexed parameters, species or groups
        block (`int('kcat')`); the loaders read them."""
        net = _write(
            tmp_path,
            "u.net",
            """\
            begin parameters
                kcat 0.3
            end parameters
            begin species
                A() 100
                B() 0
            end species
            begin groups
                Atot 1
            end groups
            begin functions
                1 fA() kcat*Atot
            end functions
            begin reactions
                1 1 2 fA
            end reactions
            """,
        )
        _assert_rhs_matches_engine(bngsim.Model.from_net(net))

    def test_694_an_overridden_derived_parameter(self, tmp_path):
        """The RHS reads each parameter's live value, so a pinned derived one is
        read pinned, as the engine reads it."""
        net = _write(
            tmp_path,
            "d.net",
            """\
            begin parameters
                1 k1   1     # Constant
                2 k2   2*k1  # ConstantExpression
            end parameters
            begin species
                1 A() 10
                2 B() 0
            end species
            begin reactions
                1 1 2 k2
            end reactions
            """,
        )
        m = bngsim.Model.from_net(net)
        m.set_param("k2", 5.0)
        _assert_rhs_matches_engine(m)

    def test_a_table_function_is_refused_by_name(self, tmp_path):
        (tmp_path / "drive.tfun").write_text("# time  drv\n0 1.0\n1 2.0\n5 10.0\n")
        net = _write(
            tmp_path,
            "t.net",
            """\
            begin parameters
                1 k 0.1
            end parameters
            begin functions
                1 drv() tfun('drive.tfun', time)
            end functions
            begin species
                1 A() 1
                2 B() 0
            end species
            begin reactions
                1 1 1,2 drv #R1
            end reactions
            """,
        )
        with pytest.raises(ValueError, match="table function"):
            jr.generate_jax_rhs(bngsim.Model.from_net(net))

    def test_an_sbml_model_is_refused_by_name(self):
        """An SBML compartment size is a writable parameter (#170), which the JAX
        RHS does not implement. It is refused naming that, never approximated."""
        pytest.importorskip("antimony")
        m = bngsim.Model.from_antimony_string(
            "model m; compartment C = 1; S in C; P in C; S = 10; P = 0; k = 0.3; "
            "J0: S -> P; k*S; end"
        )
        with pytest.raises(ValueError, match="compartment size is a parameter"):
            bngsim.Simulator(m, method="ode", jacobian="jax")


class TestTheJacobian:
    def test_matches_the_engines_analytical_jacobian(self, tmp_path):
        """A Functional rate law over an observable, and synthesis: the case AD is
        for, against the engine's own closed-form Jacobian."""
        net = _write(
            tmp_path,
            "f.net",
            """\
            begin parameters
                1 ksat 0.7
                2 Ksat 2.5
                3 ksyn 1.1
            end parameters
            begin functions
                1 fr() ksat*Stot/(Ksat+Stot)
            end functions
            begin species
                1 A() 3
                2 B() 1
            end species
            begin reactions
                1 1 2 fr #_R1
                2 0 1 ksyn #_R2
            end reactions
            begin groups
                1 Stot 1,2
            end groups
            """,
        )
        m = bngsim.Model.from_net(net)
        J_ref = m.jacobian(_positive_state(m))
        assert J_ref.source == "analytical"
        J_jax = np.asarray(
            jr.generate_jax_jacobian(m)(
                jnp.asarray(_positive_state(m)), 0.0, jnp.asarray(_params(m))
            )
        )
        np.testing.assert_allclose(J_jax, np.asarray(J_ref), rtol=1e-12, atol=1e-14)

    def test_the_engine_check_refuses_a_rhs_that_disagrees(self, tmp_path, monkeypatch):
        """The positive control for ``check_rhs_against_engine``: a JAX RHS off by
        a factor is refused before CVODE ever sees its Jacobian."""
        m = bngsim.Model.from_net(_write(tmp_path, "s.net", SYNTH))
        real = jr.generate_jax_rhs

        def doubled(model):
            rhs = real(model)
            wrong = lambda y, t, p: 2.0 * rhs(y, t, p)  # noqa: E731
            wrong.n_species, wrong.n_params = rhs.n_species, rhs.n_params
            return wrong

        monkeypatch.setattr(jr, "generate_jax_rhs", doubled)
        with pytest.raises(ValueError, match="disagrees with the engine"):
            jr.prepare_jax_jacobian(m)

    def test_jacobian_jax_needs_no_net_path_and_matches_auto(self, tmp_path):
        """jacobian='jax' changes only the Jacobian CVODE factors, so the solve
        agrees with jacobian='auto' to solver tolerance; and it reads the model,
        so it needs no file."""
        m = bngsim.Model.from_net(_write(tmp_path, "s.net", SYNTH))
        run = dict(t_span=(0.0, 4.0), n_points=9, rtol=1e-10, atol=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "2-80x slower"
            with_jax = bngsim.Simulator(m, method="ode", jacobian="jax").run(**run)
        auto = bngsim.Simulator(bngsim.Model.from_net(tmp_path / "s.net"), method="ode").run(**run)
        np.testing.assert_allclose(with_jax.species, auto.species, rtol=1e-8, atol=1e-10)
        t = np.asarray(auto.time)
        np.testing.assert_allclose(
            np.asarray(auto.species)[:, 0], 4.0 * (1 - np.exp(-0.5 * t)), rtol=1e-8, atol=1e-10
        )

    def test_net_path_is_deprecated_and_ignored(self, tmp_path):
        net = _write(tmp_path, "s.net", SYNTH)
        with pytest.warns(DeprecationWarning, match="net_path"):
            bngsim.Simulator(bngsim.Model.from_net(net), method="ode", net_path=str(net))


class TestDiffrax:
    """run_diffrax integrates the JAX RHS itself, so here a wrong RHS was a wrong
    trajectory."""

    @pytest.fixture(autouse=True)
    def _needs_diffrax(self):
        pytest.importorskip("diffrax")

    def test_synthesis_matches_the_closed_form(self, tmp_path):
        """A(t) = (ksyn/kdeg)(1 - e^{-kdeg t}) = 4(1 - e^{-t/2}); B stays 3. The
        .net path integrated dA/dt = ksyn*B - kdeg*A instead."""
        from bngsim._diffrax_solver import run_diffrax

        m = bngsim.Model.from_net(_write(tmp_path, "s.net", SYNTH))
        r = run_diffrax(m, t_end=6.0, n_points=7)
        t = r["time"]
        np.testing.assert_allclose(r["species"][:, 0], 4.0 * (1 - np.exp(-0.5 * t)), rtol=1e-6)
        np.testing.assert_allclose(r["species"][:, 1], 3.0, rtol=1e-12)
        assert r["species_names"] == ["A()", "B()"]

    def test_an_initial_condition_written_as_an_expression(self, tmp_path):
        """R() starts at 2*R0 = 20. The .net path looked the text '2*R0' up as a
        parameter name and started it at 0."""
        from bngsim._diffrax_solver import run_diffrax

        net = _write(
            tmp_path,
            "ic.net",
            """\
            begin parameters
                1 R0 10
                2 k  0.3
            end parameters
            begin species
                1 R() 2*R0
            end species
            begin reactions
                1 1 0 k
            end reactions
            """,
        )
        r = run_diffrax(bngsim.Model.from_net(net), t_end=2.0, n_points=3)
        np.testing.assert_allclose(r["species"][:, 0], 20.0 * np.exp(-0.3 * r["time"]), rtol=1e-6)

    def test_an_override_moves_the_derived_parameters_with_it(self, tmp_path):
        """k2 = 2*k1; overriding k1 to 0.25 makes the rate 0.5, as set_param does."""
        from bngsim._diffrax_solver import run_diffrax

        net = _write(
            tmp_path,
            "d.net",
            """\
            begin parameters
                1 k1   1     # Constant
                2 k2   2*k1  # ConstantExpression
            end parameters
            begin species
                1 A() 10
                2 B() 0
            end species
            begin reactions
                1 1 2 k2
            end reactions
            """,
        )
        m = bngsim.Model.from_net(net)
        r = run_diffrax(m, {"k1": 0.25}, t_end=2.0, n_points=3)
        np.testing.assert_allclose(r["species"][:, 0], 10.0 * np.exp(-0.5 * r["time"]), rtol=1e-6)
        assert m.get_param("k1") == 1.0  # the override went to a clone

    def test_a_model_with_events_is_refused(self):
        pytest.importorskip("antimony")
        from bngsim._diffrax_solver import run_diffrax

        m = bngsim.Model.from_antimony_string(
            "model m; compartment C = 1; S in C; S = 10; k = 0.3; J0: S -> ; k*S; "
            "E1: at time > 1: S = 5; end"
        )
        with pytest.raises(ValueError, match="events"):
            run_diffrax(m, t_end=2.0, n_points=3)
