"""A stat factor ahead of a typed `.net` rate law: `2*MM kcat Km` (issue #1230).

BNG2.pl writes a reaction's statistical factor ahead of the rate law's type
keyword, for every type: `2*MM kcat Km`, `2*Sat k K`, `2*Hill V K h`. A symmetric
substrate is enough to produce one. run_network reads the factor off the token
before it matches the type (`sscanf("%lf*%s")` in Network3's `read_Rxn_array`)
and multiplies the whole rate by it. bngsim's loader compared the token to `MM`,
`Sat` and `Hill` exactly, so `2*MM` fell through to the elementary branch,
which took the factor off and read `MM` as a rate constant's name, and the
model was refused as "references unknown parameter 'MM'".

What this file pins:

* the files BNG2.pl 2.9.3 writes for a symmetric substrate under each type load,
  from both doors, and give run_network's trajectory;
* `2*<law>` with kcat is the same model as `<law>` with 2·kcat on every path that
  reads the factor: the ODE right-hand side, interpreted and compiled, SSA, and
  the sensitivities;
* the factor is held to what #803 holds an elementary one to.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import ModelError
from bngsim._net_reader import build_model_from_parsed, parse_net_file


def _has_cc() -> bool:
    from bngsim import _codegen as cg

    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")

#: The reactions BNG2.pl 2.9.3 writes for `S(a~u) -> S(a~p) <law>` (and, for MM,
#: `E() + S(a~u) -> E() + S(a~p) MM(kcat,Km)`), S having two equivalent `a`
#: sites: the first reaction has two sites to act on, so its stat factor is 2.
#: The control row is the elementary law, whose prefix bngsim already read.
BNG2_REACTIONS = {
    "MM": ("1 1,2 1,3 2*MM kcat Km #_R1", "2 1,3 1,4 MM kcat Km #_R1"),
    "Sat": ("1 2 3 2*Sat kcat Km #_R1", "2 3 4 Sat kcat Km #_R1"),
    "Hill": ("1 2 3 2*Hill kcat Km h #_R1", "2 3 4 Hill kcat Km h #_R1"),
    "elementary": ("1 2 3 2*kcat #_R1", "2 3 4 kcat #_R1"),
}

#: Su = S(a~u) sites at t = 0, 1, ..., 5 from run_network (BioNetGen 2.9.3) on
#: those files, `simulate({method=>"ode",t_end=>5,n_steps=>10,atol=>1e-12,
#: rtol=>1e-12})`.
RUN_NETWORK_SU = {
    "MM": [200.0, 149.0832712371, 94.71472964411, 53.6386359706, 32.83161596094, 16.08523469375],
    "Sat": [200.0, 195.45315877, 190.3768049983, 185.1002154663, 179.7143075394, 174.2595987544],
    "Hill": [
        200.0,
        195.4518651127,
        190.0410211898,
        184.347415813,
        178.5384278319,
        172.6707111567,
    ],
    "elementary": [
        200.0,
        27.06705664737,
        3.663127777788,
        0.4957504354011,
        0.06709252560158,
        0.009079985967009,
    ],
}


def _bng2_net(tmp_path: Path, kind: str) -> Path:
    first, second = BNG2_REACTIONS[kind]
    p = tmp_path / f"sym_{kind}.net"
    p.write_text(
        textwrap.dedent(
            f"""
            # Created by BioNetGen 2.9.3
            begin parameters
                1 kcat  2  # Constant
                2 Km    3  # Constant
                3 h     2  # Constant
            end parameters
            begin species
                1 E() 10
                2 S(a~u,a~u) 100
                3 S(a~p,a~u) 0
                4 S(a~p,a~p) 0
            end species
            begin reactions
                {first}
                {second}
            end reactions
            begin groups
                1 Su                   2*2,3
                2 Sp                   3,2*4
            end groups
            """
        ).strip()
        + "\n"
    )
    return p


def _quiet(fn, *args):
    # Sat and Hill are rewritten as functions, with a UserWarning that says so.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*args)


@pytest.mark.parametrize("kind", list(BNG2_REACTIONS))
class TestTheFilesBng2Writes:
    def test_the_factor_is_read_ahead_of_the_type(self, tmp_path: Path, kind: str) -> None:
        parsed = _quiet(parse_net_file, _bng2_net(tmp_path, kind))
        first, second = parsed["reactions"]
        assert (first["stat_factor"], second["stat_factor"]) == (2.0, 1.0)
        assert first["type"] == second["type"]
        assert first["type"] == {"MM": "mm", "elementary": "elementary"}.get(kind, "functional")

    @pytest.mark.parametrize("codegen", [False, pytest.param(True, marks=requires_cc)])
    def test_both_doors_give_run_networks_trajectory(
        self, tmp_path: Path, kind: str, codegen: bool
    ) -> None:
        net = _bng2_net(tmp_path, kind)
        loaded = _quiet(bngsim.Model.from_net, str(net))
        built = _quiet(build_model_from_parsed, _quiet(parse_net_file, net))
        assert built._core.codegen_data() == loaded._core.codegen_data()
        run = bngsim.Simulator(loaded, method="ode", codegen=codegen).run(
            t_span=(0.0, 5.0), n_points=6, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(run.observables["Su"]), RUN_NETWORK_SU[kind], rtol=1e-8
        )


#: `E + S -> E + P` (MM) or `S -> P` (Sat, Hill) on a model with no symmetry, so
#: that `2*<law>` and `<law>` at twice the kcat are the same rate written two ways.
LAWS = {
    "MM": ("1,2", "1,3", "MM kcat Km"),
    "Sat": ("2", "3", "Sat kcat Km"),
    "Hill": ("2", "3", "Hill kcat Km h"),
}


def _law_net(tmp_path: Path, kind: str, prefix: str, kcat: float) -> Path:
    reactants, products, law = LAWS[kind]
    p = tmp_path / f"{kind}_{prefix.strip('*') or 'bare'}.net"
    p.write_text(
        textwrap.dedent(
            f"""
            begin parameters
              1 kcat {kcat!r}
              2 Km 3
              3 h 2
            end parameters
            begin species
              1 E() 10
              2 S() 100
              3 P() 0
            end species
            begin reactions
              1 {reactants} {products} {prefix}{law}
            end reactions
            begin groups
              1 St 2
              2 Pt 3
            end groups
            """
        ).strip()
        + "\n"
    )
    return p


@pytest.mark.parametrize("kind", list(LAWS))
class TestTheFactorScalesTheWholeRate:
    """2·(kcat · g) and (2·kcat) · g round to one double, so the two must agree
    exactly, wherever a path reads the factor."""

    def _pair(self, tmp_path: Path, kind: str):
        doubled = _quiet(bngsim.Model.from_net, str(_law_net(tmp_path, kind, "2*", 2.0)))
        plain = _quiet(bngsim.Model.from_net, str(_law_net(tmp_path, kind, "", 4.0)))
        return doubled, plain

    @pytest.mark.parametrize("codegen", [False, pytest.param(True, marks=requires_cc)])
    def test_ode(self, tmp_path: Path, kind: str, codegen: bool) -> None:
        runs = [
            bngsim.Simulator(m, method="ode", codegen=codegen).run(
                t_span=(0.0, 5.0), n_points=11, rtol=1e-12, atol=1e-12
            )
            for m in self._pair(tmp_path, kind)
        ]
        np.testing.assert_array_equal(np.asarray(runs[0].species), np.asarray(runs[1].species))

    @pytest.mark.parametrize(
        ("method", "kw"),
        [
            ("ssa", {"codegen": False}),
            pytest.param("ssa", {}, marks=requires_cc),
            ("psa", {"poplevel": 100}),
        ],
        ids=["ssa-interpreted", "ssa-default", "psa"],
    )
    def test_stochastic(self, tmp_path: Path, kind: str, method: str, kw: dict) -> None:
        runs = [
            bngsim.Simulator(m, method=method, **kw).run((0.0, 5.0), 11, seed=7)
            for m in self._pair(tmp_path, kind)
        ]
        np.testing.assert_array_equal(np.asarray(runs[0].species), np.asarray(runs[1].species))

    def test_sensitivities(self, tmp_path: Path, kind: str) -> None:
        """d/dkcat of 2·kcat·g is twice d/d(kcat') of kcat'·g; d/dKm is the same."""
        doubled, plain = (
            np.asarray(
                bngsim.Simulator(m, method="ode", sensitivity_params=["kcat", "Km"])
                .run(t_span=(0.0, 5.0), n_points=11, rtol=1e-12, atol=1e-13)
                .sensitivities
            )
            for m in self._pair(tmp_path, kind)
        )
        np.testing.assert_allclose(doubled[..., 0], 2.0 * plain[..., 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(doubled[..., 1], plain[..., 1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", list(LAWS))
@pytest.mark.parametrize("prefix", ["-1", "0", "0x2"])
def test_a_typed_laws_factor_must_be_a_positive_decimal(
    tmp_path: Path, kind: str, prefix: str
) -> None:
    """#803's rule for an elementary factor holds ahead of a type keyword too."""
    net = _law_net(tmp_path, kind, f"{prefix}*", 2.0)
    with pytest.raises(ValueError, match="must be a positive decimal number"):
        parse_net_file(net)
    with pytest.raises(ModelError, match="must be a positive decimal number"):
        bngsim.Model.from_net(str(net))


def test_a_factor_other_than_two_scales_sat_against_the_hand_written_ode(tmp_path: Path) -> None:
    """`0.5*Sat k K` on `S -> P`: dS/dt = -0.5 * k * S / (K + S), run_network's Sat
    with its stat factor, integrated by scipy as the existing Sat/Hill tests do."""
    scipy_integrate = pytest.importorskip("scipy.integrate")
    net = tmp_path / "half_sat.net"
    net.write_text(
        textwrap.dedent(
            """
            begin parameters
              1 k 1.52
              2 K 114.418
            end parameters
            begin species
              1 S() 100
              2 P() 0
            end species
            begin reactions
              1 1 2 0.5*Sat k K
            end reactions
            begin groups
              1 Stot 1
            end groups
            """
        ).strip()
        + "\n"
    )
    parsed = _quiet(parse_net_file, net)
    (rxn,) = parsed["reactions"]
    assert (rxn["type"], rxn["stat_factor"]) == ("functional", 0.5)
    model = build_model_from_parsed(parsed)
    assert (
        model._core.codegen_data() == _quiet(bngsim.Model.from_net, str(net))._core.codegen_data()
    )
    t = np.linspace(0.0, 50.0, 11)
    got = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 50.0), n_points=11, rtol=1e-10, atol=1e-10
    )
    ref = scipy_integrate.solve_ivp(
        lambda _t, y: [-0.5 * 1.52 * y[0] / (114.418 + y[0])],
        (0.0, 50.0),
        [100.0],
        t_eval=t,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(np.asarray(got.species)[:, 0], ref.y[0], rtol=1e-6)


def test_a_prefix_that_is_not_a_number_stays_one_name(tmp_path: Path) -> None:
    """`2k*MM kcat Km`: `2k` is not a factor (issue #803), so the token is one
    name, not an MM law, and the line is refused for what follows that name."""
    net = _bng2_net(tmp_path, "MM")
    net.write_text(net.read_text().replace("2*MM kcat Km", "2k*MM kcat Km"))
    with pytest.raises(ValueError, match=r"after its rate law '2k\*MM'"):
        parse_net_file(net)
    with pytest.raises(ModelError, match=r"after its rate law '2k\*MM'"):
        bngsim.Model.from_net(str(net))


@pytest.mark.parametrize("kind", list(LAWS))
def test_a_unit_conversion_prefix_in_scientific_notation_is_a_factor(
    tmp_path: Path, kind: str
) -> None:
    """A compartmental model's unit conversion is written the same way, as a
    number ahead of the type (`1.6605503e-12*MM kcat Km`); the `e` of its
    exponent is not one of the hex letters a factor is refused for."""
    parsed = _quiet(parse_net_file, _law_net(tmp_path, kind, "1.6605503e-12*", 2.0))
    (rxn,) = parsed["reactions"]
    assert rxn["stat_factor"] == 1.6605503e-12
    assert rxn["type"] == ("mm" if kind == "MM" else "functional")
