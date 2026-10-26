"""Tests for build_model_from_parsed reaction rate handling.

Since issue #803 the rate column is read once, by the C++ loader: `parse_net_file`
returns its reading and `build_model_from_parsed` hands it to ModelBuilder as the
loader does. Before, this reader wrapped any rate column that was not a
parameter name in a synthetic `__net_reader_func_<i>` function, a second reading
of the column that `Model.from_net` did not share.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import ModelError
from bngsim._net_reader import build_model_from_parsed, parse_net_file


def _write_net(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "model.net"
    p.write_text(textwrap.dedent(body).strip() + "\n")
    return p


class TestBuildModelFromParsedRates:
    def test_a_number_times_a_parameter_is_a_stat_factor(self, tmp_path: Path) -> None:
        """`0.5*kp2` is BNG2.pl's stat-factor prefix: an elementary reaction at kp2,
        scaled by 0.5, as the loader reads it. A(t) = 100 exp(-0.5*2*t)."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 0.5*kp2
            end reactions
            begin groups
            end groups
            """,
        )
        parsed = parse_net_file(net)
        (rxn,) = parsed["reactions"]
        assert (rxn["type"], rxn["rate_law"], rxn["stat_factor"]) == ("elementary", "kp2", 0.5)
        model = build_model_from_parsed(parsed)
        cd = model._core.codegen_data()
        assert cd["functions"] == []
        assert cd == bngsim.Model.from_net(str(net))._core.codegen_data()
        t = np.linspace(0.0, 2.0, 5)
        a = np.asarray(
            bngsim.Simulator(model, method="ode").run(t_span=(0.0, 2.0), n_points=5).species
        )
        np.testing.assert_allclose(a[:, 0], 100.0 * np.exp(-1.0 * t), rtol=1e-6)

    def test_elementary_reuses_declared_function_no_duplicate(self, tmp_path: Path) -> None:
        """When rate_law names an existing .net function, that function is used once."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 k1 0.1
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin functions
              1 myRate() k1*2
            end functions
            begin reactions
              1 1 2 myRate
            end reactions
            begin groups
            end groups
            """,
        )
        model = build_model_from_parsed(parse_net_file(net))
        cd = model._core.codegen_data()
        names = [f["name"] for f in cd["functions"]]
        assert names == ["myRate"]
        assert len(names) == 1
        rx = cd["reactions"][0]
        assert rx["type"] == "functional"
        assert rx["function_name"] == "myRate"

    def test_elementary_parameter_rate_unchanged(self, tmp_path: Path) -> None:
        """Declared parameter name in the rate column stays elementary."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 k1 0.1
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 k1
            end reactions
            begin groups
            end groups
            """,
        )
        model = build_model_from_parsed(parse_net_file(net))
        cd = model._core.codegen_data()
        assert cd["functions"] == []
        rx = cd["reactions"][0]
        assert rx["type"] == "elementary"
        assert rx["function_name"] == "k1"

    def test_a_rate_column_is_not_wrapped_in_a_function(self, tmp_path: Path) -> None:
        """No synthetic function is minted any more, so a user function of any name,
        the old synthetic one's included, is left alone and nothing is added."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin functions
              1 __net_reader_func_0() 1.0
            end functions
            begin reactions
              1 1 2 0.5*kp2
            end reactions
            begin groups
            end groups
            """,
        )
        model = build_model_from_parsed(parse_net_file(net))
        cd = model._core.codegen_data()
        assert [f["name"] for f in cd["functions"]] == ["__net_reader_func_0"]
        rx = cd["reactions"][0]
        assert (rx["type"], rx["function_name"], rx["stat_factor"]) == ("elementary", "kp2", 0.5)

    def test_forced_elementary_matching_function_name_stays_single_function(
        self, tmp_path: Path
    ) -> None:
        """If parsed data still says elementary but rate matches a function, reuse it."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 k1 0.1
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin functions
              1 myRate() k1*2
            end functions
            begin reactions
              1 1 2 myRate
            end reactions
            begin groups
            end groups
            """,
        )
        parsed = parse_net_file(net)
        parsed["reactions"][0]["type"] = "elementary"
        model = build_model_from_parsed(parsed)
        cd = model._core.codegen_data()
        assert [f["name"] for f in cd["functions"]] == ["myRate"]
        assert cd["reactions"][0]["type"] == "functional"
        assert cd["reactions"][0]["function_name"] == "myRate"

    def test_empty_rate_law_raises(self, tmp_path: Path) -> None:
        parsed = parse_net_file(
            _write_net(
                tmp_path,
                """
                begin parameters
                  1 k1 0.1
                end parameters
                begin species
                  1 A() 100
                  2 B() 0
                end species
                begin reactions
                  1 1 2 k1
                end reactions
                begin groups
                end groups
                """,
            )
        )
        parsed["reactions"][0]["rate_law"] = "   "
        parsed["reactions"][0]["type"] = "elementary"
        with pytest.raises(ValueError, match="empty rate_law"):
            build_model_from_parsed(parsed)

    def test_a_stat_factor_with_no_rate_constant_is_refused(self, tmp_path: Path) -> None:
        """`0.5*` used to load under Model.from_net as a reaction that never fires
        (ModelBuilder passes an empty rate name over); both doors now refuse it."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 0.5*
            end reactions
            begin groups
            end groups
            """,
        )
        with pytest.raises(ValueError, match="stat factor but no rate constant"):
            parse_net_file(net)
        with pytest.raises(ModelError, match="stat factor but no rate constant"):
            bngsim.Model.from_net(str(net))

    @pytest.mark.parametrize("prefix", ["-1", "0", "0.0", "0x10", "0x1p3"])
    def test_a_stat_factor_that_is_not_a_positive_decimal_is_refused(
        self, tmp_path: Path, prefix: str
    ) -> None:
        """stod takes a sign and C99 hex: `-1*kp2` ran the reaction backwards (a
        negative SSA propensity), `0*kp2` built one that never fires, and
        `0x10*kp2` read as 16. BNG2.pl writes positive decimal factors only."""
        net = _write_net(
            tmp_path,
            f"""
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 {prefix}*kp2
            end reactions
            begin groups
            end groups
            """,
        )
        with pytest.raises(ValueError, match="must be a positive decimal number"):
            parse_net_file(net)
        with pytest.raises(ModelError, match="must be a positive decimal number"):
            bngsim.Model.from_net(str(net))

    def test_a_rate_column_naming_nothing_is_refused(self, tmp_path: Path) -> None:
        """`(0.5*kp2` names no parameter or function; the builder refuses it from
        either door, naming the column."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 (0.5*kp2
            end reactions
            begin groups
            end groups
            """,
        )
        with pytest.raises((RuntimeError, ValueError), match=r"\(0\.5\*kp2"):
            build_model_from_parsed(parse_net_file(net))
        with pytest.raises(ModelError, match=r"\(0\.5\*kp2"):
            bngsim.Model.from_net(str(net))


def _one_reaction_net(tmp_path: Path, line: str, extra_params: str = "") -> Path:
    """A network whose reactions block is *line*; E, S, P and every name the
    cases below read are declared, so only the line itself can be refused."""
    net = tmp_path / "model.net"
    net.write_text(
        textwrap.dedent(
            """\
            begin parameters
              1 k     1.0
              2 tau   5.0
              3 kcat  2.0
              4 Km    3.0
              5 V     4.0
              6 K     5.0
              7 n     2.0
            {extra}end parameters
            begin species
              1 E() 10
              2 S() 100
              3 P() 0
            end species
            begin reactions
              {line}
            end reactions
            begin groups
              1 Ptot 3
            end groups
            begin functions
              1 f() 2*k
            end functions
            """
        ).format(line=line, extra=extra_params)
    )
    return net


class TestTokensAfterTheRateLaw:
    """A reaction line is `<index> <reactants> <products> <rate law>`, and only a
    Sat, Hill or MM rate law runs past one token, on its constants. The loader
    used to read the line and drop whatever followed, so `1 2 3 k delay=tau`
    loaded as an undelayed reaction with no error; run_network refuses it
    ("Undefined rateLaw type"). Issue #1231."""

    @pytest.mark.parametrize(
        ("line", "extra", "law"),
        [
            ("1 2 3 k delay=tau", "delay=tau", "k"),
            ("1 2 3 2*k delay=tau", "delay=tau", "2*k"),
            ("1 2 3 f delay=tau", "delay=tau", "f"),
            ("1 2 3 k 2", "2", "k"),
            ("1 2 3 k delay = tau", "delay = tau", "k"),
            ("1 1,2 1,3 MM kcat Km delay=tau", "delay=tau", "MM kcat Km"),
            ("1 1,2 1,3 MM kcat Km Km", "Km", "MM kcat Km"),
        ],
        ids=["elementary", "stat-factor", "functional", "number", "spaced", "mm", "mm-third"],
    )
    def test_a_token_after_the_rate_law_is_refused(
        self, tmp_path: Path, line: str, extra: str, law: str
    ) -> None:
        net = _one_reaction_net(tmp_path, line)
        message = re.escape(f"has '{extra}' after its rate law '{law}'")
        with pytest.raises(ValueError, match=message):
            parse_net_file(net)
        with pytest.raises(ModelError, match=message):
            bngsim.Model.from_net(str(net))

    @pytest.mark.parametrize(
        ("line", "law", "constant"),
        [
            ("1 1,2 1,3 Sat k Km delay=tau", "Sat", "delay=tau"),
            ("1 1,2 1,3 Sat k Km 2", "Sat", "2"),
            ("1 2 3 Hill V K delay=tau", "Hill", "delay=tau"),
        ],
        ids=["sat-delay", "sat-literal", "hill-delay"],
    )
    def test_a_sat_or_hill_constant_that_is_not_a_name_is_refused(
        self, tmp_path: Path, line: str, law: str, constant: str
    ) -> None:
        """Sat takes one constant per reactant, so its token count cannot tell a
        constant from a modifier; a constant is a parameter name, which run_network
        looks up ("Undefined rate constant"). With two reactants `Sat k Km 2`
        loaded with a literal 2, and `Sat k Km delay=tau` reached ExprTk."""
        net = _one_reaction_net(tmp_path, line)
        message = re.escape(f"gives its {law} rate law the constant '{constant}'")
        with pytest.raises(ValueError, match=message):
            parse_net_file(net)
        with pytest.raises(ModelError, match=message):
            bngsim.Model.from_net(str(net))

    def test_the_comment_is_not_read_as_tokens(self, tmp_path: Path) -> None:
        """What follows `#` is BNG2.pl's comment (`#_R1`, `unit_conversion=...`),
        not part of the rate law: the line loads as plain `A -> B` at k."""
        net = _one_reaction_net(tmp_path, "1 2 3 k #_R1 delay=tau")
        (rxn,) = parse_net_file(net)["reactions"]
        assert (rxn["type"], rxn["rate_law"], rxn["stat_factor"]) == ("elementary", "k", 1.0)
        bngsim.Model.from_net(str(net))

    @pytest.mark.parametrize("factor", ["", "2*"], ids=["bare", "factor"])
    @pytest.mark.parametrize("name", ["MM", "Ele", "Sat", "Hill"])
    def test_a_rate_constant_named_like_a_type_keyword_still_loads(
        self, tmp_path: Path, name: str, factor: str
    ) -> None:
        """A rate column of one token is a rate constant or function, whatever it
        is called (run_network reads it so too), with a stat factor or without.
        `Sat` or `Hill` alone used to be read as that law with no constants, and
        `2*Sat` became so once the factor came off first (issue #1230). `MM kcat`
        with one constant is refused in test_net_reader_legacy_and_tfun.py."""
        net = _one_reaction_net(tmp_path, f"1 2 3 {factor}{name}", extra_params=f"8 {name} 1.5\n")
        (rxn,) = parse_net_file(net)["reactions"]
        assert (rxn["type"], rxn["rate_law"]) == ("elementary", name)
        assert rxn["stat_factor"] == (2.0 if factor else 1.0)
        bngsim.Model.from_net(str(net))

    @pytest.mark.parametrize(
        ("line", "same_rate", "doubled"),
        [
            ("1 1,2 1,3 2*MM kcat Km", "1 1,2 1,3 MM kcat2 Km", "8 kcat2 4.0\n"),
            ("1 2 3 2*Sat k Km", "1 2 3 Sat k2 Km", "8 k2 2.0\n"),
            ("1 2 3 2*Hill V K n", "1 2 3 Hill V2 K n", "8 V2 8.0\n"),
        ],
        ids=["MM", "Sat", "Hill"],
    )
    @pytest.mark.filterwarnings("ignore:Legacy/deprecated BioNetGen .net rate law token")
    def test_a_stat_factor_on_a_typed_rate_law_scales_the_rate(
        self, tmp_path: Path, line: str, same_rate: str, doubled: str
    ) -> None:
        """run_network multiplies the whole rate by the factor (network.cpp:
        `stat_factor * kcat * Et * S / (Km + S)`, `stat_factor * param[0] * ...`),
        so `2*T c1 c2...` is `T c1' c2...` with c1' = 2*c1. BNG2.pl writes this
        form for a symmetric substrate; until issue #1230 the loader refused it."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        expect = bngsim.Model.from_net(str(_one_reaction_net(tmp_path / "b", same_rate, doubled)))
        got = bngsim.Model.from_net(str(_one_reaction_net(tmp_path / "a", line)))

        def run(model: bngsim.Model) -> np.ndarray:
            sim = bngsim.Simulator(model, method="ode")
            return np.asarray(sim.run(t_span=(0.0, 5.0), n_points=11).species)

        np.testing.assert_allclose(run(got), run(expect), rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize(
        ("line", "message"),
        [
            ("1 2 3 2*Ele k", "spells out the Ele rate-law type"),
            ("1 1,2 1,3 2*MM kcat", "an MM rate law takes two constants"),
        ],
        ids=["ele", "mm-one-constant"],
    )
    def test_a_factor_ahead_of_the_type_leaves_the_types_own_refusal(
        self, tmp_path: Path, line: str, message: str
    ) -> None:
        """The factor comes off before the type is matched (issue #1230), so the
        line is judged by its type: `2*Ele k` by the Ele refusal and `2*MM kcat`
        by MM's count, not by a stat-factor message."""
        net = _one_reaction_net(tmp_path, line)
        with pytest.raises(ValueError, match=re.escape(message)):
            parse_net_file(net)
        with pytest.raises(ModelError, match=re.escape(message)):
            bngsim.Model.from_net(str(net))

    def test_the_ele_keyword_is_refused_as_such(self, tmp_path: Path) -> None:
        """run_network reads `Ele k` as elementary k; BNG2.pl never writes it, and
        this loader refused it at build as "unknown parameter 'Ele'"."""
        net = _one_reaction_net(tmp_path, "1 2 3 Ele k")
        message = re.escape("spells out the Ele rate-law type")
        with pytest.raises(ValueError, match=message):
            parse_net_file(net)
        with pytest.raises(ModelError, match=message):
            bngsim.Model.from_net(str(net))


class TestANumberReadOnlyInPart:
    """A group factor and an inline table value are numbers and nothing else.
    stod stops at the first character it cannot use, and these two sites never
    asked how much it had read, so `1/2` read 1 with no error (issue #1235's
    mechanism, which a decimal-comma locale reached with `0.5`)."""

    @pytest.mark.parametrize("factor", ["1/2", "0.5x", "nan"])
    def test_a_group_factor_that_is_not_a_number_is_refused(
        self, tmp_path: Path, factor: str
    ) -> None:
        net = _write_net(
            tmp_path,
            f"""
            begin parameters
              1 k 1
            end parameters
            begin species
              1 A() 100
            end species
            begin reactions
              1 1 0 k
            end reactions
            begin groups
              1 Half {factor}*1
            end groups
            """,
        )
        message = re.escape(f"whose factor '{factor}' is not a number")
        with pytest.raises(ValueError, match=message):
            parse_net_file(net)
        with pytest.raises(ModelError, match=message):
            bngsim.Model.from_net(str(net))

    @pytest.mark.parametrize(
        ("xs", "ys", "bad"),
        [("0,1,2", "0,1/2,1", "1/2"), ("0,0.5x,2", "0,1,2", "0.5x")],
        ids=["value", "breakpoint"],
    )
    def test_a_table_value_that_is_not_a_number_is_refused(
        self, tmp_path: Path, xs: str, ys: str, bad: str
    ) -> None:
        net = _write_net(
            tmp_path,
            f"""
            begin parameters
              1 k 1
            end parameters
            begin species
              1 A() 0
            end species
            begin functions
              1 f() tfun([{xs}],[{ys}],time())
            end functions
            begin reactions
              1 0 1 f #_FUNC
            end reactions
            begin groups
              1 Atot 1
            end groups
            """,
        )
        message = re.escape(f"has value '{bad}', which is not a number")
        with pytest.raises(ValueError, match=message):
            parse_net_file(net)
        with pytest.raises(ModelError, match=message):
            bngsim.Model.from_net(str(net))

    def test_the_numbers_bng2pl_writes_still_load(self, tmp_path: Path) -> None:
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 k 1
            end parameters
            begin species
              1 A() 0
            end species
            begin functions
              1 f() tfun([0,1e-3,2.5],[0,0.25,1.5e+1],time())
            end functions
            begin reactions
              1 0 1 f #_FUNC
            end reactions
            begin groups
              1 Half 0.5*1,2.5e-1*1
            end groups
            """,
        )
        model = bngsim.Model.from_net(str(net))
        assert model.n_species == 1
