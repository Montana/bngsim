"""Tests for build_model_from_parsed reaction rate handling.

Since issue #803 the rate column is read once, by the C++ loader: `parse_net_file`
returns its reading and `build_model_from_parsed` hands it to ModelBuilder as the
loader does. Before, this reader wrapped any rate column that was not a
parameter name in a synthetic `__net_reader_func_<i>` function, a second reading
of the column that `Model.from_net` did not share.
"""

from __future__ import annotations

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
