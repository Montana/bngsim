"""Issue #892: method='nf' reads a table function indexed by time at its breakpoints.

NFsim's ``stepTo`` drew each waiting time from the total propensity and read
the propensities again only after a reaction fired. A table indexed by time
changes value at its breakpoints whether or not anything fires, so a rate that
is 0 until a later breakpoint never turned on: 100 ``X(p~1)`` with the table at
0 until t = 1 stayed at 100 on every row. With a total propensity next to 0 the
stale value held until the next firing. The fixture as written fires often
enough to hide it, and the test that ran it checked only the result's shape.

The tables are now read at each breakpoint, and a draw that passes one is drawn
again there, which is exact for this step table. Each start is checked against
the closed form, with the same helpers ``test_rulemonkey_model_time.py`` checks
method='rm' with, on outputs at the breakpoints and between them.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

from .test_rulemonkey_model_time import _assert_mean, _tfun_phos, _tfun_variant

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
TFUN_FILE = DATA / "nfsim" / "tfun_new_format" / "valid_file_step_time.xml"

# t = 1 and 2, the table's breakpoints, are outputs in the first grid. The second
# grid puts each one inside an output interval, where a draw past it is dropped.
ON_BREAKPOINTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
BETWEEN_BREAKPOINTS = [0.0, 0.75, 1.5, 2.25, 3.0]


def _has_nfsim() -> bool:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return bool(HAS_NFSIM)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")


def _nf(xml: Path, seed: int, times: list[float]) -> np.ndarray:
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    sim = bngsim.Simulator(m, method="nf", xml_path=str(xml))
    return np.asarray(sim.run(sample_times=times, seed=seed).observables["X_phos"], dtype=float)


@pytest.mark.parametrize("T", [ON_BREAKPOINTS, BETWEEN_BREAKPOINTS], ids=["on", "between"])
@pytest.mark.parametrize(
    "n0, n1, dephos",
    [
        (0, 100, 0.1),  # nothing can fire until the table turns on at t = 1
        (1, 99, 0.001),  # next to nothing can fire until then
        (1, 99, 1.0),
        (100, 0, 0.1),  # the fixture as written
    ],
)
def test_a_time_indexed_table_function_matches_the_closed_form(tmp_path, n0, n1, dephos, T):
    xml = _tfun_variant(tmp_path, n0, n1, dephos)
    runs = np.array([_nf(xml, seed, T) for seed in range(1, 201)])
    q1 = np.array([_tfun_phos(1, dephos, t) for t in T])
    q0 = np.array([_tfun_phos(0, dephos, t) for t in T])
    _assert_mean(runs, n1 * q1 + n0 * q0, n1 * q1 * (1 - q1) + n0 * q0 * (1 - q0))


@pytest.mark.parametrize(
    "rows, steps",
    [
        ("0 0\n1 10\n2 20\n", ((0.0, 1.0, 0.0), (1.0, 2.0, 10.0), (2.0, math.inf, 20.0))),
        # NFsim reads a table listed in decreasing time from the other side: 0 at
        # t = 0, 10 on (0, 1], 20 after. At each x it still gives the value from
        # before it, so it is read one ulp past each x instead; read at x, the
        # run kept 0 until t = 1 and 10 until t = 2.
        ("2 20\n1 10\n0 0\n", ((0.0, 1.0, 10.0), (1.0, math.inf, 20.0))),
    ],
    ids=["increasing", "decreasing"],
)
def test_a_time_indexed_table_from_a_file_matches_the_closed_form(tmp_path, rows, steps):
    (tmp_path / "table.tfun").write_text("# time tfun_rate\n" + rows, encoding="utf-8")
    xml = TFUN_FILE.read_text(encoding="utf-8")
    for old, new in (
        ('file="time_step.tfun"', 'file="table.tfun"'),
        ('concentration="100" name="X(p~0)"', 'concentration="0" name="X(p~0)"'),
        ('concentration="0" name="X(p~1)"', 'concentration="100" name="X(p~1)"'),
    ):
        assert old in xml, old
        xml = xml.replace(old, new)
    path = tmp_path / "file_table.xml"
    path.write_text(xml, encoding="utf-8")
    runs = np.array([_nf(path, seed, ON_BREAKPOINTS) for seed in range(1, 201)])
    q = np.array([_tfun_phos(1, 0.1, t, steps) for t in ON_BREAKPOINTS])
    _assert_mean(runs, 100 * q, 100 * q * (1 - q))
