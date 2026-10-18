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
the closed form, as ``test_rulemonkey_model_time.py`` checks method='rm'.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
TFUN = DATA / "nfsim" / "tfun_new_format" / "valid_inline_step_time.xml"


def _has_nfsim() -> bool:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return bool(HAS_NFSIM)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")


def _variant(tmp_path: Path, n0: int, n1: int, dephos: float) -> Path:
    xml = TFUN.read_text(encoding="utf-8")
    xml = xml.replace('concentration="100" name="X(p~0)"', f'concentration="{n0}" name="X(p~0)"')
    xml = xml.replace('concentration="0" name="X(p~1)"', f'concentration="{n1}" name="X(p~1)"')
    xml = xml.replace(
        'id="dephos" type="Constant" value="0.1"', f'id="dephos" type="Constant" value="{dephos}"'
    )
    out = tmp_path / f"tfun_{n0}_{n1}_{dephos}.xml"
    out.write_text(xml, encoding="utf-8")
    return out


def _phos(start: int, dephos: float, t: float) -> float:
    """P(X(p~1) at t) for one molecule: X(p~0) -> X(p~1) at dephos, and back at
    the step table 0 on [0, 1), 10 on [1, 2), 20 after, each piece solved."""
    p = float(start)
    for lo, hi, back in ((0.0, 1.0, 0.0), (1.0, 2.0, 10.0), (2.0, math.inf, 20.0)):
        if t <= lo:
            break
        rate = dephos + back
        eq = dephos / rate
        p = eq + (p - eq) * math.exp(-rate * (min(t, hi) - lo))
    return p


def _nf(xml: Path, seed: int, times: list[float]) -> np.ndarray:
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    sim = bngsim.Simulator(m, method="nf", xml_path=str(xml))
    return np.asarray(sim.run(sample_times=times, seed=seed).observables["X_phos"], dtype=float)


@pytest.mark.parametrize(
    "n0, n1, dephos",
    [
        (0, 100, 0.1),  # nothing can fire until the table turns on at t = 1
        (1, 99, 0.001),  # next to nothing can fire until then
        (1, 99, 1.0),
        (100, 0, 0.1),  # the fixture as written
    ],
)
def test_a_time_indexed_table_function_matches_the_closed_form(tmp_path, n0, n1, dephos):
    xml = _variant(tmp_path, n0, n1, dephos)
    T = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    runs = np.array([_nf(xml, seed, T) for seed in range(1, 201)])
    q1 = np.array([_phos(1, dephos, t) for t in T])
    q0 = np.array([_phos(0, dephos, t) for t in T])
    mean = n1 * q1 + n0 * q0
    var = n1 * q1 * (1 - q1) + n0 * q0 * (1 - q0)

    certain = var <= 0.0
    np.testing.assert_array_equal(
        runs[:, certain], np.broadcast_to(mean[certain], runs[:, certain].shape)
    )
    se = np.sqrt(var[~certain] / runs.shape[0])
    z = (runs[:, ~certain].mean(axis=0) - mean[~certain]) / se
    assert np.all(np.abs(z) < 4.5), z


def test_a_breakpoint_on_an_output_time_is_read_there(tmp_path):
    """Sampled at exactly t = 1 and 2, the breakpoints: each later row is read
    from the table's new value, not from one held over the output."""
    xml = _variant(tmp_path, 0, 100, 0.1)
    T = [0.0, 1.0, 2.0, 3.0]
    runs = np.array([_nf(xml, seed, T) for seed in range(1, 201)])
    assert (runs[:, 0] == 100).all() and (runs[:, 1] == 100).all()
    assert runs[:, 2].mean() < 5.0  # at 10 for one unit of time, e^-10 of 100 is left
