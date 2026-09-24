"""The local build A/B timer (benchmarks/perf_ab.py, GH #702) must not cry wolf.

It compares two bngsim builds by wall clock on one machine and exits 1 when it
calls a model SLOWER. A timer that flags noise gets ignored, so these tests pin
the rule that decides a flag, without running any build. Before these rules an
A/A run (one build against itself, 3 repeats) flagged a 3 ms measurement as 10 %
slower: a flag needs the ratio past the threshold, the whole bootstrap interval
past 1, an absolute change above the timer-noise floor, and enough repeats.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root()
_SCRIPT = _ROOT / "benchmarks" / "perf_ab.py" if _ROOT else None
if _SCRIPT is None or not _SCRIPT.is_file():
    pytest.skip("benchmarks/perf_ab.py not in this checkout", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("perf_ab", _SCRIPT)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def test_identical_samples_give_ratio_one_and_a_tight_interval():
    a = [0.100, 0.101, 0.099, 0.100, 0.102]
    point, lo, hi = P.ratio_ci(a, list(a))
    assert point == pytest.approx(1.0)
    assert lo <= 1.0 <= hi


def test_a_clear_slowdown_has_its_whole_interval_above_one():
    a = [0.100, 0.101, 0.099, 0.100, 0.102, 0.100, 0.101]
    b = [x * 1.20 for x in a]
    point, lo, hi = P.ratio_ci(a, b)
    assert point == pytest.approx(1.20, rel=1e-3)
    assert lo > 1.1
    assert P.verdict(point, lo, hi, 0.05, a=0.100, b=0.120, min_delta=0.002, n=7) == "SLOWER"


def test_a_clear_speedup_is_reported_as_faster():
    assert P.verdict(0.8, 0.75, 0.85, 0.05, a=0.100, b=0.080, min_delta=0.002, n=5) == "faster"


def test_a_change_below_the_absolute_floor_is_not_flagged():
    # x1.10 with the interval above 1, but 0.3 ms on a 3 ms measurement: timer noise.
    assert P.verdict(1.10, 1.03, 1.14, 0.05, a=0.0030, b=0.0033, min_delta=0.002, n=5) == ""


def test_too_few_repeats_are_never_flagged():
    assert P.verdict(1.5, 1.4, 1.6, 0.05, a=0.1, b=0.15, min_delta=0.002, n=3) == ""


def test_an_interval_that_straddles_one_is_not_flagged():
    assert P.verdict(1.08, 0.97, 1.20, 0.05, a=0.1, b=0.108, min_delta=0.002, n=5) == ""


def test_a_ratio_inside_the_threshold_is_not_flagged():
    assert P.verdict(1.03, 1.01, 1.05, 0.05, a=0.1, b=0.103, min_delta=0.002, n=5) == ""


def test_geomean_ignores_non_positive_entries():
    assert P.geomean([2.0, 0.5]) == pytest.approx(1.0)
    assert P.geomean([2.0, 0.0]) == pytest.approx(2.0)


def test_the_default_model_set_is_committed():
    class Args:
        models = ""
        max_species = 0

    models = P.select_models(Args())
    assert len(models) >= 20
    for m in models:
        assert (_ROOT / "benchmarks" / m["net_file"]).is_file(), m["net_file"]


def test_help_does_not_measure(capsys):
    with pytest.raises(SystemExit) as e:
        P.main(["--help"])
    assert e.value.code == 0
    assert "--baseline" in capsys.readouterr().out
