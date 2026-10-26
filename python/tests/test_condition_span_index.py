"""Whether a match lies inside an ``if()`` condition is answered by bisection
(issue #1256).

Four scans in ``_switch_sensitivity`` asked it of every match of a pattern
against every condition span, ``any(lo <= start and end <= hi for ...)``: a
product of the two counts. A rate law written out through a diamond of rules
holds thousands of both, and building a sensitivity simulator for #1145's size
jump took 29.6 s at 16 rules, 3.6 s of it in that one ``any``, evaluated 74
million times, and 298 s at 20, where the scan's share is what indexing it took
off: the build is 35 s now.

``_SpanIndex`` answers the same two questions from the spans sorted by start
and the running maximum of their ends: some span holds ``[start, end]`` exactly
when, of the spans that start at or before ``start``, the furthest end is at or
after ``end``. The property tests below hold it to the brute-force answer on
spans nested, overlapping, adjacent, repeated and empty.
"""

from __future__ import annotations

import random
import time

import bngsim
import pytest
from bngsim._switch_sensitivity import (
    SwitchConditionScope,
    _compares_outside_an_if,
    _SpanIndex,
    state_switch_root_conditions,
    uncompensated_condition_reason,
)


def _contains(spans, start, end):
    return any(lo <= start and end <= hi for lo, hi in spans)


def _covers(spans, start):
    return any(lo <= start < hi for lo, hi in spans)


def _random_spans(rng, n, width):
    spans = []
    for _ in range(n):
        lo = rng.randrange(width)
        kind = rng.random()
        if kind < 0.1:
            hi = lo  # empty
        elif kind < 0.2 and spans:
            lo, hi = rng.choice(spans)  # repeated
        else:
            hi = min(width, lo + rng.randrange(1, max(2, width // 3)))
        spans.append((lo, hi))
    return spans


@pytest.mark.parametrize("seed", range(40))
def test_the_index_gives_the_brute_force_answer(seed):
    rng = random.Random(seed)
    width = rng.choice([5, 20, 200])
    spans = _random_spans(rng, rng.randrange(0, 30), width)
    index = _SpanIndex(spans)
    for _ in range(300):
        start = rng.randrange(-2, width + 2)
        end = start + rng.randrange(0, 8)
        assert index.contains(start, end) == _contains(spans, start, end), (spans, start, end)
        assert index.covers(start) == _covers(spans, start), (spans, start)


def test_the_edges_of_a_span():
    """Closed for a match inside it, half-open for a start it covers."""
    index = _SpanIndex([(3, 7)])
    assert index.contains(3, 7) and index.contains(4, 5)
    assert not index.contains(2, 5) and not index.contains(5, 8)
    assert index.covers(3) and index.covers(6) and not index.covers(7) and not index.covers(2)
    assert not _SpanIndex([]).contains(0, 0) and not _SpanIndex([]).covers(0)


def test_adjacent_spans_do_not_hold_a_match_across_them():
    """A union of spans is not a span: [0, 5] and [5, 10] hold [2, 4] and
    [6, 9], and not [4, 6]."""
    index = _SpanIndex([(5, 10), (0, 5)])
    assert index.contains(2, 4) and index.contains(6, 9)
    assert not index.contains(4, 6)


_LONG_LAW = "+".join(f"if(x{i}>0,1,0)" for i in range(60_000))


def _timed(fn, *args):
    began = time.perf_counter()
    out = fn(*args)
    return time.perf_counter() - began, out


@pytest.mark.parametrize("outside", [False, True])
def test_a_long_law_is_read_in_time(outside):
    """60,000 conditions, a comparison in each: 3.6e9 span checks for the
    product, which took 0.41 s at 4,000 and four times as long at each
    doubling, so about 90 s here; by bisection, under half a second. And the
    answer: every comparison is in a condition, unless one is added outside.

    Timed by the clock, as the other scaling tests here are: pytest-timeout is
    not installed, so a ``timeout`` mark is an unknown mark and holds nothing
    to any time."""
    seconds, found = _timed(_compares_outside_an_if, _LONG_LAW + ("+(y>1)" if outside else ""))
    assert found is outside
    assert seconds < 10.0, f"{seconds:.1f} s"


def test_the_scan_the_issue_named_is_read_in_time():
    """``uncompensated_condition_reason``, where #1256's build spent its time,
    on the same law with one comparison outside every condition: it is the
    last match, so every one before it is placed first, and it is named.
    Nothing past that scan is reached, so the scope holds nothing."""
    scope = SwitchConditionScope(
        core=None,
        clocks={},
        clock_symbols=frozenset(),
        param_names=(),
        param_pats={},
        primary_names=frozenset(),
        param_idx={},
        values=(),
        derived_exprs={},
        run_constants=frozenset(),
        function_names=frozenset(),
    )
    seconds, why = _timed(uncompensated_condition_reason, _LONG_LAW + "+(y>1)", scope)
    assert why is not None and why.startswith("the comparison '>' in ")
    assert seconds < 10.0, f"{seconds:.1f} s"


_NET = """begin parameters
    1 k 0.1
end parameters
begin species
    1 X() 5
end species
begin groups
    1 Xo 1
end groups
begin functions
    1 f() {law}
end functions
begin reactions
    1 1 0 f #_R1
end reactions
"""


def test_a_comparison_written_many_times_is_skipped_once_per_copy(tmp_path):
    """``state_switch_root_conditions`` hands the call scan the spans of the
    comparisons, whose calls are their own roots' business. It found each
    atom's every copy for each entry of it, and ``Xo>3`` written 2,000 times
    is 4,000 entries (as a condition and as a comparison), so 8 million spans,
    which the index sorted for each family of call: 14 s on an Intel Mac, for
    a run with no sensitivities at all, where the linear scan it replaced took
    4 s. Each atom is found once now, its copies are 2,000 spans, and the
    whole call takes 0.1 s."""
    path = tmp_path / "m.net"
    path.write_text(_NET.format(law="+".join(["if(Xo>3,k,2*k)"] * 2000) + "+max(Xo-1,0)"))
    model = bngsim.Model.from_net(str(path))
    seconds, roots = _timed(state_switch_root_conditions, model._core)
    assert roots == ("Xo>3", "(Xo-1)<(0)")
    assert seconds < 2.0, f"{seconds:.1f} s"
