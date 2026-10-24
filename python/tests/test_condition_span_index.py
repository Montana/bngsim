"""Whether a match lies inside an ``if()`` condition is answered by bisection
(issue #1256).

Four scans in ``_switch_sensitivity`` asked it of every match of a pattern
against every condition span, ``any(lo <= start and end <= hi for ...)``: a
product of the two counts. A rate law written out through a diamond of rules
holds thousands of both, and building a sensitivity simulator for #1145's size
jump took 29 s at 16 rules and 298 s at 20, 26.8 s of the 16's in that one
``any``, evaluated 74 million times.

``_SpanIndex`` answers the same two questions from the spans sorted by start
and the running maximum of their ends: some span holds ``[start, end]`` exactly
when, of the spans that start at or before ``start``, the furthest end is at or
after ``end``. The property tests below hold it to the brute-force answer on
spans nested, overlapping, adjacent, repeated and empty.
"""

from __future__ import annotations

import random

import pytest
from bngsim._switch_sensitivity import _compares_outside_an_if, _SpanIndex


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


@pytest.mark.timeout(30)
@pytest.mark.parametrize("outside", [False, True])
def test_a_long_law_is_read_in_time(outside):
    """60,000 conditions, a comparison in each: 3.6e9 span checks for the
    product, which took 0.41 s at 4,000 and four times as long at each
    doubling, so about 90 s here; by bisection, a second or so. And the
    answer: every comparison is in a condition, unless one is added outside."""
    text = "+".join(f"if(x{i}>0,1,0)" for i in range(60_000))
    if outside:
        text += "+(y>1)"
    assert _compares_outside_an_if(text) is outside
