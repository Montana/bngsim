"""Issues #561 and #629 — a NamedArray's names describe its own columns, and
survive being sent somewhere.

``NamedArray.colnames`` is a positional label list: entry *j* names column *j*.
``__new__`` checks that once, and nothing checked it again, so every array numpy
derived from a labeled one inherited the parent's names verbatim. A
column-dropping slice — ``arr[:, 1:]``, on the plain-numpy path
``Result.as_roadrunner``'s docstring advertises — came back with one name too
many, and a lookup by name then answered with whichever column had moved into
that position: ``arr[:, 1:]["[X]"]`` returned the ``[Y]`` trajectory, and only
the last name happened to run off the end and raise.

Coverage:
- The reported repro, on a bare NamedArray and through the public
  ``Model.from_sbml_string`` → ``Simulator.run`` → ``as_roadrunner`` workflow.
- Indexing relabels: column slices, reversals, integer and boolean column
  selections, row-only keys, advanced-index broadcasts, chained slices.
- Transforms whose effect on the columns numpy does not report (arithmetic,
  transpose, in-row sort/roll/take, copy, deepcopy) carry no names at all, so a
  lookup raises rather than resolving against a column that moved.
- Why a *copy* is in that list although its columns are the parent's:
  ``.copy()`` is a numpy primitive (``np.sort(a, axis=1)`` is a copy plus an
  in-place sort), so a copy that carried the names would label each row's order
  statistics with them. The test measures the wrong answer rather than leaving
  the reason to a comment.
- The invariant: ``colnames`` is empty or one name per column, never stale.
- A pickle round trip keeps the names the array actually has, at every protocol
  (#629): numpy's reduction carried the array and dropped the subclass
  attribute, so a table collected from a worker process arrived unlabeled.
  Includes the two compatibility directions — a stream written before that fix
  still loads, and a state version this build cannot read is refused.
"""

from __future__ import annotations

import copy
import io
import pickle

import bngsim
import numpy as np
import pytest
from bngsim import NamedArray
from bngsim._named_array import _PICKLE_TAG, _PICKLE_VERSION

# Column values are disjoint between columns, so "this name still resolves to
# the data it named" is checkable by membership rather than by position.
COLNAMES = ["time", "[X]", "[Y]"]
DATA = np.array(
    [
        [0.0, 100.0, 7.0],
        [1.0, 90.0, 17.0],
        [2.0, 80.0, 27.0],
    ]
)


#: Deliberately out of order in every column, for the sort case below.
DATA_UNSORTED = np.array(
    [
        [3.0, 1.0, 2.0],
        [9.0, 8.0, 7.0],
        [4.0, 6.0, 5.0],
    ]
)


def _arr() -> NamedArray:
    return NamedArray(DATA.copy(), list(COLNAMES))


# ── The reported defect ───────────────────────────────────────────────


def test_column_slice_relabels_instead_of_keeping_stale_names():
    arr = _arr()
    sub = arr[:, 1:]
    assert sub.shape == (3, 2)
    assert sub.colnames == ["[X]", "[Y]"]
    # Pre-fix these returned the [Y] column and the [X] column respectively.
    np.testing.assert_array_equal(sub["[X]"], arr["[X]"])
    np.testing.assert_array_equal(sub["[Y]"], arr["[Y]"])
    # 'time' was dropped by the slice, so it must not resolve to anything.
    with pytest.raises(KeyError, match="Invalid selection 'time'"):
        _ = sub["time"]


def test_public_workflow_slice_keeps_the_species_it_named():
    """The defect was reachable from the documented as_roadrunner path."""
    result = bngsim.Simulator(bngsim.Model.from_sbml_string(_DECAY_2SP), method="ode").run(
        t_span=(0, 10), n_points=5
    )
    arr = result.as_roadrunner()
    assert arr.colnames == ["time", "[X]", "[Y]"]

    sub = arr[:, 1:]
    assert sub.colnames == ["[X]", "[Y]"]
    np.testing.assert_array_equal(sub["[X]"], arr["[X]"])
    # X decays from 100 into Y; a mislabeled column is off by more than a
    # tolerance rather than by a rounding step.
    assert sub["[X]"][0] == pytest.approx(100.0)
    assert sub["[Y]"][0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(KeyError, match="Invalid selection 'time'"):
        _ = sub["time"]


# ── Indexing: the names follow the columns that survive the key ───────

# (label, indexing operation, expected colnames)
INDEXING_CASES = [
    ("trailing column slice", lambda a: a[:, 1:], ["[X]", "[Y]"]),
    ("leading column slice", lambda a: a[:, :2], ["time", "[X]"]),
    ("reversed columns", lambda a: a[:, ::-1], ["[Y]", "[X]", "time"]),
    ("integer column list", lambda a: a[:, [2, 0]], ["[Y]", "time"]),
    ("boolean column mask", lambda a: a[:, np.array([False, True, True])], ["[X]", "[Y]"]),
    ("column names", lambda a: a[:, ["time", "[Y]"]], ["time", "[Y]"]),
    ("row slice", lambda a: a[1:], COLNAMES),
    ("row list", lambda a: a[[0, 2]], COLNAMES),
    ("row mask", lambda a: a[np.asarray(a)[:, 0] > 0], COLNAMES),
    ("ellipsis", lambda a: a[...], COLNAMES),
    ("everything", lambda a: a[:], COLNAMES),
    ("rows then ellipsis", lambda a: a[1:, ...], COLNAMES),
    ("ellipsis then columns", lambda a: a[..., 1:], ["[X]", "[Y]"]),
    ("broadcast advanced index", lambda a: a[[[0], [1]], [0, 2]], ["time", "[Y]"]),
    ("chained slices", lambda a: a[:, 1:][:, ::-1], ["[Y]", "[X]"]),
    ("no columns", lambda a: a[:, []], []),
]


@pytest.mark.parametrize(
    ("take", "expected"),
    [pytest.param(op, exp, id=label) for label, op, exp in INDEXING_CASES],
)
def test_indexing_relabels_the_surviving_columns(take, expected):
    arr = _arr()
    out = take(arr)
    assert isinstance(out, NamedArray)
    assert out.colnames == expected
    assert len(out.colnames) == out.shape[1]
    # Every name that survived still resolves to the data it named: the
    # columns are value-disjoint, so a shifted label cannot pass this.
    for name in out.colnames:
        assert np.isin(out[name], arr[name]).all()


def test_names_are_not_invented_for_a_slice_of_an_unlabeled_array():
    stripped = _arr() * 2.0  # names dropped (see below)
    assert stripped[:, 1:].colnames == []


# ── Transforms numpy does not describe: no names rather than wrong ones ──

UNTRACKED_CASES = [
    ("scaled", lambda a: a * 2.0),
    ("log10", lambda a: np.log10(a + 1.0)),
    # DATA is square, so a transpose keeps the column count: a check on the
    # length alone would let these names through.
    ("transposed", lambda a: a.T),
    ("swapaxes", lambda a: a.swapaxes(0, 1)),
    ("sorted within rows", lambda a: np.sort(a, axis=1)),
    ("rolled columns", lambda a: np.roll(a, 1, axis=1)),
    ("take on the column axis", lambda a: a.take([2, 1, 0], axis=1)),
    ("cumulative sum across columns", lambda a: np.cumsum(a, axis=1)),
    ("single row", lambda a: a[0]),
    ("single column", lambda a: a[:, 0]),
    ("flattened", lambda a: a.reshape(-1)),
    # Copies. Their columns ARE the parent's, so carrying the names across
    # looks sound and is not: `.copy()` is a numpy primitive — `np.sort(a,
    # axis=1)` is `a.copy(order="K")` plus an in-place sort — so a
    # name-preserving copy hands labels to operations that then permute the
    # columns. The "sorted within rows" case above is what fails when that is
    # attempted, and test_a_copy_that_carried_names_would_mislabel_a_sort
    # measures the wrong answer it would produce.
    ("copy", lambda a: a.copy()),
    ("copy.copy", lambda a: copy.copy(a)),
    ("deep-copied", lambda a: copy.deepcopy(a)),
    # Copies numpy makes through its own constructor, where there is no hook to
    # attach a name to even in principle. Pinned so the edge stays known.
    ("np.copy(subok=True)", lambda a: np.copy(a, subok=True)),
    ("np.array(subok=True)", lambda a: np.array(a, subok=True)),
    ("astype", lambda a: a.astype(np.float64)),
    ("view", lambda a: a.view(NamedArray)),
]


@pytest.mark.parametrize(
    "transform",
    [pytest.param(op, id=label) for label, op in UNTRACKED_CASES],
)
def test_untracked_transform_carries_no_names(transform):
    out = transform(_arr())
    assert out.colnames == []
    with pytest.raises(KeyError, match="Invalid selection"):
        _ = out["[X]"]


def test_untracked_transform_leaves_the_numbers_alone():
    """Only the labels are dropped — the array itself is untouched."""
    arr = _arr()
    np.testing.assert_array_equal(np.asarray(arr * 2.0), DATA * 2.0)
    np.testing.assert_array_equal(np.asarray(arr.T), DATA.T)


def test_unlabeled_lookup_says_why_and_what_to_do():
    with pytest.raises(KeyError) as excinfo:
        _ = (_arr() * 2.0)["[X]"]
    message = str(excinfo.value)
    assert "no column names" in message
    assert "as_roadrunner" in message


def test_a_copy_that_carried_names_would_mislabel_a_sort():
    """Why ``copy()`` has no override, in the form of the answer it would give.

    ``np.sort(a, axis=1)`` is a copy followed by an in-place sort, so a
    ``NamedArray.copy`` that carried the parent's names would put them on an
    array whose columns are now each row's order statistics: ``sorted["time"]``
    would answer with the row-wise minima. This measures both halves — that the
    sort does route through ``copy``, and what the labels would then be worth —
    so the reason survives in runnable form rather than as a comment nobody
    re-derives.
    """
    arr = NamedArray(DATA_UNSORTED.copy(), list(COLNAMES))
    calls = []
    original = NamedArray.copy

    def counting_copy(self, order="C"):
        calls.append(order)
        out = original(self, order)
        out.colnames = list(self.colnames)  # the override this test argues against
        return out

    NamedArray.copy = counting_copy
    try:
        mislabeled = np.sort(arr, axis=1)
    finally:
        NamedArray.copy = original

    assert calls, "np.sort no longer routes through NamedArray.copy"
    assert mislabeled.colnames == COLNAMES
    # The column under 'time' is now the row-wise minimum, not the time column.
    np.testing.assert_array_equal(mislabeled["time"], np.sort(DATA_UNSORTED, axis=1)[:, 0])
    assert not np.array_equal(mislabeled["time"], arr["time"])

    # And with no override, which is what ships: no names, so no wrong answer.
    assert np.sort(arr, axis=1).colnames == []


# ── Issue #629: the names survive a pickle round trip ────────────────


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_round_trip_at_every_protocol(protocol):
    """Every protocol, because numpy picks the reduction per protocol.

    A base ndarray at protocol 5 reduces through ``_frombuffer``, which has no
    state slot and could not carry a name if it were used; a subclass reduces
    through ``_reconstruct`` at every protocol, which is what routes pickling
    through ``NamedArray.__reduce__``. That routing is numpy's to change, so it
    is pinned here rather than assumed: a protocol that stopped carrying the
    names would fail this test instead of silently unlabeling a worker's result.
    """
    arr = _arr()
    back = pickle.loads(pickle.dumps(arr, protocol=protocol))
    assert isinstance(back, NamedArray)
    assert back.colnames == COLNAMES
    np.testing.assert_array_equal(np.asarray(back), DATA)
    # The names are attached to the right columns, not merely present.
    for name in COLNAMES:
        np.testing.assert_array_equal(back[name], arr[name])


def test_the_state_format_is_the_one_released_builds_write():
    """The wire format is a compatibility surface, not an implementation detail.

    A stream written by one build is read by another, so the positions below
    are fixed: ``(tag, version, payload, numpy's own state)``, names as the
    payload. The literals are spelled out rather than read from the module's
    constants — a test that imported them would follow a rename that breaks
    every pickle already written. Issue #635 moved this code into a wrapper
    shared with JacobianMatrix, which is exactly the kind of refactor that can
    move a format without meaning to.
    """
    arr = _arr()
    _reconstruct, _args, state = arr.__reduce__()
    tag, version, payload, inner = state
    assert (tag, version, payload) == ("bngsim.NamedArray", 1, COLNAMES)

    # And the reader takes that exact shape back, numpy's state included.
    fresh = NamedArray(np.zeros_like(DATA), ["a", "b", "c"])
    fresh.__setstate__((tag, version, list(COLNAMES), inner))
    assert fresh.colnames == COLNAMES
    np.testing.assert_array_equal(np.asarray(fresh), DATA)


def test_pickle_carries_the_names_a_derived_array_actually_has():
    """Not the parent's: a round trip preserves, it does not restore."""
    arr = _arr()
    sliced = pickle.loads(pickle.dumps(arr[:, 1:]))
    assert sliced.colnames == ["[X]", "[Y]"]
    np.testing.assert_array_equal(sliced["[X]"], arr["[X]"])

    scaled = pickle.loads(pickle.dumps(arr * 2.0))
    assert scaled.colnames == []
    with pytest.raises(KeyError, match="Invalid selection"):
        _ = scaled["[X]"]


def test_a_stream_written_before_the_fix_still_loads():
    """numpy's own state, which is what a pre-#629 build wrote.

    It has to keep loading — a saved pickle outlives the build that wrote it —
    and it arrives unlabeled, which is what that build would have given it.
    """
    arr = _arr()
    raw = _LegacyPickler.dumps(arr)
    back = pickle.loads(raw)
    assert isinstance(back, NamedArray)
    assert back.colnames == []
    np.testing.assert_array_equal(np.asarray(back), DATA)
    with pytest.raises(KeyError, match="Invalid selection"):
        _ = back["[X]"]


def test_an_unreadable_state_version_says_so():
    """A stream from a future format is refused by name, not misread."""
    with pytest.raises(ValueError, match="pickle state version"):
        _arr().__setstate__((_PICKLE_TAG, _PICKLE_VERSION + 1, ["time"], None))


class _LegacyPickler(pickle.Pickler):
    """Writes a NamedArray the way a build without ``__reduce__`` did."""

    def reducer_override(self, obj):
        if isinstance(obj, NamedArray):
            return np.ndarray.__reduce__(obj)
        return NotImplemented

    @classmethod
    def dumps(cls, obj):
        buf = io.BytesIO()
        cls(buf).dump(obj)
        return buf.getvalue()


# ── The construction-time invariant, unchanged ───────────────────────


def test_new_still_requires_one_name_per_column():
    with pytest.raises(ValueError, match="does not match number of columns"):
        NamedArray(DATA, ["time", "[X]"])
    with pytest.raises(ValueError, match="must be 2-D"):
        NamedArray(np.zeros(3), ["time"])


def test_named_lookups_on_the_constructed_array_are_unchanged():
    arr = _arr()
    np.testing.assert_array_equal(arr["time"], DATA[:, 0])
    np.testing.assert_array_equal(arr[:, "[X]"], DATA[:, 1])
    assert arr[1, "[Y]"] == DATA[1, 2]
    with pytest.raises(KeyError, match=r"Valid selections: \['time', '\[X\]', '\[Y\]'\]"):
        _ = arr["[Z]"]


# ── Reference SBML: X → Y, so the two species traces never coincide ───

_DECAY_2SP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="decay_2sp">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="Y" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="conv" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Y" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""
