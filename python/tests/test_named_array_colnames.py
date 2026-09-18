"""Issue #561 — a NamedArray's column names always describe its own columns.

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
  transpose, in-row sort/roll/take, copy, pickle) carry no names at all, so a
  lookup raises rather than resolving against a column that moved.
- The invariant: ``colnames`` is empty or one name per column, never stale.
"""

from __future__ import annotations

import copy
import pickle

import bngsim
import numpy as np
import pytest
from bngsim import NamedArray

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
    ("copy", lambda a: a.copy()),
    ("pickled", lambda a: pickle.loads(pickle.dumps(a))),
    ("deep-copied", lambda a: copy.deepcopy(a)),
    ("single row", lambda a: a[0]),
    ("single column", lambda a: a[:, 0]),
    ("flattened", lambda a: a.reshape(-1)),
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
