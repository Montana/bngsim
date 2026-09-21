"""``JacobianMatrix.source`` survives a pickle round trip (issue #635).

``source`` says whether the matrix came from the model's closed form or from a
difference quotient, and a consumer weighs the numbers by it. numpy's
reduction carries the array and none of a subclass's attributes, and on
unpickling ``__array_finalize__`` is called with no parent, so the attribute
used to be missing entirely: reading it raised ``AttributeError``. Pickle is
how multiprocessing, ``concurrent.futures`` and ``joblib.Memory`` move and
cache an object, so every Jacobian that crossed a process boundary or came
back from a disk cache had lost its provenance.

The pattern is the one NamedArray's column names use (issue #629), and these
tests mirror test_named_array_colnames.py. None needs a model: the class is
pure Python over the array.
"""

from __future__ import annotations

import copy
import io
import pickle

import numpy as np
import pytest
from bngsim import JacobianMatrix
from bngsim._evaluators import ANALYTICAL, FINITE_DIFFERENCE

DATA = np.array([[-0.1, 0.0], [0.1, 0.0]])


@pytest.mark.parametrize("source", [ANALYTICAL, FINITE_DIFFERENCE])
@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_round_trip_at_every_protocol(protocol, source):
    """Every protocol, because numpy picks the reduction per protocol and a
    subclass reaching ``__reduce__`` at protocol 5 is numpy's to change."""
    J = JacobianMatrix(DATA, source)
    back = pickle.loads(pickle.dumps(J, protocol=protocol))
    assert isinstance(back, JacobianMatrix)
    assert back.source == source
    np.testing.assert_array_equal(np.asarray(back), DATA)


def test_non_contiguous_and_fortran_order_round_trip():
    raw = np.arange(16.0).reshape(4, 4)
    big = JacobianMatrix(raw, FINITE_DIFFERENCE)
    fortran = JacobianMatrix(np.asfortranarray(raw), FINITE_DIFFERENCE)
    assert fortran.flags.f_contiguous and not fortran.flags.c_contiguous
    for J in (big[::2, ::2], big.T, fortran):
        back = pickle.loads(pickle.dumps(J))
        assert isinstance(back, JacobianMatrix)
        assert back.source == FINITE_DIFFERENCE
        np.testing.assert_array_equal(np.asarray(back), np.asarray(J))


def test_copy_and_deepcopy_keep_the_source():
    """Unlike a NamedArray's column names, ``source`` is not positional — a
    transform of the matrix does not make it any less analytical — so every
    derived array inherits it, and copies were never affected."""
    J = JacobianMatrix(DATA, FINITE_DIFFERENCE)
    assert J.copy().source == FINITE_DIFFERENCE
    assert copy.copy(J).source == FINITE_DIFFERENCE
    assert copy.deepcopy(J).source == FINITE_DIFFERENCE


def test_an_array_built_with_no_parent_has_a_source():
    """The path unpickling takes: numpy builds the array from nothing and calls
    ``__array_finalize__(None)``. The attribute must exist, not raise."""
    bare = np.ndarray.__new__(JacobianMatrix, (2, 2))
    assert bare.source == ""


def test_a_stream_written_before_the_fix_still_loads():
    """numpy's own state, which is what a pre-#635 build wrote. It keeps
    loading, with ``source == ""`` — the value for provenance not recorded —
    rather than the AttributeError that build itself would have given."""
    raw = _LegacyPickler.dumps(JacobianMatrix(DATA, ANALYTICAL))
    back = pickle.loads(raw)
    assert isinstance(back, JacobianMatrix)
    assert back.source == ""
    np.testing.assert_array_equal(np.asarray(back), DATA)


def test_an_unreadable_state_version_says_so():
    """A stream from a future format is refused by name, not misread.

    The tag and the version-to-come are spelled out for the same reason the
    format test spells them out: read from the module, this test would follow a
    rename or a bump instead of noticing one, and it would go on passing while
    the thing it is guarding moved (issue #635).
    """
    J = JacobianMatrix(DATA, ANALYTICAL)
    with pytest.raises(ValueError, match="pickle state version"):
        J.__setstate__(("bngsim.JacobianMatrix", 2, ANALYTICAL, None))


def test_the_state_format_is_the_one_released_builds_write():
    """The wire format is a compatibility surface, not an implementation detail.

    The mirror of NamedArray's test of the same name, for the other class on the
    shared ``_ndarray_pickle`` wrapper. The positions are fixed — ``(tag,
    version, payload, numpy's own state)``, the provenance string as the payload
    — because a stream written by one build is read by another, and the literals
    are spelled out rather than imported: a test that read them from the module
    would follow a rename rather than catch it. That goes for the payload as
    well as the tag — ``ANALYTICAL``'s *value* is what lands in the stream, so
    it is written here as the string a released build wrote.

    That is not hypothetical. Before this test existed, renaming
    ``_PICKLE_TAG`` on JacobianMatrix left every test in this file and in
    test_named_array_colnames.py passing, while a stream pickled beforehand
    stopped loading — and not cleanly, because a tag the reader does not
    recognize falls through to the legacy-stream path and surfaces as numpy's
    ``__setstate__() argument 1, item 0 must be tuple, not str``. The same
    rename on NamedArray was caught at once, by its own copy of this test.
    Reported by Michael Mendy (@Montana) on issue #635.
    """
    J = JacobianMatrix(DATA, ANALYTICAL)
    _reconstruct, _args, state = J.__reduce__()
    tag, version, payload, inner = state
    assert (tag, version, payload) == ("bngsim.JacobianMatrix", 1, "analytical")

    # And the reader takes that exact shape back, numpy's state included.
    fresh = JacobianMatrix(np.zeros_like(DATA), FINITE_DIFFERENCE)
    fresh.__setstate__((tag, version, payload, inner))
    assert fresh.source == ANALYTICAL
    np.testing.assert_array_equal(np.asarray(fresh), DATA)


class _LegacyPickler(pickle.Pickler):
    """Writes a JacobianMatrix the way a build without ``__reduce__`` did."""

    def reducer_override(self, obj):
        if isinstance(obj, JacobianMatrix):
            return np.ndarray.__reduce__(obj)
        return NotImplemented

    @classmethod
    def dumps(cls, obj):
        buf = io.BytesIO()
        cls(buf).dump(obj)
        return buf.getvalue()
