"""The shared pickle wrapper's answers, and the ones it used to conflate (#646).

``_ndarray_pickle.unwrap_state`` is asked two questions at once — "did bngsim
write this state?" and "did *this class* write it?" — and it used to answer both
with one equality test against the caller's own tag. A tag it did not recognise
(a sibling class's, or its own from before a rename) therefore came back as "not
ours at all", which the callers read as *a stream older than the tagging* and
handed to numpy. numpy can only report that as a shape complaint::

    TypeError: __setstate__() argument 1, item 0 must be tuple, not str

— for a stream that bngsim itself wrote and could have named. Nothing reaches it
through ordinary pickling, since pickle pairs a state with the class that
reduced it, so this was filed as a latent bug against the diagnostic path the
version check already builds: a newer stream is refused by name, and a
differently-tagged one should be too.

Also covered, from @Montana's review of #641:
- The version refusal says which direction the mismatch runs.
- ``tagged`` is a flag, not a ``None`` payload, so a payload that legitimately
  *is* ``None`` is not read as a pre-tag stream.
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest
from bngsim import JacobianMatrix, NamedArray
from bngsim._ndarray_pickle import unwrap_state

DATA = np.array([[-0.1, 0.0], [0.1, 0.0]])
TAG = "bngsim.JacobianMatrix"


# ── A tagged state this class cannot read is refused by name ─────────


@pytest.mark.parametrize(
    ("writer", "reader", "written_tag", "reader_tag"),
    [
        (
            lambda: NamedArray(DATA.copy(), ["a", "b"]),
            lambda: JacobianMatrix(DATA.copy(), "analytical"),
            "bngsim.NamedArray",
            "bngsim.JacobianMatrix",
        ),
        (
            lambda: JacobianMatrix(DATA.copy(), "analytical"),
            lambda: NamedArray(DATA.copy(), ["a", "b"]),
            "bngsim.JacobianMatrix",
            "bngsim.NamedArray",
        ),
    ],
    ids=["NamedArray-state-into-JacobianMatrix", "JacobianMatrix-state-into-NamedArray"],
)
def test_a_sibling_classes_state_is_refused_by_name(writer, reader, written_tag, reader_tag):
    """The reproduction from the issue: no rename needed, the two classes on the
    shared wrapper already write tags the other does not read."""
    _reconstruct, _args, foreign_state = writer().__reduce__()
    assert foreign_state[0] == written_tag

    with pytest.raises(ValueError) as excinfo:
        reader().__setstate__(foreign_state)

    message = str(excinfo.value)
    # Both tags, so the reader can see what it got and what it wanted.
    assert written_tag in message
    assert reader_tag in message
    # And it says which thing this is not, because that was the confusion.
    assert "pre-tag stream" in message


def test_a_tag_from_a_renamed_build_is_refused_too():
    """The case the format-pin tests guard at development time, met at runtime."""
    with pytest.raises(ValueError, match="bngsim.JacobianMatrixV2"):
        JacobianMatrix(DATA.copy(), "analytical").__setstate__(
            ("bngsim.JacobianMatrixV2", 1, "analytical", None)
        )


def test_the_refusal_is_not_the_shape_error_it_used_to_be():
    """Pinned as its own fact: a TypeError from inside numpy is what this
    replaced, and a regression would look exactly like it coming back."""
    _reconstruct, _args, foreign = NamedArray(DATA.copy(), ["a", "b"]).__reduce__()
    with pytest.raises(ValueError):  # specifically NOT TypeError
        JacobianMatrix(DATA.copy(), "analytical").__setstate__(foreign)


# ── ...without refusing what it should still accept ──────────────────


def test_a_stream_older_than_the_tagging_still_loads_unlabeled():
    """The passthrough the refusal must not swallow: numpy's own state."""

    class _LegacyPickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, (JacobianMatrix, NamedArray)):
                return np.ndarray.__reduce__(obj)
            return NotImplemented

        @classmethod
        def dumps(cls, obj):
            buf = io.BytesIO()
            cls(buf).dump(obj)
            return buf.getvalue()

    back_j = pickle.loads(_LegacyPickler.dumps(JacobianMatrix(DATA.copy(), "analytical")))
    assert back_j.source == ""
    np.testing.assert_array_equal(np.asarray(back_j), DATA)

    back_n = pickle.loads(_LegacyPickler.dumps(NamedArray(DATA.copy(), ["a", "b"])))
    assert back_n.colnames == []
    np.testing.assert_array_equal(np.asarray(back_n), DATA)


def test_a_four_tuple_that_is_not_ours_is_left_to_numpy():
    """The namespace is what keeps the refusal narrow.

    numpy's state is its own to change, and a 4-tuple from numpy must go on
    reaching numpy rather than being refused as a foreign bngsim tag.
    """
    tagged, payload, inner = unwrap_state((1, (2, 2), "f8", False), TAG, 1, "JacobianMatrix")
    assert tagged is False
    assert payload is None
    assert inner == (1, (2, 2), "f8", False)


# ── The two answers that used to be one (@Montana, on #641) ──────────


def test_a_payload_that_is_none_is_not_read_as_an_untagged_stream():
    """``tagged`` is a flag rather than a ``None`` payload.

    Neither class writes a ``None`` payload today, so this is the trap a third
    class on the wrapper would fall into: its legitimate ``None`` would come
    back indistinguishable from a stream written before the tagging.
    """
    tagged, payload, _inner = unwrap_state((TAG, 1, None, ("numpy", "state")), TAG, 1, "J")
    assert tagged is True
    assert payload is None

    untagged, payload2, _ = unwrap_state(("numpy", "state"), TAG, 1, "J")
    assert untagged is False
    assert payload2 is None
    # Same payload, different answer — which is the whole point.


@pytest.mark.parametrize(
    ("got", "must_say"),
    [(2, "newer bngsim"), (99, "newer bngsim"), (0, "older bngsim")],
    ids=["one-ahead", "far-ahead", "behind"],
)
def test_the_version_refusal_says_which_direction_the_mismatch_runs(got, must_say):
    """It used to say "written by a newer bngsim" for either direction, which
    sends the reader of a dropped-support stream the wrong way."""
    with pytest.raises(ValueError) as excinfo:
        unwrap_state((TAG, got, "analytical", None), TAG, 1, "JacobianMatrix")

    message = str(excinfo.value)
    assert must_say in message
    assert f"version {got!r}" in message
    if got < 1:
        assert "no longer reads" in message
    else:
        assert "upgrade to read it" in message
