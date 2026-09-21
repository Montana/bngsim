"""bngsim.NamedArray — RoadRunner-compatible labeled ndarray.

Mirrors the shape of :class:`roadrunner.NamedArray`: a 2-D
:class:`numpy.ndarray` subclass that carries column names and
supports column lookup by string. Used by
:meth:`bngsim.Result.as_roadrunner` to provide a drop-in replacement
for ``rr.simulate(...)`` output in PyBNF-style stochastic-fitting
workflows.

See ``dev/plans/SBML_SSA_SUPPORT_PLAN.md`` Phase 4.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

#: Marks a pickle state written by :meth:`NamedArray.__reduce__`. A stream
#: written before issue #629 carries numpy's own state and no names, and
#: :meth:`NamedArray.__setstate__` tells the two apart by this tag rather than
#: by the state's shape, which is numpy's to change.
_PICKLE_TAG = "bngsim.NamedArray"
_PICKLE_VERSION = 1


class NamedArray(np.ndarray):
    """A 2-D ndarray with named columns (RoadRunner-compatible).

    Constructed via :meth:`bngsim.Result.as_roadrunner`. End users
    rarely instantiate it directly; subclass primarily exists so that
    callers can identify the array shape (``arr.colnames``) and look
    up columns by name (``arr["[X]"]``, ``arr[:, "[X]"]``).

    Attributes
    ----------
    colnames : list[str]
        One name per column, in column order — the invariant
        :meth:`__new__` checks, and one that holds for the lifetime of
        every array of this class (issue #561). Names are *positional*,
        so they are carried only where the columns they label can be
        followed: construction; indexing, which relabels whatever
        survives the key (``arr[:, 1:]``, ``arr[:, ["time", "[X]"]]``,
        ``arr[:, ::-1]``, a row slice); and a pickle round trip, which
        rebuilds the same columns (issue #629), so an array that crosses
        a process boundary arrives labeled. Every other derived array —
        a product, a transpose, a reduction, ``copy()``,
        ``copy.deepcopy()`` — has ``colnames == []`` and raises
        :class:`KeyError` on a lookup by name, rather than answering
        with whichever column now sits at the name's old position.
        libroadrunner drops the labels on every derived array, slices
        included, and likewise keeps them through pickle.

    Examples
    --------
    >>> result = sim.run(t_span=(0, 10), n_points=11)
    >>> arr = result.as_roadrunner()
    >>> arr.colnames
    ['time', '[X]', '[Y]']
    >>> arr["[X]"]                      # 1-D column view
    array([ ... ])
    >>> arr[:, "[X]"]                   # equivalent
    array([ ... ])
    >>> arr[:, 1:].colnames             # the columns the slice kept
    ['[X]', '[Y]']
    """

    # __slots__ omitted: ndarray subclasses must not declare __slots__
    # (numpy stores subclass attributes via __array_finalize__).

    def __new__(cls, data: NDArray[np.float64], colnames: list[str]) -> NamedArray:
        arr = np.asarray(data, dtype=np.float64).view(cls)
        if arr.ndim != 2:
            raise ValueError(f"NamedArray must be 2-D, got shape {arr.shape}")
        if arr.shape[1] != len(colnames):
            raise ValueError(
                f"colnames length {len(colnames)} does not match number of columns {arr.shape[1]}"
            )
        arr.colnames = list(colnames)
        return arr

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        # Deliberately does *not* inherit the parent's names (issue #561).
        # They label columns by position, and numpy hands this hook no
        # description of the transform that produced ``self``: a result of the
        # parent's shape is as likely to be ``np.sort(parent, axis=1)``, with
        # the columns permuted, as ``parent * 2``, with them intact. Inheriting
        # them is what let a column-dropping slice keep one name per column of
        # the *parent*, so that ``arr[:, 1:]["[X]"]`` returned the ``[Y]``
        # column and nothing raised. Names are attached only where the
        # surviving columns are known: __new__ and __getitem__.
        self.colnames: list[str] = []

    def __reduce__(self) -> tuple[Any, ...]:
        # numpy's reduction carries the array and none of a subclass's
        # attributes, so the names used to be lost on every round trip through a
        # process boundary: a fit collecting worker results got its tables back
        # unlabeled, and every lookup by name raised (issue #629). Wrapping
        # numpy's state rather than serializing the array here leaves the buffer
        # handling to numpy. pickle calls ``__reduce_ex__``, which numpy
        # implements and which routes a SUBCLASS through this method at every
        # protocol — the no-state ``_frombuffer`` form numpy uses for a base
        # ndarray at protocol 5 cannot rebuild a subclass — and the round trip
        # is pinned at each protocol in test_named_array_colnames.py rather than
        # guessed at here.
        reconstruct, args, state = cast("tuple[Any, Any, Any]", super().__reduce__())
        return reconstruct, args, (_PICKLE_TAG, _PICKLE_VERSION, list(self.colnames), state)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 4 and state[0] == _PICKLE_TAG:
            _tag, version, colnames, inner = state
            if version != _PICKLE_VERSION:
                raise ValueError(
                    f"NamedArray pickle state version {version!r} is not supported by this "
                    f"build (it writes and reads version {_PICKLE_VERSION}). The stream was "
                    "written by a newer bngsim; upgrade to read it."
                )
            self.colnames = list(colnames)
            super().__setstate__(inner)
            return
        # Written before #629: numpy's own state, carrying no names. The array
        # still loads; it simply arrives unlabeled, as it did then.
        self.colnames = []
        super().__setstate__(state)

    def __getitem__(self, key: Any) -> NDArray[np.float64]:  # type: ignore[override]
        # Forms supported:
        #   arr["name"]         → 1-D column   (RR convention)
        #   arr[:, "name"]      → 1-D column
        #   arr[i, "name"]      → scalar
        #   arr[i:j, ["a","b"]] → 2-D NamedArray slice
        # Anything else falls through to ndarray.__getitem__, and the names of
        # the columns that survived the key are computed back onto the result.
        if isinstance(key, str):
            return self._col_by_name(key)
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(col_key, str):
                idx = self._col_index(col_key)
                return np.asarray(self).__getitem__((row_key, idx))
            if isinstance(col_key, list) and col_key and isinstance(col_key[0], str):
                idxs = [self._col_index(name) for name in col_key]
                sub = np.asarray(self).__getitem__((row_key, idxs))
                if sub.ndim == 2:
                    return NamedArray(sub, [self.colnames[i] for i in idxs])
                return sub
        out = super().__getitem__(key)
        if isinstance(out, NamedArray):
            out.colnames = self._colnames_after(key, out)
        return out

    def _colnames_after(self, key: Any, out: NamedArray) -> list[str]:
        """Names for the columns of ``self[key]``; ``[]`` when not derivable.

        Whatever selects along axis 1 of a 2-D array selects the same entries
        of a 1-D array of column positions, so basic and advanced indexing are
        both answered by indexing ``np.arange(ncols)`` with the key's column
        part. A key that leaves no recognizable column axis — one whose result
        is not 2-D, or a broadcast advanced index whose axis 1 is not the
        column axis — yields no names rather than a guess.
        """
        if out.ndim != 2 or not self.colnames:
            return []
        col_key: Any = slice(None)  # a key that indexes rows only keeps every column
        if isinstance(key, tuple):
            if len(key) > 2:
                return []
            if len(key) == 2:
                col_key = key[1]
        try:
            idxs = np.arange(len(self.colnames))[col_key]
        except (IndexError, TypeError, ValueError):
            return []
        if not isinstance(idxs, np.ndarray) or idxs.ndim != 1 or idxs.size != out.shape[1]:
            return []
        return [self.colnames[int(i)] for i in idxs]

    def _col_index(self, name: str) -> int:
        try:
            return self.colnames.index(name)
        except ValueError:
            raise KeyError(self._unknown_selector_message(name)) from None

    def _col_by_name(self, name: str) -> NDArray[np.float64]:
        return np.asarray(self)[:, self._col_index(name)]

    def _unknown_selector_message(self, name: str) -> str:
        # Match RoadRunner's selector-not-found text closely enough that
        # PyBNF code that catches RR errors keeps working.
        if not self.colnames:
            return (
                f"Invalid selection '{name}'. This array carries no column names — it was "
                "either built without any, or derived by an operation whose effect on the "
                "columns is not tracked (arithmetic, a transpose, a reduction, unpickling), "
                "which drops the names rather than leaving them pointing at columns that may "
                "have moved (issue #561). Look the column up on the array "
                "Result.as_roadrunner() returned, or index that array by name "
                "(arr[:, ['time', '[X]']]) to carry the names over."
            )
        return f"Invalid selection '{name}'. Valid selections: {self.colnames}"

    def __repr__(self) -> str:
        body = np.array2string(np.asarray(self), separator=", ")
        return f"NamedArray(\n{body},\ncolnames={self.colnames})"
