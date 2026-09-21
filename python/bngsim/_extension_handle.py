"""What objects backed by the compiled extension do under pickle and copy.

:class:`bngsim.Result`, :class:`bngsim.Model` and :class:`bngsim.Simulator` each
hold a handle into ``bngsim._bngsim_core``. A pybind11 object has no pickle
support, so none of the three can be serialized — which is fine, and is not the
problem. The problem is what a caller was told: ``cannot pickle
'bngsim._bngsim_core.ResultCore' object`` names a type they have never seen and
says nothing about what to do instead, while the caller is a fitting or scanning
harness returning work from a process pool and the way through is to convert
first (issue #636).

Pickling and copying share one hook, so both live here. ``copy.copy`` falls back
to ``__reduce_ex__``, which defers to ``__reduce__`` when a class overrides it,
so a refusal written there would also refuse a *shallow* copy — which works
today and is left working: the classes are ``__slots__``-based, and
:func:`shallow_copy` reproduces what the default did. ``copy.deepcopy`` reaches
the refusal and is meant to: it failed before too, with the message this module
exists to replace.
"""

from __future__ import annotations

from typing import Any, NoReturn, TypeVar

T = TypeVar("T")


def raise_cannot_pickle(what: str, remedy: str) -> NoReturn:
    """Refuse to serialize an extension-backed object, and say what to do.

    *remedy* is the class's own sentence: which conversion carries the part of
    it that a worker process actually needs.
    """
    raise TypeError(
        f"cannot pickle a {what}: it holds a handle into bngsim's compiled extension, so "
        f"it cannot cross a process boundary or be deep-copied. {remedy}"
    )


def shallow_copy(obj: T) -> T:
    """A shallow copy of a ``__slots__`` object, as ``copy.copy`` used to make.

    The copy shares every value the original holds, the extension handle
    included — the semantics ``copy.copy`` has always had here, reproduced
    rather than changed, because a refusal on ``__reduce__`` would otherwise
    take it away as a side effect.
    """
    new = object.__new__(type(obj))
    for klass in type(obj).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            try:
                value: Any = getattr(obj, slot)
            except AttributeError:
                continue  # an unset slot stays unset, as it would in a default copy
            setattr(new, slot, value)
    return new
