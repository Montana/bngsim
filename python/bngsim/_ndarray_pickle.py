"""Carrying a subclass's own attribute through numpy's pickle protocol.

numpy reduces an ndarray to a reconstructor, its arguments and a state tuple
that describes the buffer. None of it carries what a *subclass* added, so an
``ndarray`` subclass that means to keep an attribute across a process boundary
has to wrap that reduction itself: :class:`bngsim.NamedArray` lost its column
names that way (issue #629) and :class:`bngsim.JacobianMatrix` lost the
provenance of its numbers (issue #635).

Both do the same four things — tag the state so it is recognisable, version it
so a format change is diagnosable, keep numpy's own state underneath, and read
a stream written before any of this existed — which is why the wrapper lives
here rather than twice. The format is::

    (tag, version, payload, <numpy's own state>)

and it is what a released bngsim already writes for ``NamedArray``, so the
shape above is fixed, not an implementation detail to re-choose.

A state is *ours* when it is a 4-tuple whose first item is a string in the
``bngsim.`` namespace. The namespace, not the exact tag, is what separates the
two questions a reader has to answer separately (issue #646): "did bngsim write
this?" and "did *this class* write it?". Answering both with one equality test
made a tag we do not recognise — a sibling class's, or our own after a rename —
indistinguishable from a pre-tag stream, so it was handed to numpy as if it were
numpy's own state and died there as ``__setstate__() argument 1, item 0 must be
tuple, not str``. Length alone would not do it either: the contents of numpy's
state are numpy's to change, and the namespace check is what keeps a future
4-tuple from numpy out of this branch.

Wrap ``__reduce__`` rather than ``__reduce_ex__``: pickle calls the latter,
numpy implements it, and it routes a subclass through ``__reduce__`` at every
protocol — the no-state ``_frombuffer`` form numpy uses for a base ndarray at
protocol 5 cannot rebuild a subclass. Each class pins that per protocol in its
own tests rather than trusting the sentence.
"""

from __future__ import annotations

from typing import Any

#: The namespace every tag this wrapper writes lives in. A 4-tuple state whose
#: first item is a string starting with this was written by some bngsim build;
#: anything else is numpy's own state, from a stream older than the tagging.
_TAG_NAMESPACE = "bngsim."


def wrap_state(
    reduction: tuple[Any, ...], tag: str, version: int, payload: Any
) -> tuple[Any, ...]:
    """numpy's *reduction* with *payload* tagged onto its state."""
    reconstruct, args, state, *rest = reduction
    return (reconstruct, args, (tag, version, payload, state), *rest)


def _is_tagged(state: Any) -> bool:
    """Whether *state* was written by this wrapper, in any build or class."""
    return (
        isinstance(state, tuple)
        and len(state) == 4
        and isinstance(state[0], str)
        and state[0].startswith(_TAG_NAMESPACE)
    )


def unwrap_state(state: Any, tag: str, version: int, owner: str) -> tuple[bool, Any, Any]:
    """Split a state into ``(tagged, payload, numpy_state)``.

    *tagged* is ``False`` only for a stream written before this class carried
    its attribute, where *payload* is meaningless and the caller supplies
    whatever an unlabeled instance should hold. It is a flag rather than a
    ``None`` payload because those are different answers: a future subclass
    whose payload legitimately *is* ``None`` would otherwise be read as a
    pre-tag stream and silently given the unlabeled default.

    A state this wrapper wrote but this class cannot read — a sibling's tag, or
    our own from before a rename — is refused by name rather than passed down to
    numpy, which can only report it as a tuple-shape error (issue #646).
    """
    if not _is_tagged(state):
        return False, None, state

    got_tag, got_version, payload, inner = state
    if got_tag != tag:
        raise ValueError(
            f"{owner} cannot read a pickle state tagged {got_tag!r}; this build reads "
            f"{tag!r}. The stream is a bngsim tagged state from another class or from a "
            "build that spelled this one differently — not a pre-tag stream, which "
            "carries numpy's own state and still loads."
        )
    if got_version != version:
        direction = (
            "The stream was written by a newer bngsim; upgrade to read it."
            if got_version > version
            else "The stream was written by an older bngsim, and this build no longer "
            "reads that version."
        )
        raise ValueError(
            f"{owner} pickle state version {got_version!r} is not supported by this "
            f"build (it writes and reads version {version}). {direction}"
        )
    return True, payload, inner
