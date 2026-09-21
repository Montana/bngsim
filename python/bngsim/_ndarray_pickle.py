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

Wrap ``__reduce__`` rather than ``__reduce_ex__``: pickle calls the latter,
numpy implements it, and it routes a subclass through ``__reduce__`` at every
protocol — the no-state ``_frombuffer`` form numpy uses for a base ndarray at
protocol 5 cannot rebuild a subclass. Each class pins that per protocol in its
own tests rather than trusting the sentence.
"""

from __future__ import annotations

from typing import Any


def wrap_state(
    reduction: tuple[Any, ...], tag: str, version: int, payload: Any
) -> tuple[Any, ...]:
    """numpy's *reduction* with *payload* tagged onto its state."""
    reconstruct, args, state = reduction[0], reduction[1], reduction[2]
    return (reconstruct, args, (tag, version, payload, state), *reduction[3:])


def unwrap_state(state: Any, tag: str, version: int, owner: str) -> tuple[Any, Any]:
    """Split a state into ``(payload, numpy_state)``.

    *payload* is ``None`` when *state* is numpy's own — a stream written before
    this class carried its attribute — which the caller answers with whatever
    an unlabeled instance should hold. The tag is what tells the two apart:
    the length and contents of numpy's state are numpy's to change, and a
    reader that guessed from shape would misread the day they did.
    """
    if isinstance(state, tuple) and len(state) == 4 and state[0] == tag:
        _tag, got, payload, inner = state
        if got != version:
            raise ValueError(
                f"{owner} pickle state version {got!r} is not supported by this build "
                f"(it writes and reads version {version}). The stream was written by a "
                "newer bngsim; upgrade to read it."
            )
        return payload, inner
    return None, state
