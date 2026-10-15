"""Checks a BNG XML must pass before a network-free backend is handed it.

Issue #862: NFsim cannot prepare a function that reads ``time()`` and fails at
setup with ``Error preparing function f in class GlobalFunction``, which names
neither the cause nor an alternative. It is refused here, before the backend is
built, naming the functions. The clock is read under either spelling,
``time()`` or a bare ``time``, and both are refused.

RuleMonkey runs such a function since 3.11.0, which integrates a rate that reads
the clock between events (richardposner/RuleMonkey#86), and it runs on absolute
model time (issue #1022). So does a table function indexed by time
(``type="TFUN"`` with a clock ``ctrName``), which keeps its clock in an
attribute while its ``<Expression>`` reads a placeholder. NFsim runs one, but
refreshes its value only when a reaction fires, so a rate that is 0 at the
start is not seen (issue #892), and it reads elapsed time, BNG2.pl's
simulate_nf semantics. That is left to the fix there, since refusing it would
take away a table function NFsim reads correctly whenever reactions fire often.

A RuleMonkey session uses :func:`clock_readers` to refuse a segment whose labels
are not the clock the model reads.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bngsim._exceptions import ModelError

# `time()` or a bare `time`: the engine binds the clock under both spellings.
_TIME_REF = re.compile(r"(?<![A-Za-z0-9_])time(?![A-Za-z0-9_])")

# The counter names RuleMonkey reads as the clock for a table function, checked
# before any parameter or observable of that name (resolve_tfun_counter in
# third_party/rulemonkey/cpp/rulemonkey/simulator.cpp).
_RULEMONKEY_CLOCK_COUNTERS = frozenset({"time", "t", "time()", "t()"})

_BACKENDS = {
    "nfsim": (
        "NFsim (method='nf' / 'nf_reject')",
        "NFsim cannot prepare a function that reads time() and fails at setup",
    ),
}


def _functions(xml_path: str | Path) -> list[ET.Element] | None:
    """Every ``<Function>`` element in *xml_path*, or ``None`` when the file
    cannot be parsed: the backend then reports its own parse error, which is
    more specific than anything these checks could say."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    return [
        el
        for el in root.iter()
        if el.tag.endswith("Function") and not el.tag.endswith("ListOfFunctions")
    ]


def _reads_time(func: ET.Element) -> bool:
    return any(
        expr.tag.endswith("Expression") and _TIME_REF.search(expr.text or "") for expr in func
    )


def _indexed_by_time(func: ET.Element) -> bool:
    return (func.get("type") or "").lower() == "tfun" and (
        (func.get("ctrName") or "").strip() in _RULEMONKEY_CLOCK_COUNTERS
    )


def time_reading_functions(xml_path: str | Path) -> list[str]:
    """The ids of the functions in *xml_path* whose expression reads the clock,
    as ``time()`` or as a bare ``time``. Empty when the file cannot be parsed."""
    return [f.get("id", "?") for f in _functions(xml_path) or () if _reads_time(f)]


def time_indexed_table_functions(xml_path: str | Path) -> list[str]:
    """The ids of the table functions in *xml_path* whose counter RuleMonkey
    reads as the clock. Empty when the file cannot be parsed."""
    return [f.get("id", "?") for f in _functions(xml_path) or () if _indexed_by_time(f)]


def clock_readers(xml_path: str | Path) -> list[str]:
    """The ids of the functions in *xml_path* that read the clock, through
    ``time()``, a bare ``time``, or a table indexed by time. Empty when the file
    cannot be parsed."""
    return [
        f.get("id", "?")
        for f in _functions(xml_path) or ()
        if _reads_time(f) or _indexed_by_time(f)
    ]


def refuse_time_dependent_rate_laws(xml_path: str | Path, backend: str) -> None:
    """Raise :class:`ModelError` if *xml_path* has a function reading the clock.

    *backend* is ``"nfsim"``, the one network-free backend that cannot run one.
    """
    reads = time_reading_functions(xml_path)
    if not reads:
        return
    label, why = _BACKENDS[backend]
    raise ModelError(
        f"{label} cannot run a model whose functions read time(): "
        + ", ".join(f"'{n}'" for n in reads)
        + f". {why}. Simulate it with method='rm', or with method='ode' or "
        "method='ssa' on the generated network, which read time() as the model "
        "time (issue #862)."
    )
