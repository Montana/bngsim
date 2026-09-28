"""Checks a BNG XML must pass before a network-free backend is handed it.

Issue #862: a rate law that reads ``time()`` is wrong or refused on both
network-free backends. RuleMonkey treats a total propensity of 0 as absorbing,
so ``if(time()>=6, k, 0)`` never fires and the run returns the initial state at
every row with no warning (richardposner/RuleMonkey#86). NFsim cannot prepare
such a function and fails at setup with ``Error preparing function f in class
GlobalFunction``, which names neither the cause nor an alternative. Both are
refused here, before the backend is built, naming the functions.

A table function indexed by time (``type="TFUN"`` with ``ctrName="time"``) is
not affected: its clock is an attribute, and its ``<Expression>`` reads a
placeholder, not ``time()``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bngsim._exceptions import ModelError

_TIME_CALL = re.compile(r"(?<![A-Za-z0-9_])time\s*\(\s*\)")

_BACKENDS = {
    "rulemonkey": (
        "RuleMonkey (method='rm' / 'nf_exact')",
        "RuleMonkey treats a total propensity of 0 as absorbing, so a rate that is 0 at "
        "the start and turns on later never fires, and the run returns the initial "
        "state with no warning (richardposner/RuleMonkey#86)",
    ),
    "nfsim": (
        "NFsim (method='nf' / 'nf_reject')",
        "NFsim cannot prepare a function that reads time() and fails at setup",
    ),
}


def time_reading_functions(xml_path: str | Path) -> list[str]:
    """The ids of the functions in *xml_path* whose expression reads ``time()``.

    Empty when the file cannot be parsed: the backend then reports its own
    parse error, which is more specific than anything this could say.
    """
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return []
    found = []
    for func in root.iter():
        if not func.tag.endswith("Function") or func.tag.endswith("ListOfFunctions"):
            continue
        for expr in func:
            if expr.tag.endswith("Expression") and _TIME_CALL.search(expr.text or ""):
                found.append(func.get("id", "?"))
                break
    return found


def refuse_time_dependent_rate_laws(xml_path: str | Path, backend: str) -> None:
    """Raise :class:`ModelError` if *xml_path* has a function reading ``time()``.

    *backend* is ``"rulemonkey"`` or ``"nfsim"``.
    """
    names = time_reading_functions(xml_path)
    if not names:
        return
    label, why = _BACKENDS[backend]
    listed = ", ".join(f"'{n}'" for n in names)
    raise ModelError(
        f"{label} cannot run a model whose functions read time(): {listed}. {why}. "
        "Simulate it with method='ode' or method='ssa' on the generated network, which "
        "read time() as the model time (issue #862)."
    )
