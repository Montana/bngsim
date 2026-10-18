"""Run a clock-reading model on RuleMonkey on elapsed time (issue #1063).

Since #1022, ``method="rm"`` runs a model that reads the clock (``time()``, a
bare ``time``, or a table function indexed by time) on absolute model time, as
``ode`` and ``ssa`` do. RuleMonkey places each event at ``t + tau`` in doubles,
so from a large ``t_start`` the waiting times round away once the total
propensity times |t| nears 1e15, and the run is biased with no error: from
``t_start = 1e12``, ``A(s~0) -> A(s~1)`` at 0.5 from 1e5 copies left 22743 at
``1e12 + 2`` against the binomial mean 36788 (z = -291). A model that reads no
clock runs on elapsed time and is exact (#1060).

So a model that reads the clock through ``time()`` runs on elapsed time too,
and gets absolute time back through the model. :func:`offset_clock_xml` writes
a copy of the BNG XML in which every function reads ``time()`` as
``(time()+__bngsim_t0)``, and the caller sets ``__bngsim_t0`` to the run's
``t_start`` before it runs. RuleMonkey resolves a parameter from the model's
parameter table wherever an expression names it (``<ListOfReferences>`` lists
only observables for it). Output functions are evaluated through the same
rewritten expressions, so they report on absolute time as before.

A table function indexed by the clock is not rewritten, and a model with one
runs on model time as before (:func:`offsettable`). Its counter would have to
be a function that reads ``time()+__bngsim_t0``, and RuleMonkey 3.11.0 recurses
without end evaluating a table whose counter is a function: rebuild_eval_vars
reads the counter through update_eval_vars, whose generation stamp is set only
after the rebuild (engine.cpp), and the run fails with a stack overflow
(richardposner/RuleMonkey#111). Offset such a table too once that is fixed
and vendored.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from bngsim._exceptions import ModelError

# The parameter the run's t_start is written to.
OFFSET_PARAM = "__bngsim_t0"

# `time()` or a bare `time`, as the engine binds the clock under both spellings.
_TIME_CALL = re.compile(r"(?<![A-Za-z0-9_])time(?:\s*\(\s*\))?(?![A-Za-z0-9_(])")
# The counter names RuleMonkey reads as the clock for a table function
# (resolve_tfun_counter in third_party/rulemonkey/cpp/rulemonkey/simulator.cpp).
_CLOCK_COUNTERS = frozenset({"time", "t", "time()", "t()"})

_FUNCTION = re.compile(r"<Function\b(?P<attrs>[^>]*)>(?P<body>.*?)</Function>", re.DOTALL)
_EXPRESSION = re.compile(r"(<Expression>)(.*?)(</Expression>)", re.DOTALL)
_ATTR = re.compile(r'(\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*")(?P<value>[^"]*)(")')


def _offset_expression(text: str) -> str:
    return _TIME_CALL.sub(f"(time()+{OFFSET_PARAM})", text)


def _set_attr(attrs: str, name: str, value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return m.group(1) + value + m.group(4) if m.group("name") == name else m.group(0)

    return _ATTR.sub(repl, attrs)


def _attr(attrs: str, name: str) -> str | None:
    for m in _ATTR.finditer(attrs):
        if m.group("name") == name:
            return m.group("value")
    return None


def _table_file(xml_dir: Path, file_attr: str) -> str:
    """The table's data file as RuleMonkey would find it beside the original XML
    (``<xml_dir>/<file>``, then ``<xml_dir>/../<file>``), made absolute so the
    copy written elsewhere finds the same one."""
    path = Path(file_attr)
    if path.is_absolute():
        return file_attr
    for candidate in (xml_dir / path, xml_dir / ".." / path):
        if candidate.exists():
            return str(candidate.resolve())
    return str((xml_dir / path).resolve())


def offsettable(xml: str) -> bool:
    """Whether :func:`offset_clock_text` can run *xml* on elapsed time: no table
    function in it is indexed by the clock (see the module note)."""
    for m in _FUNCTION.finditer(xml):
        attrs = m.group("attrs")
        if (_attr(attrs, "type") or "").lower() == "tfun" and (
            _attr(attrs, "ctrName") or ""
        ).strip() in _CLOCK_COUNTERS:
            return False
    return True


def offset_clock_text(xml: str, xml_dir: Path) -> str:
    """*xml* with the model's clock read as ``time()+__bngsim_t0``. *xml* must be
    :func:`offsettable`."""
    if re.search(rf"\bid\s*=\s*\"{OFFSET_PARAM}\"", xml):
        raise ModelError(
            f"the BNG XML already has an element named {OFFSET_PARAM!r}, which bngsim "
            "uses to run a model that reads the clock on RuleMonkey (issue #1063). "
            "Rename it."
        )
    if not offsettable(xml):
        raise ValueError("a table function indexed by the clock is not rewritten")

    def function(m: re.Match[str]) -> str:
        attrs, body = m.group("attrs"), m.group("body")
        if (_attr(attrs, "type") or "").lower() == "tfun":
            # The copy is written elsewhere: its data file, where RuleMonkey
            # found it beside the original.
            file_attr = (_attr(attrs, "file") or "").strip()
            if file_attr:
                attrs = _set_attr(attrs, "file", _table_file(xml_dir, file_attr))
        body = _EXPRESSION.sub(
            lambda e: e.group(1) + _offset_expression(e.group(2)) + e.group(3), body
        )
        return f"<Function{attrs}>{body}</Function>"

    xml = _FUNCTION.sub(function, xml)

    parameter = f'<Parameter id="{OFFSET_PARAM}" type="Constant" value="0" expr="0"/>'
    if "<ListOfParameters>" in xml:
        xml = xml.replace("<ListOfParameters>", f"<ListOfParameters>\n      {parameter}", 1)
    elif "<ListOfParameters/>" in xml:
        listed = f"<ListOfParameters>\n      {parameter}\n    </ListOfParameters>"
        xml = xml.replace("<ListOfParameters/>", listed, 1)
    else:
        raise ModelError("the BNG XML has no <ListOfParameters> to add the clock's offset to")
    return xml


def offset_clock_xml(xml_path: str | Path) -> tuple[str, tempfile.TemporaryDirectory]:
    """Write :func:`offset_clock_text` of *xml_path* to a temporary directory,
    and return the file's path and the directory, which removes itself once
    the caller lets go of it."""
    source = Path(xml_path)
    text = offset_clock_text(source.read_text(encoding="utf-8"), source.resolve().parent)
    tmp = tempfile.TemporaryDirectory(prefix="bngsim-rm-clock-")
    out = Path(tmp.name) / source.name
    out.write_text(text, encoding="utf-8")
    return str(out), tmp
