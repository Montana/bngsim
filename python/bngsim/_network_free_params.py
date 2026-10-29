"""The model's parameter values, handed to a network-free backend (issue #1360).

NFsim and RuleMonkey are built from the BNG XML alone, so a ``Model.set_param``
reached neither: with the only phosphorylation rule switched off (``kph=0``),
SSA gave RP = 0 and nf 121, rm 127, with no warning. Each run now hands the
backend, as its own parameter overrides, every parameter the model holds at a
value other than the one it was loaded with. Both backends re-derive the
parameters an override feeds and the seed amounts that read them (NFsim, issue
#29; RuleMonkey, GH #44).

Other than the one it was loaded with, and not other than the XML's: BNG2.pl's
``writeXML`` prints ``value=`` as ``%.8g`` (``BNGOutput.pm``), so an unchanged
``NA = 6.02214076e23`` reads ``6.0221408e+23`` there, and a derived ``1/(NA*V)``
differs from the model's in its ninth digit. Compared with the XML, every such
parameter would be pushed, and a model nobody changed would run on other rate
constants than it ran on before, and than BNG2.pl's ``simulate_nf`` runs it on.
Compared with the XML's value as printed, a change in the ninth digit would be
dropped. The model's own values at load (``Model._loaded_params``) are exact.

Only the parameters the XML declares are pushed: a backend refuses a name it
does not have.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def xml_parameter_names(xml_path: str | Path) -> frozenset[str]:
    """The ids of the model's ``<Parameter>`` elements in *xml_path*. Empty
    when the file cannot be parsed."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return frozenset()
    return frozenset(
        name
        for lst in root.iter()
        if _local(lst.tag) == "ListOfParameters"
        for p in lst
        if _local(p.tag) == "Parameter" and (name := p.get("id"))
    )


def parameter_overrides(model: Any, xml_names: frozenset[str]) -> dict[str, float]:
    """The parameters of *model* the XML declares, that the model holds at a
    value other than its value at load, with the model's values."""
    core = model._core
    loaded = model._loaded_params
    out: dict[str, float] = {}
    for name in core.param_names:
        if name not in xml_names:
            continue
        value = float(core.get_param(name))
        if loaded.get(name) != value:
            out[name] = value
    return out
