"""GH #563 — a foreign SED-ML's @target must beat its display @name.

bngsim writes the verbatim bngsim selector into a dataGenerator's ``@name``,
and that is the round-trip source of truth. Every other writer uses the pair
the standard describes: ``@target`` carries the machine-readable SBML XPath and
``@name`` a label for a human — "Free receptor (nM)". ``read_sedml`` took
``@name`` whenever it was present and only fell back to ``@target`` when it was
absent, so reading a third-party document threw away the one resolvable field
and returned prose as the output selector. The spec looked fine until something
tried to use it, and then ``load_protocol().evaluate()`` failed with
"Unresolved output selector 'Free receptor (nM)': not found".

``@name`` now wins only when it already reads as a typed bngsim selector
(``species:``, ``observable:``, ``expression:`` and the ``state:`` /
``function:`` aliases), which is exactly the document bngsim itself wrote. A
label that is not one yields to the target, and a generator with neither a
typed name nor a usable target — an observable or expression, which have no
standard SBML element to point at — still falls back to the name.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from bngsim import EvaluationSpec
from bngsim.convert import read_sedml, write_sedml

DATA = Path(
    os.environ.get("BNGSIM_TEST_DATA") or Path(__file__).resolve().parents[2] / "tests" / "data"
)

HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<sedML xmlns="http://sed-ml.org/sed-ml/level1/version3" level="1" version="3">
  <listOfSimulations>
    <uniformTimeCourse id="sim0" initialTime="0" outputStartTime="0"
                       outputEndTime="10" numberOfPoints="10"/>
  </listOfSimulations>
  <listOfModels>
    <model id="m0" language="urn:sedml:language:sbml" source="{source}"/>
  </listOfModels>
  <listOfTasks>
    <task id="t" modelReference="m0" simulationReference="sim0"/>
  </listOfTasks>
  <listOfDataGenerators>
    <dataGenerator id="dgt" name="Time (s)">
      <listOfVariables>
        <variable id="vt" taskReference="t" symbol="urn:sedml:symbol:time"/>
      </listOfVariables>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>vt</ci></math>
    </dataGenerator>
"""

GEN = """    <dataGenerator id="d{i}"{name}>
      <listOfVariables>
        <variable id="v{i}" taskReference="t"{attr}/>
      </listOfVariables>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>v{i}</ci></math>
    </dataGenerator>
"""

TAIL = """  </listOfDataGenerators>
</sedML>
"""


def _doc(generators, source="model.xml"):
    """Assemble a SED-ML document from (name, variable-attrs) pairs."""
    body = "".join(
        GEN.format(i=i, name=f' name="{n}"' if n is not None else "", attr=a)
        for i, (n, a) in enumerate(generators)
    )
    return HEAD.format(source=source) + body + TAIL


def _species_target(sid, quote="'"):
    return (
        f' target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id={quote}{sid}{quote}]"'
    )


# ── A foreign document ───────────────────────────────────────────────────────


def test_a_display_name_yields_to_the_target():
    """The issue's case: the label is prose, the target is the real selector."""
    doc = _doc([("Free receptor (nM)", _species_target("R"))])
    assert read_sedml(doc).outputs == ("species:R",)


def test_a_double_quoted_target_is_read_too():
    """Both quotings are well-formed XML; the old prefix match saw only one."""
    doc = _doc([("Bound receptor", _species_target("RL", quote="&quot;"))])
    assert read_sedml(doc).outputs == ("species:RL",)


def test_an_unprefixed_xpath_is_read():
    """The namespace prefix is whatever the document bound, so it is not part
    of the match."""
    doc = _doc([("Free receptor", " target=\"/sbml/model/listOfSpecies/species[@id='R']\"")])
    assert read_sedml(doc).outputs == ("species:R",)


def test_several_generators_keep_their_order():
    doc = _doc(
        [
            ("Free receptor (nM)", _species_target("R")),
            ("Bound receptor (nM)", _species_target("RL")),
        ]
    )
    assert read_sedml(doc).outputs == ("species:R", "species:RL")


def test_a_named_time_generator_is_still_dropped():
    """The time variable is recognised by its symbol, not by its label."""
    doc = _doc([("Free receptor (nM)", _species_target("R"))])
    assert "time" not in read_sedml(doc).outputs
    assert len(read_sedml(doc).outputs) == 1


# ── A bngsim document is unchanged ───────────────────────────────────────────


@pytest.mark.parametrize(
    "selector",
    ["species:R", "observable:Atot", "expression:f", "state:R", "function:f"],
)
def test_a_typed_name_still_wins(selector):
    """What bngsim writes, including when a target sits beside it."""
    doc = _doc([(selector, _species_target("SOMETHING_ELSE"))])
    assert read_sedml(doc).outputs == (selector,)


def test_the_round_trip_is_exact():
    spec = EvaluationSpec(
        model_source="m.xml",
        model_format="sbml",
        method="ode",
        t_span=(0.0, 10.0),
        n_points=11,
        outputs=("species:R", "observable:Atot", "expression:f"),
    )
    assert read_sedml(write_sedml(spec)).outputs == spec.outputs


# ── Nothing else to fall back to ─────────────────────────────────────────────


def test_a_name_with_no_usable_target_is_kept():
    """An observable or expression generator has no SBML element to point at,
    so the name is all there is — the pre-fix behavior, still correct here."""
    doc = _doc([("Atot", "")])
    assert read_sedml(doc).outputs == ("Atot",)


def test_a_nameless_generator_uses_its_target():
    doc = _doc([(None, _species_target("R"))])
    assert read_sedml(doc).outputs == ("species:R",)


# ── The selector the foreign document yields actually resolves ───────────────


def test_the_recovered_selector_resolves_on_a_real_model(tmp_path):
    """The end of the failure the issue reports: the spec now evaluates instead
    of raising "Unresolved output selector"."""
    sbml = DATA / "BIOMD0000000003.xml"
    if not sbml.exists():
        pytest.skip(f"{sbml.name} not in this checkout")
    doc = _doc([("Cyclin (nM)", _species_target("C"))], source=str(sbml))
    spec = read_sedml(doc)
    assert spec.outputs == ("species:C",)
    result = spec.evaluate()
    assert result.outputs("species:C").shape[0] == spec.n_points
