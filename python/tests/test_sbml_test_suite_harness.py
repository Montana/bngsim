"""The SBML Test Suite harness scores a case whose event assigns a parameter or
resizes a compartment.

bngsim promotes a symbol an event assigns to integrator state, so it appears in
``Model.species_names``, but a Result does not report it as a species column
(GH #71); it reports a same-named observable instead (GH #202). The harness's
own grading indexed the Result's species block with the model's list, so every
such case (79 of the semantic suite) died with an IndexError and a full run
stopped at the first one; it also converted amounts with the compartment's t=0
volume, so after an event resized a compartment it reported the concentration
as the amount. It now grades through the shared kernel of
``benchmarks/suites/sbml_test_suite`` (GH #225), which handles both.

Each test writes one synthetic case in the suite's file layout and scores it
with ``run_single_case``, the function the harness runs per case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("libsbml")

_HARNESS = Path(__file__).resolve().parents[2] / "harness" / "sbml_test_suite"

_MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
)


def _sbml(event_assignments: str) -> str:
    # -> S at rate k1 (amount per time), from S = 1; an event at t = 0.75, off
    # the sample grid, so no sample sits on the discontinuity.
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments>
   <compartment id="C" spatialDimensions="3" size="1" constant="false"/>
  </listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialAmount="1" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters><parameter id="k1" value="1" constant="false"/></listOfParameters>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfProducts>
     <speciesReference species="S" stoichiometry="1" constant="true"/>
    </listOfProducts>
    <kineticLaw><math {_MATH}><ci>k1</ci></math></kineticLaw>
   </reaction>
  </listOfReactions>
  <listOfEvents>
   <event id="E" useValuesFromTriggerTime="true">
    <trigger initialValue="false" persistent="true">
     <math {_MATH}><apply><geq/>{_TIME}<cn>0.75</cn></apply></math>
    </trigger>
    <listOfEventAssignments>{event_assignments}</listOfEventAssignments>
   </event>
  </listOfEvents>
 </model>
</sbml>
"""


def _assign(var: str, value: str) -> str:
    return (
        f'<eventAssignment variable="{var}"><math {_MATH}><cn>{value}</cn></math>'
        "</eventAssignment>"
    )


def _case(tmp_path: Path, sbml: str, variables: list[str], rows: list[list[float]]) -> Path:
    cid = "90001"
    d = tmp_path / cid
    d.mkdir()
    (d / f"{cid}-sbml-l3v2.xml").write_text(sbml)
    (d / f"{cid}-model.m").write_text(
        "componentTags: Compartment, EventNoDelay, Parameter, Reaction, Species\n"
        "testTags:      Amount, NonConstantParameter\n"
        "testType:      TimeCourse\n"
    )
    (d / f"{cid}-settings.txt").write_text(
        "start: 0\nduration: 2\nsteps: 4\n"
        f"variables: {', '.join(variables)}\n"
        "absolute: 1e-6\nrelative: 1e-6\namount: S\nconcentration:\n"
    )
    lines = ["time," + ",".join(variables)]
    for t, row in zip([0.0, 0.5, 1.0, 1.5, 2.0], rows, strict=True):
        lines.append(",".join(str(v) for v in [t, *row]))
    (d / f"{cid}-results.csv").write_text("\n".join(lines) + "\n")
    return d


def _score(case_dir: Path) -> dict:
    sys.path.insert(0, str(_HARNESS))
    try:
        import run_sbml_test_suite as harness
    finally:
        sys.path.remove(str(_HARNESS))
    return harness.run_single_case(case_dir, case_dir.name)


# S amount: 1 + t up to t = 0.75, then 1.75 + 3 (t - 0.75) once k1 = 3.
_S = [1.0, 1.5, 2.5, 4.0, 5.5]


def test_a_parameter_an_event_assigns_is_scored(tmp_path):
    case = _case(
        tmp_path,
        _sbml(_assign("k1", "3")),
        ["S", "k1"],
        [[s, k] for s, k in zip(_S, [1, 1, 3, 3, 3], strict=True)],
    )
    r = _score(case)
    assert r["status"] == "pass", r


def test_an_amount_after_an_event_resizes_its_compartment_uses_the_live_volume(tmp_path):
    """C goes 1 -> 2 at the event. The amount of S is unchanged by the resize
    (its concentration halves), and the t=0 volume conversion reported the
    halved concentration as the amount."""
    case = _case(
        tmp_path,
        _sbml(_assign("k1", "3") + _assign("C", "2")),
        ["S", "k1", "C"],
        [[s, k, c] for s, k, c in zip(_S, [1, 1, 3, 3, 3], [1, 1, 2, 2, 2], strict=True)],
    )
    r = _score(case)
    assert r["status"] == "pass", r
