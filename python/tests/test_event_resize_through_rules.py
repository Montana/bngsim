"""Issue #1055 — an event resizes an assignment-rule compartment it reaches
only through another rule, or through another compartment's species.

SBML keeps a species' amount across a change of its compartment's size, so an
event that changes the size rescales the concentration by V_old/V_new (#74).
That was done only when the size rule named the event's variable itself:

- ``cv := 1 + q``, ``q := 0.5*p``, event ``p := 4``: cv goes 1.5 -> 3 and
  [Sn] 5 -> 2.5, but Sn stayed at 5;
- ``cv := 1 + 0.1*Kw``, Kw in ``ce``, event ``ce := 2``: [Kw] goes 10 -> 5,
  cv 2 -> 1.5, and [Sn] 5 -> 6.667, but Sn stayed at 5.

The second also needs the engine to read cv's new size after Kw's rescale,
which it read before. RoadRunner 2.9.2 gives the values asserted here.

Two more ways depended on the order the document lists the rules in: a size
rule that reads another rule compartment (``cv := 2*cu``, ``cu := 1 + q``,
with cu's rule first) left cv's species unrescaled, and a size rule that
reaches a change before the species it reads moves (``cw := p + 0.1*Sn``,
written before Sn's ``cv := 1 + 0.1*Kw``) read Sn before its rescale. The
sensitivities through each resize are checked against the closed form.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'


TRIGGER = f"<math {M}><apply><gt/>{T}<cn>1</cn></apply></math>"


def _doc(comps, species, params, rules, assignments, uvftt):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
 <listOfCompartments>{comps}</listOfCompartments>
 <listOfSpecies>{species}</listOfSpecies>
 <listOfParameters>{params}</listOfParameters>
 <listOfRules>{rules}</listOfRules>
 <listOfEvents><event id="e" useValuesFromTriggerTime="{uvftt}">
  <trigger initialValue="false" persistent="true">{TRIGGER}</trigger>
  <listOfEventAssignments>{assignments}</listOfEventAssignments>
 </event></listOfEvents>
</model></sbml>"""


def _comp(cid, size=1):
    return f'<compartment id="{cid}" size="{size}" constant="false"/>'


def _species(sid, comp, conc, constant=False):
    return (
        f'<species id="{sid}" compartment="{comp}" initialConcentration="{conc}" '
        f'hasOnlySubstanceUnits="false" boundaryCondition="false" '
        f'constant="{str(constant).lower()}"/>'
    )


def _rule(var, body):
    return f'<assignmentRule variable="{var}"><math {M}>{body}</math></assignmentRule>'


def _plus_tenth(name):
    return f"<apply><plus/><cn>1</cn><apply><times/><cn>0.1</cn><ci>{name}</ci></apply></apply>"


def _set(var, value):
    return _assign(var, f"<cn>{value}</cn>")


def _assign(var, body):
    return f'<eventAssignment variable="{var}"><math {M}>{body}</math></eventAssignment>'


def _param(pid, value, constant=False):
    return f'<parameter id="{pid}" value="{value}" constant="{str(constant).lower()}"/>'


def _apply(op, *args):
    return f"<apply><{op}/>{''.join(args)}</apply>"


def _ci(name):
    return f"<ci>{name}</ci>"


def _cn(value):
    return f"<cn>{value}</cn>"


def _chain(uvftt):
    return _doc(
        _comp("cv"),
        _species("Sn", "cv", 5),
        '<parameter id="p" value="1" constant="false"/>'
        '<parameter id="q" value="0" constant="false"/>',
        _rule("cv", "<apply><plus/><cn>1</cn><ci>q</ci></apply>")
        + _rule("q", "<apply><times/><cn>0.5</cn><ci>p</ci></apply>"),
        _set("p", 4),
        uvftt,
    )


def _via_species(uvftt):
    return _doc(
        _comp("cv") + _comp("ce"),
        _species("Sn", "cv", 5) + _species("Kw", "ce", 10, constant=True),
        "",
        _rule("cv", _plus_tenth("Kw")),
        _set("ce", 2),
        uvftt,
    )


def _cascade(uvftt):
    """ce -> [Kw] -> cv -> [Sn] -> cw -> [Z], with cw's rule written first."""
    return _doc(
        _comp("cv") + _comp("ce") + _comp("cw"),
        _species("Sn", "cv", 5) + _species("Kw", "ce", 10, constant=True) + _species("Z", "cw", 3),
        "",
        _rule("cw", _plus_tenth("Sn")) + _rule("cv", _plus_tenth("Kw")),
        _set("ce", 2),
        uvftt,
    )


def _direct(uvftt):
    return _doc(
        _comp("cv"),
        _species("Sn", "cv", 5),
        '<parameter id="p" value="1" constant="false"/>',
        _rule("cv", "<ci>p</ci>"),
        _set("p", 2),
        uvftt,
    )


def _rule_compartment_reads_another(uvftt):
    """cv := 2*cu, cu := 1 + q, q := 0.5*p, with cu's rule written before cv's."""
    return _doc(
        _comp("cu") + _comp("cv"),
        _species("U", "cu", 4) + _species("Sn", "cv", 5),
        _param("p", 1) + _param("q", 0),
        _rule("cu", _apply("plus", _cn(1), _ci("q")))
        + _rule("cv", _apply("times", _cn(2), _ci("cu")))
        + _rule("q", _apply("times", _cn(0.5), _ci("p"))),
        _set("p", 4),
        uvftt,
    )


def _reaches_a_change_first(uvftt):
    """cw := p + 0.1*Sn, written before Sn's cv := 1 + 0.1*Kw, and an event
    setting p and Kw's compartment: cw reaches p before it reaches Sn."""
    return _doc(
        _comp("cv") + _comp("ce") + _comp("cw"),
        _species("Sn", "cv", 5) + _species("Kw", "ce", 10, constant=True) + _species("Z", "cw", 3),
        _param("p", 1),
        _rule("cw", _apply("plus", _ci("p"), _apply("times", _cn(0.1), _ci("Sn"))))
        + _rule("cv", _plus_tenth("Kw")),
        _set("ce", 2) + _set("p", 2),
        uvftt,
    )


CASES = {
    # amount 7.5 in cv = 1.5, then cv = 3
    "through another rule": (_chain, {"Sn": 2.5}),
    # [Kw] 10 -> 5, cv 2 -> 1.5, amount 10 in cv
    "through a rescaled species": (_via_species, {"Sn": 10 / 1.5, "Kw": 5.0}),
    # and then cw 1.5 -> 1 + 0.1 * 20/3, amount 4.5 in cw
    "a cascade of three": (_cascade, {"Sn": 10 / 1.5, "Kw": 5.0, "Z": 4.5 / (1 + 2 / 3)}),
    # cu 1.5 -> 3 and cv 3 -> 6, amounts 6 and 15
    "through another rule compartment": (_rule_compartment_reads_another, {"U": 2.0, "Sn": 2.5}),
    # cv 2 -> 1.5, then cw 1.5 -> 2 + 0.1 * 20/3, amount 4.5 in cw
    "reaching a change before the species it reads": (
        _reaches_a_change_first,
        {"Sn": 10 / 1.5, "Z": 4.5 / (2 + 2 / 3)},
    ),
    # control: the size rule names the event's variable
    "direct (control)": (_direct, {"Sn": 2.5}),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("uvftt", ["true", "false"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_amount_is_kept_across_the_resize(case, uvftt, codegen):
    build, want = CASES[case]
    model = bngsim.Model.from_sbml_string(build(uvftt))
    result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
        t_span=(0.0, 4.0), n_points=9
    )
    names = list(result.species_names)
    species = np.asarray(result.species)
    for sid, value in want.items():
        column = species[:, names.index(sid)]
        # Before the event at t = 1 nothing moves; after it, the new value holds.
        np.testing.assert_allclose(column[-1], value, rtol=1e-6, err_msg=sid)
        np.testing.assert_allclose(column[3:], value, rtol=1e-6, err_msg=sid)


def _ab(a, b):
    return _param("a", a, constant=True) + _param("b", b, constant=True)


# (document, species, its value after the event, its derivative in a and b), in
# closed form. Nothing moves between events, so each is the event's own jump.
SENSITIVITIES = {
    # [Sn] = 5(a + 0.5)/(a + b/2)
    "through another rule": (
        _doc(
            _comp("cv"),
            _species("Sn", "cv", 5),
            _ab(1, 4) + _param("p", 1) + _param("q", 0),
            _rule("cv", _apply("plus", _ci("a"), _ci("q")))
            + _rule("q", _apply("times", _cn(0.5), _ci("p"))),
            _assign("p", _ci("b")),
            "true",
        ),
        "Sn",
        2.5,
        [5 / 6, -5 / 12],
    ),
    # [Sn] = 5(2a + 1)/(2a + b)
    "through another rule compartment": (
        _doc(
            _comp("cu") + _comp("cv"),
            _species("Sn", "cv", 5),
            _ab(1, 2) + _param("p", 1) + _param("q", 0),
            _rule("cu", _apply("plus", _ci("a"), _ci("q")))
            + _rule("cv", _apply("times", _cn(2), _ci("cu")))
            + _rule("q", _apply("times", _cn(0.5), _ci("p"))),
            _assign("p", _ci("b")),
            "true",
        ),
        "Sn",
        3.75,
        [0.625, -0.9375],
    ),
    # [Sn] = 10b/(b + 1), then [Z] = 3(1 + 5a)/(2 + 10ab/(b + 1))
    "reaching a change before the species it reads": (
        _doc(
            _comp("cv") + _comp("ce") + _comp("cw"),
            _species("Sn", "cv", 5)
            + _species("Kw", "ce", 10, constant=True)
            + _species("Z", "cw", 3),
            _ab(0.1, 2) + _param("p", 1),
            _rule("cw", _apply("plus", _ci("p"), _apply("times", _ci("a"), _ci("Sn"))))
            + _rule("cv", _plus_tenth("Kw")),
            _assign("ce", _ci("b")) + _set("p", 2),
            "true",
        ),
        "Z",
        1.6875,
        [1.40625, -0.0703125],
    ),
    # X = b in ce = 1 keeps its amount b in ce = 2, so [X] = b/2 and
    # [Sn] = 5(a + 1)/(a + 0.05b)
    "a species the event assigns in a compartment it resizes": (
        _doc(
            _comp("ce") + _comp("cv"),
            _species("X", "ce", 10) + _species("Sn", "cv", 5),
            _ab(1, 10),
            _rule("cv", _apply("plus", _ci("a"), _apply("times", _cn(0.1), _ci("X")))),
            _set("ce", 2) + _assign("X", _ci("b")),
            "true",
        ),
        "Sn",
        20 / 3,
        [-10 / 9, -2 / 9],
    ),
}


@pytest.mark.parametrize("case", list(SENSITIVITIES))
def test_the_sensitivity_through_the_resize(case):
    doc, sid, value, grad = SENSITIVITIES[case]
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(doc), method="ode", sensitivity_params=["a", "b"]
    ).run(t_span=(0.0, 4.0), n_points=9)
    names = list(result.species_names)
    np.testing.assert_allclose(np.asarray(result.species)[-1, names.index(sid)], value, rtol=1e-6)
    np.testing.assert_allclose(
        np.ravel(np.asarray(result.output_sensitivities(f"species:{sid}"))[-1]), grad, rtol=1e-5
    )


# A piecewise size whose condition reads a concentration an event moves (#1054
# beside #1055). The event's resize carries the jump; the jump's own rescale
# (#1054) added a second one at the instant, so [Sn] read 20 for 10.
_KW = (
    "compartment ce = 1; species Kw in ce; Kw = 10; compartment cv = 1; species Sn in cv; Sn = 5; "
)
PIECEWISE_AT_AN_EVENT = {
    "a species the event's resize rescales": (
        _KW + "cv := piecewise(2, Kw > 7, 1); at (time > 1): ce = 2;",
        10.0,
    ),
    "the same species, read through a rule": (
        _KW + "q := 0.5*Kw; cv := piecewise(2, q > 3.5, 1); at (time > 1): ce = 2;",
        10.0,
    ),
    # Kw is not constant, but no reaction or rule moves it, so only the event
    # does: it was refused as a condition the run moves too.
    "a species the event assigns": (
        _KW + "cv := piecewise(1, Kw > 7, 2); at (time > 1): Kw = 2;",
        2.5,
    ),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(PIECEWISE_AT_AN_EVENT))
def test_a_piecewise_size_an_event_flips_is_rescaled_once(case, codegen):
    text, after = PIECEWISE_AT_AN_EVENT[case]
    result = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", codegen=codegen
    ).run(t_span=(0.0, 3.0), n_points=7, rtol=1e-10, atol=1e-12)
    t = np.asarray(result.time)
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    np.testing.assert_allclose(sn, np.where(t <= 1, 5.0, after), rtol=1e-8)


def test_a_piecewise_size_an_event_and_a_reaction_both_flip_is_refused():
    """A reaction moves Kw as well, so the condition can flip at the event and
    by the run: refused by name, not rescaled twice."""
    with pytest.raises(bngsim.ModelError, match="Kw, which an event changes"):
        bngsim.Model.from_antimony_string(
            _KW + "J: Kw => ; 0.01*Kw; cv := piecewise(2, Kw > 7, 1); at (time > 1): ce = 2;"
        )
