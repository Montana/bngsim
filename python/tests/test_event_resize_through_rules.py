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


# ── Issue #1124, item 4: a diamond of rules ─────────────────────────────────
# r0 := k*p, r1 := p + 1, r_i := r_{i-1} + r_{i-2}, and cv := 1 + 1e-6*r_{n-1}.
# Written out through its rules, the new size read every rule once per path,
# and its text was exponential in n: 278 KB at 22 rules, 1.9 MB at 26. Past a
# limit, a rule more than one path reaches is now a function of its own.


def _diamond(n):
    # r_{n-1} = A*k*p + B*(p + 1)
    coef = [(1, 0), (0, 1)]
    for _ in range(2, n):
        coef.append((coef[-1][0] + coef[-2][0], coef[-1][1] + coef[-2][1]))
    return coef[n - 1]


def _diamond_doc(n, k=1.0):
    rules = _rule("r0", _apply("times", _ci("k"), _ci("p"))) + _rule(
        "r1", _apply("plus", _ci("p"), _cn(1))
    )
    rules += "".join(
        _rule(f"r{i}", _apply("plus", _ci(f"r{i - 1}"), _ci(f"r{i - 2}"))) for i in range(2, n)
    )
    rules += _rule("cv", _apply("plus", _cn(1), _apply("times", _cn(1e-6), _ci(f"r{n - 1}"))))
    return _doc(
        _comp("cv"),
        _species("Sn", "cv", 5),
        _param("p", 1)
        + _param("k", k, constant=True)
        + "".join(_param(f"r{i}", 0) for i in range(n)),
        rules,
        _set("p", 2),
        "true",
    )


def _diamond_size(n, k, p):
    a, b = _diamond(n)
    return 1 + 1e-6 * (a * k * p + b * (p + 1))


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("n", [10, 26])
def test_a_diamond_of_rules_is_rescaled(n, codegen):
    """At 10 rules the size is written out as before; at 26 it reads
    functions of its own. Each gives 5·V(p = 1)/V(p = 2) after the event."""
    model = bngsim.Model.from_sbml_string(_diamond_doc(n))
    helpers = [f for f in model.function_names if f.startswith("_rsz_")]
    assert bool(helpers) == (n == 26)
    result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
        t_span=(0.0, 4.0), n_points=9
    )
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    want = 5 * _diamond_size(n, 1.0, 1) / _diamond_size(n, 1.0, 2)
    np.testing.assert_allclose(sn[3:], want, rtol=1e-12)
    np.testing.assert_allclose(sn[:3], 5.0, rtol=1e-12)


def test_the_sensitivity_through_a_diamond_of_rules():
    """d[Sn]/dk after the event, through the functions: with
    V1 = V(k, p = 1) and V2 = V(k, p = 2), 5·(dV1/dk·V2 − V1·dV2/dk)/V2²."""
    n = 26
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(_diamond_doc(n)), method="ode", sensitivity_params=["k"]
    ).run(t_span=(0.0, 4.0), n_points=9, rtol=1e-10, atol=1e-12)
    a, _ = _diamond(n)
    v1, v2 = _diamond_size(n, 1.0, 1), _diamond_size(n, 1.0, 2)
    want = 5 * (1e-6 * a * v2 - v1 * 2e-6 * a) / v2**2
    w = list(result.species_names).index("Sn")
    np.testing.assert_allclose(np.asarray(result.sensitivities)[-1, w, 0], want, rtol=1e-6)


def test_a_size_jump_followed_through_a_diamond_of_rules():
    """cv := piecewise(1, t < 2, 5) takes [Sn] 5 -> 1, and cw := 1 + 1e-9·r25
    reads it through 26 rules (r0 := [Sn], r1 := [Sn] + 1): the jump's
    rescale of [Sw] goes through functions of its own too."""
    n = 26
    lt2 = f"<apply><lt/>{T}<cn>2</cn></apply>"
    step = (
        f"<piecewise><piece><cn>1</cn>{lt2}</piece><otherwise><cn>5</cn></otherwise></piecewise>"
    )
    rules = (
        _rule("cv", step) + _rule("r0", _ci("Sn")) + _rule("r1", _apply("plus", _ci("Sn"), _cn(1)))
    )
    rules += "".join(
        _rule(f"r{i}", _apply("plus", _ci(f"r{i - 1}"), _ci(f"r{i - 2}"))) for i in range(2, n)
    )
    rules += _rule("cw", _apply("plus", _cn(1), _apply("times", _cn(1e-9), _ci(f"r{n - 1}"))))
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
 <listOfCompartments>{_comp("cv")}{_comp("cw")}</listOfCompartments>
 <listOfSpecies>{_species("Sn", "cv", 5)}{_species("Sw", "cw", 3)}</listOfSpecies>
 <listOfParameters>{"".join(_param(f"r{i}", 0) for i in range(n))}</listOfParameters>
 <listOfRules>{rules}</listOfRules>
</model></sbml>"""
    model = bngsim.Model.from_sbml_string(xml)
    assert any(f.startswith("_rsz_") for f in model.function_names)
    result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 4.0), n_points=9)
    t = np.asarray(result.time)
    sw = np.asarray(result.species)[:, list(result.species_names).index("Sw")]
    a, b = _diamond(n)  # r25 = a·r0 + b·r1 = (a + b)·[Sn] + b

    def cw(sn):
        return 1 + 1e-9 * ((a + b) * sn + b)

    np.testing.assert_allclose(sw, np.where(t < 2, 3.0, 3 * cw(5) / cw(1)), rtol=1e-12)


# The engine forms each rescale itself, from the compartment's size read by
# name before and after the event (#936), so the trajectory reads none of
# these functions: the tests above pass with every one of them wrong. Only a
# sensitivity reads them, through the new size's text. So each case below is
# checked by its sensitivity.


def _diamond_rules(n, r0, r1):
    """``r0``, ``r1`` and r_i := r_{i-1} + r_{i-2}, up to r_{n-1}."""
    return (
        _rule("r0", r0)
        + _rule("r1", r1)
        + "".join(
            _rule(f"r{i}", _apply("plus", _ci(f"r{i - 1}"), _ci(f"r{i - 2}"))) for i in range(2, n)
        )
    )


def _size_of(name, scale):
    """``1 + scale·name``."""
    return _apply("plus", _cn(1), _apply("times", _cn(scale), _ci(name)))


def _event(eid, at, assignments):
    after = f"<math {M}><apply><gt/>{T}<cn>{at}</cn></apply></math>"
    return f"""<event id="{eid}" useValuesFromTriggerTime="true">
  <trigger initialValue="false" persistent="true">{after}</trigger>
  <listOfEventAssignments>{assignments}</listOfEventAssignments>
 </event>"""


def _model(comps, species, params, rules, events=""):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
 <listOfCompartments>{comps}</listOfCompartments>
 <listOfSpecies>{species}</listOfSpecies>
 <listOfParameters>{params}</listOfParameters>
 <listOfRules>{rules}</listOfRules>
 <listOfEvents>{events}</listOfEvents>
</model></sbml>"""


def _final_sensitivity(model, *species):
    """d[species]/dk at t = 4, after the events."""
    result = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
        t_span=(0.0, 4.0), n_points=9, rtol=1e-10, atol=1e-12
    )
    names = list(result.species_names)
    return np.asarray(result.sensitivities)[-1, [names.index(s) for s in species], 0]


def _rsz(model):
    return [f for f in model.function_names if f.startswith("_rsz_")]


def test_the_sensitivity_through_a_size_jump_followed_through_a_diamond_of_rules():
    """With r0 := k·[Sn], the jump's rescale of [Sw] reads k through
    functions of its own: d[Sw]/dk after the jump is 3·(W5'·W1 − W5·W1')/W1²,
    W_s being cw at [Sn] = s."""
    n = 26
    lt2 = f"<apply><lt/>{T}<cn>2</cn></apply>"
    step = (
        f"<piecewise><piece><cn>1</cn>{lt2}</piece><otherwise><cn>5</cn></otherwise></piecewise>"
    )
    rules = _rule("cv", step) + _diamond_rules(
        n, _apply("times", _ci("k"), _ci("Sn")), _apply("plus", _ci("Sn"), _cn(1))
    )
    model = bngsim.Model.from_sbml_string(
        _model(
            _comp("cv") + _comp("cw"),
            _species("Sn", "cv", 5) + _species("Sw", "cw", 3),
            _param("k", 1.0, constant=True) + "".join(_param(f"r{i}", 0) for i in range(n)),
            rules + _rule("cw", _size_of(f"r{n - 1}", 1e-9)),
        )
    )
    assert _rsz(model)
    a, b = _diamond(n)

    def w(sn):
        return 1 + 1e-9 * (a * sn + b * (sn + 1))

    def dw(sn):
        return 1e-9 * a * sn

    want = 3 * (dw(5) * w(1) - w(5) * dw(1)) / w(1) ** 2
    np.testing.assert_allclose(_final_sensitivity(model, "Sw"), want, rtol=1e-6)


def _two_compartments_two_events(n, event_ids, comp_ids, values):
    """Each compartment reads the same diamond from p, and each event sets
    p: so each event resizes both."""
    rules = _diamond_rules(
        n, _apply("times", _ci("k"), _ci("p")), _apply("plus", _ci("p"), _cn(1))
    ) + "".join(_rule(c, _size_of(f"r{n - 1}", 1e-6)) for c in comp_ids)
    return _model(
        "".join(_comp(c) for c in comp_ids),
        _species("A", comp_ids[0], 5) + _species("B", comp_ids[1], 7),
        _param("p", 1)
        + _param("k", 1.0, constant=True)
        + "".join(_param(f"r{i}", 0) for i in range(n)),
        rules,
        "".join(
            _event(eid, at, _set("p", v))
            for eid, at, v in zip(event_ids, (1, 2), values, strict=True)
        ),
    )


def test_two_events_whose_ids_meet_read_their_own_functions():
    """Event e with compartment c_x and event e_c with compartment x both
    name their functions _rsz_e_c_x_…: one name held two functions, and the
    sensitivity of the first event was refused as a second fire of its
    instant. After both, [A] = 5·V(1)/V(3) for V(p) = 1 + 1e-6·(a·k·p +
    b·(p + 1))."""
    n = 26
    model = bngsim.Model.from_sbml_string(
        _two_compartments_two_events(n, ("e", "e_c"), ("c_x", "x"), (2, 3))
    )
    names = _rsz(model)
    assert names and len(set(names)) == len(names)
    a, b = _diamond(n)

    def v(p):
        return 1 + 1e-6 * (a * p + b * (p + 1))

    def dv(p):
        return 1e-6 * a * p

    want = [c0 * (dv(1) * v(3) - v(1) * dv(3)) / v(3) ** 2 for c0 in (5, 7)]
    np.testing.assert_allclose(_final_sensitivity(model, "A", "B"), want, rtol=1e-6)


def test_events_that_put_in_the_same_values_share_their_functions():
    """Every function is evaluated at every step, so a text two events
    write is one function, not two."""
    n = 26
    one = len(_rsz(bngsim.Model.from_sbml_string(_diamond_doc(n))))
    both = bngsim.Model.from_sbml_string(
        _two_compartments_two_events(n, ("e1", "e2"), ("cv", "cw"), (2, 2))
    )
    # Two compartments read the diamond, through the same rules.
    assert len(_rsz(both)) == one


def test_no_function_is_kept_that_no_rescale_reads():
    """A species with hasOnlySubstanceUnits keeps its amount with no
    rescale, so nothing reads its compartment's new size: its functions were
    added all the same, and evaluated at every step for nothing. [Sn] is
    reported as 5/V, from 5/V(p = 1) to 5/V(p = 2)."""
    hosu = (
        '<species id="Sn" compartment="cv" initialAmount="5" hasOnlySubstanceUnits="true" '
        'boundaryCondition="false" constant="false"/>'
    )
    xml = _diamond_doc(26).replace(_species("Sn", "cv", 5), hosu)
    assert hosu in xml
    model = bngsim.Model.from_sbml_string(xml)
    assert _rsz(model) == []
    result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 4.0), n_points=9)
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    np.testing.assert_allclose(sn[:3], 5 / _diamond_size(26, 1.0, 1), rtol=1e-12)
    np.testing.assert_allclose(sn[3:], 5 / _diamond_size(26, 1.0, 2), rtol=1e-12)


def test_the_limit_counts_the_values_an_event_puts_in():
    """16 rules write out in 15 KB, but each of their 987 readings of p is
    given p's new value: one of 60 terms made the new size 720 KB, under a
    limit of 20,000 characters that counted the rules alone."""
    n, terms = 16, 60
    value = _apply(
        "plus", _cn(2), *(_apply("times", _cn(0.001), _ci(f"q{j}")) for j in range(terms))
    )
    xml = _doc(
        _comp("cv"),
        _species("Sn", "cv", 5),
        _param("p", 1)
        + _param("k", 1.0, constant=True)
        + "".join(_param(f"q{j}", 1.0, constant=True) for j in range(terms))
        + "".join(_param(f"r{i}", 0) for i in range(n)),
        _diamond_rules(n, _apply("times", _ci("k"), _ci("p")), _apply("plus", _ci("p"), _cn(1)))
        + _rule("cv", _size_of(f"r{n - 1}", 1e-6)),
        _assign("p", value),
        "true",
    )
    model = bngsim.Model.from_sbml_string(xml)
    assert _rsz(model)
    a, b = _diamond(n)
    p2 = 2 + 0.001 * terms

    def v(p):
        return 1 + 1e-6 * (a * p + b * (p + 1))

    want = 5 * (1e-6 * a * v(p2) - v(1) * 1e-6 * a * p2) / v(p2) ** 2
    np.testing.assert_allclose(_final_sensitivity(model, "Sn"), want, rtol=1e-6)


def test_the_sensitivity_through_a_chain_of_rules_deeper_than_was_written_out():
    """r0 := k·p, r_i := r_{i-1} + 1 up to r69, cv := 1 + 1e-3·r69: the new
    size was written out 65 rules deep and read the rest by name, at their
    value before the event. Its sensitivity was refused as a second fire of
    the instant (#936)."""
    n = 70
    rules = _rule("r0", _apply("times", _ci("k"), _ci("p")))
    rules += "".join(_rule(f"r{i}", _apply("plus", _ci(f"r{i - 1}"), _cn(1))) for i in range(1, n))
    model = bngsim.Model.from_sbml_string(
        _doc(
            _comp("cv"),
            _species("Sn", "cv", 5),
            _param("p", 1)
            + _param("k", 1.0, constant=True)
            + "".join(_param(f"r{i}", 0) for i in range(n)),
            rules + _rule("cv", _size_of(f"r{n - 1}", 1e-3)),
            _set("p", 2),
            "true",
        )
    )

    def v(p):
        return 1 + 1e-3 * (p + n - 1)

    want = 5 * (1e-3 * v(2) - v(1) * 2e-3) / v(2) ** 2
    np.testing.assert_allclose(_final_sensitivity(model, "Sn"), want, rtol=1e-6)
