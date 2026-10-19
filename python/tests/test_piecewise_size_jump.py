"""Issue #1054 — a piecewise assignment-rule compartment size that jumps.

SBML keeps a species' amount across a change of its compartment's size, so a
concentration jumps by V_before/V_after where the size does. The §8c dilution
term carries the size's rate between jumps, but a jump is a delta no ODE term
carries, and the concentration was held there instead:

- ``cv := piecewise(1 + 0.1*S, S > 3, 1 + 0.2*S)``, S decaying from 10 at 0.5,
  jumps 1.30 -> 1.60 at t = 2.41, and [Sn] stayed high by 1.60/1.30 after;
- ``cv := piecewise(1 + t, t < 2, 5)`` jumps 3 -> 5 at t = 2, and [Sn] stayed
  at 1.667 where the amount, 5, over 5 is 1.

Conditions that change at one root, ``t < 2`` and ``t >= 2`` in a piecewise
with no otherwise, are one jump, not one each. Each case is checked against
amount/V(t) in closed form.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'


def _pw(*pieces, otherwise=None):
    body = "".join(f"<piece>{v}{c}</piece>" for v, c in pieces)
    tail = "" if otherwise is None else f"<otherwise>{otherwise}</otherwise>"
    return f"<piecewise>{body}{tail}</piecewise>"


def _plus(a, b):
    return f"<apply><plus/>{a}{b}</apply>"


def _times(a, b):
    return f"<apply><times/>{a}{b}</apply>"


def _rel(op, a, b):
    return f"<apply><{op}/>{a}{b}</apply>"


def _cn(x):
    return f"<cn>{x}</cn>"


def _ci(x):
    return f"<ci>{x}</ci>"


def _doc(rules, *, functions="", params="", events="", species=""):
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {M}>{b}</math></assignmentRule>' for v, b in rules
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
 <listOfFunctionDefinitions>{functions}</listOfFunctionDefinitions>
 <listOfCompartments><compartment id="c" size="1" constant="true"/>
  <compartment id="cv" size="1" constant="false"/></listOfCompartments>
 <listOfSpecies>
  <species id="S" compartment="c" initialAmount="10" hasOnlySubstanceUnits="false"
   boundaryCondition="false" constant="false"/>
  <species id="Sn" compartment="cv" initialConcentration="5"
   hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>{species}
 </listOfSpecies>
 <listOfParameters><parameter id="k" value="0.5" constant="true"/>{params}</listOfParameters>
 <listOfRules>{rule_xml}</listOfRules>
 <listOfReactions><reaction id="degS" reversible="false">
  <listOfReactants>
   <speciesReference species="S" stoichiometry="1" constant="true"/>
  </listOfReactants>
  <kineticLaw><math {M}>{_times(_times(_ci("k"), _ci("S")), _ci("c"))}</math></kineticLaw>
 </reaction></listOfReactions>
 <listOfEvents>{events}</listOfEvents>
</model></sbml>"""


S_SWITCH = _pw(
    (_plus(_cn(1), _times(_cn(0.1), _ci("S"))), _rel("gt", _ci("S"), _cn(3))),
    otherwise=_plus(_cn(1), _times(_cn(0.2), _ci("S"))),
)
T_STEP = _pw((_plus(_cn(1), T), _rel("lt", T, _cn(2))), otherwise=_cn(5))
# up at t = 1 and back down at t = 3
T_WINDOW = _pw(
    (_cn(2), f"<apply><and/>{_rel('gt', T, _cn(1))}{_rel('lt', T, _cn(3))}</apply>"),
    otherwise=_cn(1),
)
# no otherwise: `t < 2` and `t >= 2` flip at one root
T_NO_OTHERWISE = _pw((_plus(_cn(1), T), _rel("lt", T, _cn(2))), (_cn(5), _rel("geq", T, _cn(2))))
S_NO_OTHERWISE = _pw(
    (_plus(_cn(1), _times(_cn(0.1), _ci("S"))), _rel("gt", _ci("S"), _cn(3))),
    (_plus(_cn(1), _times(_cn(0.2), _ci("S"))), _rel("leq", _ci("S"), _cn(3))),
)
# no piece at t = 2.1 itself; the jump's rescale divided by the 0 read there
T_GAP = _pw((_cn(1), _rel("lt", T, _cn(2.1))), (_cn(5), _rel("gt", T, _cn(2.1))))
# continuous at its switch: 1 + t meets 3 at t = 2
T_CONTINUOUS = _pw((_plus(_cn(1), T), _rel("lt", T, _cn(2))), otherwise=_cn(3))
STEP_FN = (
    '<functionDefinition id="step"><math '
    + M
    + "><lambda><bvar><ci>x</ci></bvar>"
    + _pw((_plus(_cn(1), _ci("x")), _rel("lt", _ci("x"), _cn(2))), otherwise=_cn(5))
    + "</lambda></math></functionDefinition>"
)


def _s(t):
    return 10 * np.exp(-0.5 * t)


CASES = {
    # the size jumps 1.30 -> 1.60 as S falls through 3; V(0) = 2, amount 10
    "on a species": (
        _doc([("cv", S_SWITCH)]),
        lambda t: 10 / np.where(_s(t) > 3, 1 + 0.1 * _s(t), 1 + 0.2 * _s(t)),
    ),
    "on time": (_doc([("cv", T_STEP)]), lambda t: 5 / np.where(t < 2, 1 + t, 5)),
    "both ways": (
        _doc([("cv", T_WINDOW)]),
        lambda t: 5 / np.where((t > 1) & (t < 3), 2, 1),
    ),
    "through another rule": (
        _doc(
            [("cv", _plus(_cn(0), _ci("q"))), ("q", T_STEP)],
            params='<parameter id="q" value="1" constant="false"/>',
        ),
        lambda t: 5 / np.where(t < 2, 1 + t, 5),
    ),
    "through a function definition": (
        _doc([("cv", f"<apply><ci>step</ci>{T}</apply>")], functions=STEP_FN),
        lambda t: 5 / np.where(t < 2, 1 + t, 5),
    ),
    # each pair is one jump, not two rescales at the instant
    "with no otherwise, on time": (
        _doc([("cv", T_NO_OTHERWISE)]),
        lambda t: 5 / np.where(t < 2, 1 + t, 5),
    ),
    "with no otherwise, on a species": (
        _doc([("cv", S_NO_OTHERWISE)]),
        lambda t: 10 / np.where(_s(t) > 3, 1 + 0.1 * _s(t), 1 + 0.2 * _s(t)),
    ),
    "with no piece at the switch": (
        _doc([("cv", T_GAP)]),
        lambda t: 5 / np.where(t < 2.1, 1, 5),
    ),
    # control: no jump, so nothing to rescale
    "continuous (control)": (
        _doc([("cv", T_CONTINUOUS)]),
        lambda t: 5 / np.where(t < 2, 1 + t, 3),
    ),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_amount_is_kept_across_the_jump(case, codegen):
    xml, closed_form = CASES[case]
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", codegen=codegen
    ).run(t_span=(0.0, 4.0), n_points=17)
    t = np.asarray(result.time)
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    np.testing.assert_allclose(sn, closed_form(t), rtol=1e-5, atol=1e-8)


def _set_sn_at_2(priority):
    return (
        '<event id="e" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true"><math {M}>'
        f"{_rel('geq', T, _cn(2))}</math></trigger>"
        + (f"<priority><math {M}>{_cn(priority)}</math></priority>" if priority else "")
        + f'<listOfEventAssignments><eventAssignment variable="Sn"><math {M}>{_cn(10)}'
        "</math></eventAssignment></listOfEventAssignments></event>"
    )


@pytest.mark.parametrize(
    "priority", [None, 5, "1e308"], ids=["no priority", "a priority", "the jump's priority"]
)
def test_an_event_at_the_jump_keeps_the_concentration_it_sets(priority):
    """``at (t >= 2): Sn = 10`` fires where ``cv := piecewise(1 + t, t < 2,
    5)`` jumps. Sn is 10 after, the concentration the event sets in the
    compartment as it is then; the jump's rescale of Sn from before the instant
    overwrote it whenever it ran second, by the priority or the draw between
    equal ones. At the jump's own priority, 1e308, the draw was a fair one, and
    the model did not load: 1e308 is an integer too large for the int field a
    constant priority was given."""
    xml = _doc([("cv", T_STEP)], events=_set_sn_at_2(priority))
    for seed in range(4):
        result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
            t_span=(0.0, 4.0), n_points=17, seed=seed
        )
        t = np.asarray(result.time)
        sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
        np.testing.assert_allclose(sn, np.where(t < 2, 5 / (1 + t), 10), rtol=1e-5, atol=1e-8)


# ─── What an event at the jump reads (issue #1054's review) ─────────────────
#
# `cv := piecewise(1 + t, t < 2, 5)` takes [Sn] from 5/3 just before t = 2 to 1
# at t = 2. The jump's rescale executed first, but an event triggered at the
# instant had its trigger, values and delay read with the rest of the batch,
# before it, at [Sn] = 5/3: `Sn = Sn + 1` left 8/3 where [Sn] at t = 2 is 1, and
# libRoadRunner gives 2.


def _event(eid, trigger, assignments, *, uvftt=True, persistent=True, delay=None, priority=None):
    body = "".join(
        f'<eventAssignment variable="{v}"><math {M}>{e}</math></eventAssignment>'
        for v, e in assignments
    )
    return (
        f'<event id="{eid}" useValuesFromTriggerTime="{str(uvftt).lower()}">'
        f'<trigger initialValue="false" persistent="{str(persistent).lower()}">'
        f"<math {M}>{trigger}</math></trigger>"
        + (f"<delay><math {M}>{delay}</math></delay>" if delay is not None else "")
        + (f"<priority><math {M}>{priority}</math></priority>" if priority is not None else "")
        + f"<listOfEventAssignments>{body}</listOfEventAssignments></event>"
    )


def _sp(sid):
    return (
        f'<species id="{sid}" compartment="c" initialConcentration="0" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


AT_2 = _rel("geq", T, _cn(2))
SN_PLUS_1 = _plus(_ci("Sn"), _cn(1))


def _sn(after):
    return lambda t: np.where(t < 2, 5 / (1 + t), after)


def _from(t0, value):
    return lambda t: np.where(t < t0, 0.0, value)


AT_THE_JUMP = {
    "a value frozen at the trigger": ([_event("e", AT_2, [("Sn", SN_PLUS_1)])], {"Sn": _sn(2.0)}),
    # control: read when it executes, after the jump, it was right
    "a value read when it executes": (
        [_event("e", AT_2, [("Sn", SN_PLUS_1)], uvftt=False)],
        {"Sn": _sn(2.0)},
    ),
    "another species set from it": ([_event("e", AT_2, [("X", _ci("Sn"))])], {"X": _from(2, 1.0)}),
    "a priority": (
        [_event("e", AT_2, [("X", _ci("Sn"))], priority=_cn(5))],
        {"X": _from(2, 1.0)},
    ),
    "after a delay": (
        [_event("e", AT_2, [("X", _ci("Sn"))], delay=_cn(0.9))],
        {"X": _from(2.9, 1.0)},
    ),
    # 0.6·[Sn] is 0.6 at t = 2, and was read 1
    "a delay that reads it": (
        [_event("e", AT_2, [("X", _cn(1))], delay=_times(_cn(0.6), _ci("Sn")))],
        {"X": _from(2.6, 1.0)},
    ),
    # [Sn] > 1.5 is false at t = 2, and was read true
    "a trigger that reads it": (
        [
            _event(
                "e",
                f"<apply><and/>{AT_2}{_rel('gt', _ci('Sn'), _cn(1.5))}</apply>",
                [("X", _cn(7))],
            )
        ],
        {"X": _from(0, 0.0)},
    ),
    # control: the jump cancelled it, as its trigger fell
    "a trigger that reads it, not persistent": (
        [
            _event(
                "e",
                f"<apply><and/>{AT_2}{_rel('gt', _ci('Sn'), _cn(1.5))}</apply>",
                [("X", _cn(7))],
                persistent=False,
            )
        ],
        {"X": _from(0, 0.0)},
    ),
    # Y is frozen at the instant too: before e sets X, after the jump
    "two events in turn": (
        [
            _event("e", AT_2, [("X", _ci("Sn"))], priority=_cn(2)),
            _event("f", AT_2, [("Y", _plus(_ci("X"), _ci("Sn")))], priority=_cn(1)),
        ],
        {"X": _from(2, 1.0), "Y": _from(2, 1.0)},
    ),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(AT_THE_JUMP))
def test_an_event_at_the_jump_reads_the_concentration_after_it(case, codegen):
    events, closed_forms = AT_THE_JUMP[case]
    xml = _doc([("cv", T_STEP)], events="".join(events), species=_sp("X") + _sp("Y"))
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", codegen=codegen
    ).run(t_span=(0.0, 4.0), n_points=17)
    t = np.asarray(result.time)
    names = list(result.species_names)
    for sid, closed_form in closed_forms.items():
        got = np.asarray(result.species)[:, names.index(sid)]
        np.testing.assert_allclose(got, closed_form(t), rtol=1e-5, atol=1e-8, err_msg=sid)


def _sensitivities(events):
    xml = _doc([("cv", T_STEP)], events="".join(events), species=_sp("X") + _sp("Y"))
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", sensitivity_params=["k"]
    )
    return sim.run(t_span=(0.0, 4.0), n_points=17)


def test_the_sensitivity_of_a_value_frozen_after_the_jump():
    """``X = k*Sn`` at the jump: dX/dk is [Sn] at t = 2, 1. Read before the
    jump it was 5/3, beside an X of 5/6."""
    result = _sensitivities([_event("e", AT_2, [("X", _times(_ci("k"), _ci("Sn")))])])
    t = np.asarray(result.time)
    x = list(result.species_names).index("X")
    np.testing.assert_allclose(
        np.asarray(result.species)[:, x], _from(2, 0.5)(t), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[:, x, 0], _from(2, 1.0)(t), rtol=1e-5, atol=1e-8
    )


def test_the_sensitivity_of_a_value_frozen_between_two_fires_is_refused():
    """f's values are frozen after the jump and before e sets X, which they
    read: that state is neither the one before the batch nor the one at f, the
    two the sensitivity jump reads at."""
    events = AT_THE_JUMP["two events in turn"][0]
    with pytest.raises(bngsim.SimulationError, match=r"frozen between two fires of one instant"):
        _sensitivities(events)


def test_run_diffrax_names_the_jump():
    """The model has no event of its own; the two are the jump's, each way."""
    pytest.importorskip("diffrax")
    from bngsim._diffrax_solver import run_diffrax

    model = bngsim.Model.from_sbml_string(_doc([("cv", S_SWITCH)]))
    assert model.n_events == 2
    with pytest.raises(ValueError, match=r"keeps the amounts in compartment cv across a jump"):
        run_diffrax(model, t_end=4.0)


def test_a_condition_an_event_also_moves_is_refused():
    """``t > p`` flips with the clock and when the event sets p: the event's
    own rescale (#74) and the jump's would both apply, so it is refused."""
    size = _pw((_cn(2), _rel("gt", T, _ci("p"))), otherwise=_cn(1))
    event = (
        '<event id="e" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true"><math {M}>'
        f"{_rel('gt', T, _cn(0.5))}</math></trigger>"
        f'<listOfEventAssignments><eventAssignment variable="p"><math {M}>{_cn(0.25)}'
        "</math></eventAssignment></listOfEventAssignments></event>"
    )
    xml = _doc(
        [("cv", size)],
        params='<parameter id="p" value="3" constant="false"/>',
        events=event,
    )
    with pytest.raises(bngsim.ModelError, match=r"p, which an event changes"):
        bngsim.Model.from_sbml_string(xml)
