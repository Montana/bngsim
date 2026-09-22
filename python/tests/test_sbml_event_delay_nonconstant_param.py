"""GH #558 — an event ``<delay>`` or ``<priority>`` that reads a parameter something
changes during the run is evaluated when the trigger fires, not folded to t=0.

SBML L3v2 §4.11.4 evaluates a delay "at the time the trigger transitions to true",
from the state at that moment; §4.11.3 does the same for a priority. The loader
constant-folds both opportunistically, and its fold context held *every* global
parameter's declared ``value`` (plus every ``initialAssignment`` / assignment-rule
value at t=0). A delay reading a parameter driven by a rate rule, an assignment
rule or another event therefore folded to that parameter's initial value, and the
event fired at the wrong time — silently, with no warning.

The fold context is now restricted to the parameters nothing writes
(``_const_param_ids``, the same predicate the initial-condition seed uses). A
delay or priority that reads anything else takes the existing ``delay_expr`` /
``priority_expr`` path, which the C++ event dispatcher evaluates at trigger time.

Every model below has one event that triggers at ``time > 1`` and sets ``w`` to 1
after a delay of ``d``, where ``d`` is 0.05 at t=0 and 1.0 by t=1. Evaluated at
trigger time the event fires at t=2.0; folded to t=0 it fired at t=1.05.
"""

import bngsim
import numpy as np
import pytest

_MATHML = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
)


def _time_gt(t: float) -> str:
    return f"<apply><gt/>{_TIME}<cn>{t}</cn></apply>"


def _sbml(params: str, rules: str = "", events: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m558">
    <listOfParameters>
      <parameter id="w" value="0" constant="false"/>
      {params}
    </listOfParameters>
    {f"<listOfRules>{rules}</listOfRules>" if rules else ""}
    <listOfEvents>{events}</listOfEvents>
  </model>
</sbml>"""


def _event(
    eid: str,
    trigger_ml: str,
    var: str,
    value_ml: str,
    *,
    delay_ml: str = "",
    priority_ml: str = "",
) -> str:
    delay = f"<delay><math {_MATHML}>{delay_ml}</math></delay>" if delay_ml else ""
    priority = f"<priority><math {_MATHML}>{priority_ml}</math></priority>" if priority_ml else ""
    return f"""
      <event id="{eid}" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math {_MATHML}>{trigger_ml}</math>
        </trigger>
        {delay}
        {priority}
        <listOfEventAssignments>
          <eventAssignment variable="{var}">
            <math {_MATHML}>{value_ml}</math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>"""


_FIRE_ON_D = _event("E", _time_gt(1), "w", "<cn>1</cn>", delay_ml="<ci>d</ci>")


def _w(sbml: str, t_end: float = 3.0, n_points: int = 301) -> tuple[np.ndarray, np.ndarray]:
    model = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, t_end), n_points=n_points, rtol=1e-10, atol=1e-12
    )
    return np.asarray(r.time), np.asarray(r.observables["w"])


def _fire_time(sbml: str) -> float:
    t, w = _w(sbml)
    fired = np.nonzero(w > 0.5)[0]
    assert fired.size, "event never fired"
    return float(t[fired[0]])


def test_literal_delay_control():
    """Control: a literal delay was always right, and stays right."""
    sbml = _sbml("", events=_event("E", _time_gt(1), "w", "<cn>1</cn>", delay_ml="<cn>1</cn>"))
    assert _fire_time(sbml) == pytest.approx(2.0, abs=0.011)


def test_constant_parameter_delay_still_folds():
    """A parameter nothing writes is still a constant: it folds as before, even when
    declared ``constant="false"`` (COPASI emits that routinely, #379)."""
    sbml = _sbml('<parameter id="d" value="1" constant="false"/>', events=_FIRE_ON_D)
    assert _fire_time(sbml) == pytest.approx(2.0, abs=0.011)


def test_delay_reads_rate_rule_parameter():
    """Case A of the issue: ``d' = 0.95`` from d(0)=0.05, so d(1) = 1.0."""
    sbml = _sbml(
        '<parameter id="d" value="0.05" constant="false"/>',
        rules=f'<rateRule variable="d"><math {_MATHML}><cn>0.95</cn></math></rateRule>',
        events=_FIRE_ON_D,
    )
    assert _fire_time(sbml) == pytest.approx(2.0, abs=0.011)


def test_delay_reads_assignment_rule_parameter():
    """Case B of the issue: ``d := 0.05 + 0.95*time``, so d(1) = 1.0."""
    rule = (
        f'<assignmentRule variable="d"><math {_MATHML}>'
        f"<apply><plus/><cn>0.05</cn><apply><times/><cn>0.95</cn>{_TIME}</apply></apply>"
        "</math></assignmentRule>"
    )
    sbml = _sbml(
        '<parameter id="d" value="0.05" constant="false"/>', rules=rule, events=_FIRE_ON_D
    )
    assert _fire_time(sbml) == pytest.approx(2.0, abs=0.011)


def test_delay_reads_event_assigned_parameter():
    """A third writer the fold context also ignored: another event sets d=1 at t=0.5,
    before E triggers at t=1."""
    set_d = _event("SetD", _time_gt(0.5), "d", "<cn>1</cn>")
    sbml = _sbml('<parameter id="d" value="0.05" constant="false"/>', events=set_d + _FIRE_ON_D)
    assert _fire_time(sbml) == pytest.approx(2.0, abs=0.011)


def test_delay_reads_initial_assignment_of_constant_parameter():
    """An initialAssignment on a parameter nothing else writes is a constant, and
    still folds (the IA value, not the declared one)."""
    sbml = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m558ia">
    <listOfParameters>
      <parameter id="w" value="0" constant="false"/>
      <parameter id="d" value="0.05" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="d"><math {_MATHML}><cn>1</cn></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfEvents>{_FIRE_ON_D}</listOfEvents>
  </model>
</sbml>"""
    assert _fire_time(sbml) == pytest.approx(2.0, abs=0.011)


def test_priority_reads_rate_rule_parameter():
    """The same defect in ``<priority>``. Two events trigger together at t>1 and both
    write w; the one that fires last wins. Ea's priority is ``p``, which a rate rule
    drives from 0 to 3 by t=1; Eb's is a constant 2.

    At trigger time p=3 > 2, so Ea fires first and Eb last: w=20. Folded to p(0)=0,
    Ea fired last: w=10.
    """
    ea = _event("Ea", _time_gt(1), "w", "<cn>10</cn>", priority_ml="<ci>p</ci>")
    eb = _event("Eb", _time_gt(1), "w", "<cn>20</cn>", priority_ml="<cn>2</cn>")
    sbml = _sbml(
        '<parameter id="p" value="0" constant="false"/>',
        rules=f'<rateRule variable="p"><math {_MATHML}><cn>3</cn></math></rateRule>',
        events=ea + eb,
    )
    _, w = _w(sbml, t_end=2.0, n_points=21)
    assert float(w[-1]) == pytest.approx(20.0)
