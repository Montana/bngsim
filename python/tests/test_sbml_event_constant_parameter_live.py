"""GH #835 — event delays and priorities must read live parameter values.

SBML evaluates both quantities when an event trigger fires. The loader must not
freeze a parameter-bearing expression at model-load time: callers can update a
constant parameter with ``set_param`` or scan it before each simulation.
"""

import bngsim
import numpy as np
import pytest

_MATHML = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
)


def _time_gt(value: float) -> str:
    return f"<apply><gt/>{_TIME}<cn>{value}</cn></apply>"


def _model(params: str, events: str, initial_assignments: str = "") -> str:
    initial_assignments_xml = ""
    if initial_assignments:
        initial_assignments_xml = (
            f"<listOfInitialAssignments>{initial_assignments}</listOfInitialAssignments>"
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="issue835">
    <listOfParameters>
      <parameter id="y" value="0" constant="false"/>
      {params}
    </listOfParameters>
    {initial_assignments_xml}
    <listOfEvents>{events}</listOfEvents>
  </model>
</sbml>"""


def _event(
    eid: str,
    var: str,
    value_math: str,
    *,
    delay_math: str = "",
    priority_math: str = "",
    use_trigger_values: bool = True,
) -> str:
    delay = f"<delay><math {_MATHML}>{delay_math}</math></delay>" if delay_math else ""
    priority = (
        f"<priority><math {_MATHML}>{priority_math}</math></priority>" if priority_math else ""
    )
    return f"""
      <event id="{eid}" useValuesFromTriggerTime="{"true" if use_trigger_values else "false"}">
        <trigger initialValue="false" persistent="true">
          <math {_MATHML}>{_time_gt(1)}</math>
        </trigger>
        {delay}{priority}
        <listOfEventAssignments>
          <eventAssignment variable="{var}"><math {_MATHML}>{value_math}</math></eventAssignment>
        </listOfEventAssignments>
      </event>"""


def _run(sbml: str, end: float = 4.0, *, sensitivity_params=()):
    model = bngsim.Model.from_sbml_string(sbml)
    result = bngsim.Simulator(
        model, method="ode", sensitivity_params=list(sensitivity_params)
    ).run(t_span=(0, end), n_points=int(end * 100) + 1, rtol=1e-10, atol=1e-12)
    return model, result


def _first_nonzero_time(result, name: str = "y") -> float:
    index = result.species_names.index(name)
    active = np.flatnonzero(result.species[:, index] > 0.5)
    assert active.size, "event did not execute"
    return float(result.time[active[0]])


@pytest.mark.parametrize("delay", [0.5, 1.0, 2.0])
def test_set_param_changes_constant_parameter_delay(delay):
    event = _event("E", "y", _TIME, delay_math="<ci>d</ci>", use_trigger_values=False)
    sbml = _model('<parameter id="d" value="1" constant="true"/>', event)
    model = bngsim.Model.from_sbml_string(sbml)
    model.set_param("d", delay)
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 4), n_points=401, rtol=1e-10, atol=1e-12
    )
    assert _first_nonzero_time(result) == pytest.approx(1.0 + delay, abs=0.011)


def test_initial_assignment_parameter_dependency_is_live_after_set_param():
    event = _event("E", "y", _TIME, delay_math="<ci>d</ci>", use_trigger_values=False)
    sbml = _model(
        '<parameter id="k" value="0.5" constant="true"/>'
        '<parameter id="d" value="1" constant="true"/>',
        event,
        f'<initialAssignment symbol="d"><math {_MATHML}>'
        "<apply><times/><cn>2</cn><ci>k</ci></apply>"
        "</math></initialAssignment>",
    )
    model = bngsim.Model.from_sbml_string(sbml)
    model.set_param("k", 1.0)
    assert model.get_param("d") == pytest.approx(2.0)
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 4), n_points=401, rtol=1e-10, atol=1e-12
    )
    assert _first_nonzero_time(result) == pytest.approx(3.0, abs=0.011)


def test_parameter_scan_changes_constant_parameter_delay():
    event = _event("E", "y", _TIME, delay_math="<ci>d</ci>", use_trigger_values=False)
    model = bngsim.Model.from_sbml_string(
        _model('<parameter id="d" value="1" constant="true"/>', event)
    )
    sim = bngsim.Simulator(model, method="ode")
    results = sim.parameter_scan("d", [0.5, 1.0, 2.0], t_span=(0, 4), n_points=401)
    for result, delay in zip(results, [0.5, 1.0, 2.0], strict=True):
        assert _first_nonzero_time(result) == pytest.approx(1.0 + delay, abs=0.011)


def _priority_model(p_value: float) -> str:
    ea = _event("A", "y", "<cn>1</cn>", priority_math="<ci>pa</ci>")
    eb = _event("B", "y", "<cn>2</cn>", priority_math="<cn>1</cn>")
    return _model(f'<parameter id="pa" value="{p_value}" constant="true"/>', ea + eb)


@pytest.mark.parametrize(("priority", "expected"), [(0.5, 1.0), (2.0, 2.0), (2.5, 2.0)])
def test_set_param_changes_constant_parameter_priority(priority, expected):
    model = bngsim.Model.from_sbml_string(_priority_model(2.0))
    model.set_param("pa", priority)
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 2), n_points=21, rtol=1e-10, atol=1e-12
    )
    assert float(result.species[-1, result.species_names.index("y")]) == pytest.approx(expected)


def test_parameter_scan_changes_constant_parameter_priority():
    model = bngsim.Model.from_sbml_string(_priority_model(2.0))
    results = bngsim.Simulator(model, method="ode").parameter_scan(
        "pa", [0.5, 2.5], t_span=(0, 2), n_points=21
    )
    observed = [float(r.species[-1, r.species_names.index("y")]) for r in results]
    assert observed == pytest.approx([1.0, 2.0])


def test_zero_constant_parameter_delay_does_not_block_sensitivities():
    event = _event("E", "y", "<cn>1</cn>", delay_math="<ci>d</ci>")
    sbml = _model('<parameter id="d" value="0" constant="true"/>', event)
    _, result = _run(sbml, end=2, sensitivity_params=("d",))
    assert result.species[-1, result.species_names.index("y")] == pytest.approx(1.0)
