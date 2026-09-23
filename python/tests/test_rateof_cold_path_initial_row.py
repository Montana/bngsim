"""GH #567 — the cold path's t=0 row must probe dx/dt like every other row.

A ``rate_of__<species>`` accessor reads ``model.current_derivs``, which is
refreshed only as a side effect of an RHS evaluation. At t=0, before any step
has been taken, that buffer holds whatever the last run left in it: zero on a
fresh model, an arbitrary stale derivative on a reused one.

The warm path probes dx/dt for every row it records, the initial one included,
and the cold path does the same at each output point — but not for its initial
row. So a cold run of a rateOf model reported a wrong first sample and a
correct rest, which reads as a transient rather than as a bug.

The cold path is not a corner: a forward-sensitivity run, ``jacobian="jax"``, a
non-empty ``crossing_stops`` and ``BNGSIM_NO_WARM_CVODE`` all take it. The
model below has an assignment rule ``r := rateOf(A)`` over ``A' = -k·A``, so
``r(0)`` is ``-k·A(0)`` exactly, and the two paths are compared against that
number rather than against each other alone.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest

_RATEOF_CSYMBOL = (
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/rateOf"> rateOf </csymbol>'
)

K = 0.7
A0 = 10.0
EXPECT_R0 = -K * A0  # -7.0

SBML = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfParameters>
      <parameter id="k" value="{K}" constant="true"/>
      <parameter id="r" constant="false"/>
    </listOfParameters>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfRules>
      <assignmentRule variable="r"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply></math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci><ci>c</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


@pytest.fixture
def sbml_path(tmp_path):
    p = tmp_path / "rateof.xml"
    p.write_text(SBML)
    return str(p)


def _series(result, name):
    """A scalar trajectory by SBML id, from whichever column kind carries it."""
    for names, data in (("species_names", "species"), ("observable_names", "observables")):
        col = list(getattr(result, names))
        if name in col:
            return np.asarray(getattr(result, data))[:, col.index(name)]
    if name in list(result.expression_names):
        return np.asarray(result.expressions[name])
    raise AssertionError(f"{name!r} is in no column of this result")


def _run(sbml_path, *, model=None, **kw):
    with contextlib.redirect_stderr(io.StringIO()):
        m = model if model is not None else bngsim.Model.from_sbml(sbml_path)
        return bngsim.Simulator(m, method="ode", **kw).run(t_span=(0.0, 2.0), n_points=5)


# ── The initial row ──────────────────────────────────────────────────────────


def test_the_warm_path_is_right_to_begin_with(sbml_path):
    """The reference the cold path is held to; it probes every row already."""
    assert _series(_run(sbml_path), "r")[0] == pytest.approx(EXPECT_R0, rel=1e-9)


def test_the_cold_path_initial_row_is_the_real_derivative(sbml_path):
    """The defect: this read a stale buffer and reported 0.0 for -7.0."""
    cold = _series(_run(sbml_path, sensitivity_params=["k"]), "r")
    assert cold[0] == pytest.approx(EXPECT_R0, rel=1e-9)


def test_the_cold_path_initial_row_is_right_without_warm_cvode(sbml_path, monkeypatch):
    """A second way into the cold path, so the fix is not tied to one trigger."""
    monkeypatch.setenv("BNGSIM_NO_WARM_CVODE", "1")
    assert _series(_run(sbml_path), "r")[0] == pytest.approx(EXPECT_R0, rel=1e-9)


def test_a_stale_non_zero_buffer_is_not_reported_either(sbml_path):
    """A fresh model's buffer is zero, which is a recognisable wrong answer. A
    model that has already been run carries a real derivative from wherever it
    left off — a plausible wrong answer, which is worse."""
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_sbml(sbml_path)
    _run(sbml_path, model=model)  # leaves current_derivs at t=2's value
    model.reset()
    cold = _series(_run(sbml_path, model=model, sensitivity_params=["k"]), "r")
    assert cold[0] == pytest.approx(EXPECT_R0, rel=1e-9)


# ── The rest of the trajectory is untouched ──────────────────────────────────


def test_the_two_paths_agree_over_the_whole_run(sbml_path):
    warm = _series(_run(sbml_path), "r")
    cold = _series(_run(sbml_path, sensitivity_params=["k"]), "r")
    assert cold == pytest.approx(warm, rel=1e-6, abs=1e-8)


def test_the_column_follows_the_closed_form(sbml_path):
    """r(t) = -k·A(0)·exp(-k·t) — the analytic check, so agreement between the
    two paths cannot be agreement on the same wrong numbers."""
    res = _run(sbml_path, sensitivity_params=["k"])
    t = np.asarray(res.time)
    assert _series(res, "r") == pytest.approx(-K * A0 * np.exp(-K * t), rel=1e-5, abs=1e-7)


def test_a_model_without_rateof_is_unaffected(tmp_path):
    """The refresh is guarded on uses_rateof(), so a plain model does no extra
    work and records exactly what it did before."""
    plain = SBML.replace(
        f"""<assignmentRule variable="r"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply></math></assignmentRule>""",
        """<assignmentRule variable="r"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><cn>2</cn><ci>A</ci></apply></math></assignmentRule>""",
    )
    p = tmp_path / "plain.xml"
    p.write_text(plain)
    warm = _series(_run(str(p)), "r")
    cold = _series(_run(str(p), sensitivity_params=["k"]), "r")
    assert warm[0] == pytest.approx(2 * A0, rel=1e-9)
    assert cold == pytest.approx(warm, rel=1e-6, abs=1e-8)
