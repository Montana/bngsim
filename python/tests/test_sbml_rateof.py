"""GH #106 — SBML ``rateOf`` csymbol (live species derivative dx/dt).

``rateOf(x)`` is the instantaneous time-derivative of ``x`` — a quantity only
the integrator knows. bngsim normalizes BOTH SBML encodings to a single
per-species accessor (``rate_of__<species>``) and publishes the live dx/dt into
a buffer via a derivative *probe* before every RHS / trigger evaluation:

  * the official csymbol — libsbml ``AST_FUNCTION_RATE_OF`` (type 323);
  * the COPASI ``functionDefinition`` idiom — a funcDef whose body is
    ``<notanumber/>`` carrying a ``…/Derivative`` annotation.

One probe is exact because every rateOf argument is a species whose derivative
is independent of the rateOf consumers (no algebraic loop). See
``dev/notes/gh106_rateof_kickoff.md`` for the full design.

The primary regression cases below use **closed-form oracles** (dependency-free)
because the rr_parity BioModels corpus does not live on this branch. An optional
roadrunner-gated section cross-checks the four real corpus models when both
roadrunner and the corpus are present locally.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest

# rateOf csymbol MathML (official encoding). libsbml parses this to type 323.
_RATEOF_CSYMBOL = (
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/rateOf"> rateOf </csymbol>'
)


def _sbml(model_body: str) -> str:
    """Wrap a ``<model>`` body in a minimal SBML L3V2 document."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
        'level="3" version="2">\n'
        f"{model_body}\n"
        "</sbml>\n"
    )


def _decay_reaction(species: str = "A", rate_param: str = "k") -> str:
    """A first-order decay reaction ``species -> ∅`` with rate ``k*species*c``
    (compartment ``c`` size 1, so the kinetic law is k·[species])."""
    return f"""
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="{species}" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>{rate_param}</ci><ci>{species}</ci><ci>c</ci></apply>
        </math></kineticLaw>
      </reaction>"""


def _series(result, name: str) -> np.ndarray | None:
    """Pull a named series from a result as species or function, else None."""
    sp = list(result.species_names)
    if name in sp:
        return np.asarray(result.species[:, sp.index(name)])
    fn = list(getattr(result, "function_names", []) or [])
    if name in fn:
        return np.asarray(result.functions[:, fn.index(name)])
    return None


def _run(model, *, codegen: bool = False, t_end: float = 5.0, n_points: int = 11):
    return bngsim.Simulator(model, method="ode", codegen=codegen).run(
        t_span=(0.0, t_end), n_points=n_points
    )


# ─── Closed-form oracles (always run) ────────────────────────────────────────


@pytest.mark.parametrize("codegen", [False, True])
def test_rate_rule_rateof_tracks_its_argument(codegen):
    """A rate rule ``B' = rateOf(A)`` makes B integrate dA/dt, so d(B-A)/dt = 0
    and ``B - A`` is conserved exactly — a derivative-free oracle for
    ``rateOf(A) == dA/dt``. The decay also lets us check the closed form
    ``A(t) = A0·exp(-k·t)``. Exercises the official csymbol on the rate-rule /
    functional-reaction RHS path (BIOMD0000000696 pattern), interpreted and
    codegen.
    """
    A0, k = 10.0, 0.7
    body = f"""
  <model id="rate_rule_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{k}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is True

    res = _run(model, codegen=codegen)
    t = np.asarray(res.time)
    A, B = _series(res, "A"), _series(res, "B")

    # rateOf(A) == dA/dt: B integrates it, so B - A holds its initial value.
    assert np.allclose(B - A, -A0, atol=1e-9, rtol=0)
    # And the species itself follows the analytic decay.
    assert np.allclose(A, A0 * np.exp(-k * t), atol=0, rtol=1e-5)


@pytest.mark.parametrize("codegen", [False, True])
def test_assignment_rule_rateof_feeds_dynamics(codegen):
    """An assignment rule ``r := rateOf(A)`` whose value drives a second state
    (``C' = -r = k·A``) — the BIOMD0000000775 pattern where rateOf is consumed
    *inside* the RHS. Since ``C' = -A' ``, ``C + A`` is conserved at ``A0``. This
    is the case that needs rateOf live during ``compute_derivs`` (not just in a
    trigger).
    """
    A0, k = 10.0, 0.7
    body = f"""
  <model id="assign_rule_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="C" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="{k}" constant="true"/>
      <parameter id="r" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="r"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply>
      </math></assignmentRule>
      <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><ci>r</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is True

    res = _run(model, codegen=codegen)
    A, C = _series(res, "A"), _series(res, "C")
    assert np.allclose(C + A, A0, atol=1e-9, rtol=0)


def test_event_trigger_on_rateof_fires_at_predicted_time():
    """A trigger gated on ``rateOf(A) > -1`` with ``A' = -A`` (k=1). rateOf(A)
    = -A starts at -10 (< -1, false) and rises toward 0, crossing -1 when
    ``A = 1`` ⇒ ``t = ln(10)``. The event sets a reported species ``marker := 1``.

    Also pins the t=0 initialization fix: with a zero-initialized derivative
    buffer the trigger would read ``0 > -1`` = true and fire spuriously at t=0.
    """
    body = f"""
  <model id="event_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="marker" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="1" constant="true"/></listOfParameters>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
    <listOfEvents>
      <event id="ev1" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply><cn>-1</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="marker">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is True

    res = _run(model, t_end=5.0, n_points=501)
    t = np.asarray(res.time)
    marker = _series(res, "marker")

    assert marker[0] == pytest.approx(0.0)  # no spurious t=0 fire
    fire_t = t[int(np.argmax(marker > 0.5))]
    assert fire_t == pytest.approx(math.log(10.0), abs=2 * (t[1] - t[0]))


def test_copasi_funcdef_idiom_normalized_to_derivative():
    """The COPASI rateOf idiom — a ``functionDefinition`` whose body is
    ``<notanumber/>`` with a ``…/Derivative`` annotation — must be intercepted
    and emit the live derivative rather than inlining NaN. Same B-A invariant as
    the csymbol case, driven through the funcDef call ``rateOf(A)``.
    """
    A0, k = 8.0, 0.5
    body = f"""
  <model id="copasi_idiom">
    <listOfFunctionDefinitions>
      <functionDefinition id="rateOf">
        <annotation>
          <symbols xmlns="http://sbml.org/annotations/symbols"
                   definition="http://en.wikipedia.org/wiki/Derivative"/>
        </annotation>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda><bvar><ci>a</ci></bvar><notanumber/></lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{k}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><ci>rateOf</ci><ci>A</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is True
    res = _run(model)
    A, B = _series(res, "A"), _series(res, "B")
    assert np.all(np.isfinite(B))  # NaN body must NOT have been inlined
    assert np.allclose(B - A, -A0, atol=1e-9, rtol=0)


def test_unused_rateof_funcdef_does_not_enable_feature():
    """A model that *defines* a rateOf idiom funcDef but never *calls* it
    (MODEL2403070001 pattern) must stay on the byte-identical non-rateOf path —
    ``uses_rateof`` is False and the dead funcDef is simply never consumed.
    """
    body = (
        """
  <model id="unused_rateof">
    <listOfFunctionDefinitions>
      <functionDefinition id="rateOf">
        <annotation>
          <symbols xmlns="http://sbml.org/annotations/symbols"
                   definition="http://en.wikipedia.org/wiki/Derivative"/>
        </annotation>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda><bvar><ci>a</ci></bvar><notanumber/></lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3" constant="true"/></listOfParameters>
    <listOfReactions>"""
        + _decay_reaction()
        + """</listOfReactions>
  </model>"""
    )
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model._core.uses_rateof is False
    res = _run(model)
    A = _series(res, "A")
    assert np.allclose(A, 10.0 * np.exp(-0.3 * np.asarray(res.time)), atol=0, rtol=1e-5)


def test_ssa_rejects_rateof_model():
    """rateOf has no defined value in a stochastic trajectory; SSA must reject
    a rateOf model loudly rather than feed a stale/zero derivative."""
    A0, k = 10.0, 0.7
    body = f"""
  <model id="ssa_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{k}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    with pytest.raises(Exception, match="rateOf"):
        bngsim.Simulator(model, method="ssa").run(t_span=(0.0, 1.0), n_points=2)


def test_clone_preserves_rateof():
    """A clone (used by every parallel worker) must rebind the rateOf accessors
    to its own derivative buffer and reproduce the original trajectory."""
    from bngsim._model import Model

    A0, k = 10.0, 0.7
    body = f"""
  <model id="clone_rateof">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{k}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply>
      </math></rateRule>
    </listOfRules>
    <listOfReactions>{_decay_reaction()}</listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    clone = Model(_core=model._core.clone())
    assert clone._core.uses_rateof is True

    r0 = _run(model)
    r1 = _run(clone)
    assert np.allclose(_series(r0, "B"), _series(r1, "B"), atol=0, rtol=0)


# ─── Optional corpus cross-check vs libRoadRunner ────────────────────────────
#
# The four rr_parity models that use rateOf, with the libRoadRunner oracle. The
# corpus is not committed on this branch, so each case skips unless its file and
# roadrunner are both present locally.

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"

# (model dir, xml filename, [(series, max_rel_tol)], t_end, n_points)
_CORPUS = [
    ("MODEL1910030001", "MODEL1910030001.xml", [("A", 1e-3), ("N_2", 1e-3)], 100.0, 21),
    ("BIOMD0000000775", "Iarosz2015.xml", [("G", 1e-6), ("N", 1e-6)], 50.0, 11),
    (
        "BIOMD0000000696",
        "BIOMD0000000696_url.xml",
        [("x8", 1e-4), ("sum_abs_dx8", 1e-4)],
        50.0,
        11,
    ),
]


@pytest.mark.parametrize("model_dir,xml,checks,t_end,n_points", _CORPUS)
def test_corpus_rateof_matches_roadrunner(model_dir, xml, checks, t_end, n_points):
    roadrunner = pytest.importorskip("roadrunner")
    path = _MODELS_DIR / model_dir / xml
    if not path.exists():
        pytest.skip(f"rr_parity corpus model not present: {path}")
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)

    sel = [name for name, _tol in checks]
    rr = roadrunner.RoadRunner(str(path))
    rr.timeCourseSelections = ["time"] + sel
    ref = np.asarray(rr.simulate(0.0, t_end, n_points))

    model = bngsim.Model.from_sbml(str(path))
    assert model._core.uses_rateof is True
    res = _run(model, t_end=t_end, n_points=n_points)

    for i, (name, tol) in enumerate(checks):
        got = _series(res, name)
        assert got is not None, f"{name} not found in bngsim result"
        want = ref[:, 1 + i]
        max_rel = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-9))
        assert max_rel < tol, f"{model_dir}:{name} max_rel={max_rel:.3e} >= {tol}"


# ─── GH #231: per-row refresh + local-param / no-math-rate-rule rateOf ────────
#
# The recorded value of a rate_of__<species> accessor was the last *internal*
# integration step's derivative — and at t=0, before any step, a stale (zero)
# buffer. fill_row / the event loop now probe dx/dt at the exact recorded
# (t, y) so every row is exact. Plus two loader fixes: a kinetic-law local
# parameter is constant so rateOf(local) ≡ 0 (it also has no species index, so
# the accessor would mis-bind or fail to compile); and a rate rule with no
# <math> means dvar/dt = 0, which still promotes var to a species so a
# rateOf(var) accessor binds.


def _read_value(result, name):
    """A scalar trajectory by SBML id from any column kind."""
    for names, data in (
        ("species_names", "species"),
        ("observable_names", "observables"),
    ):
        col = list(getattr(result, names))
        if name in col:
            return np.asarray(getattr(result, data))[:, col.index(name)]
    exprs = list(result.expression_names)
    if name in exprs:
        return np.asarray(result.expressions[name])
    return None


def test_rateof_recorded_value_exact_at_t0():
    # p1 rate-ruled dp1/dt = 0.5*p1, p1(0)=2 ⇒ rateOf(p1)(0) = 1 exactly. Before
    # the per-row refresh the t=0 row read a stale derivative buffer.
    body = f"""
      <model id="m">
        <listOfParameters>
          <parameter id="p1" value="2" constant="false"/>
          <parameter id="p2" constant="false"/>
        </listOfParameters>
        <listOfRules>
          <rateRule variable="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><cn>0.5</cn><ci>p1</ci></apply></math></rateRule>
          <assignmentRule variable="p2"><math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>{_RATEOF_CSYMBOL}<ci>p1</ci></apply></math></assignmentRule>
        </listOfRules>
      </model>"""
    res = _run(bngsim.Model.from_sbml_string(_sbml(body)), t_end=4.0, n_points=5)
    p1 = _read_value(res, "p1")
    p2 = _read_value(res, "p2")
    assert p2 is not None
    # rateOf(p1) == 0.5*p1 at every recorded row, including t=0.
    np.testing.assert_allclose(p2, 0.5 * p1, rtol=1e-7, atol=1e-9)
    assert abs(p2[0] - 1.0) < 1e-9


def test_rateof_of_local_parameter_is_zero():
    # The kinetic law rateOf(k) reads the kinetic-law-local k (constant) ⇒ 0, so
    # S never changes. (Previously load-failed on an unbound rate_of__k.)
    body = f"""
      <model id="m">
        <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
        <listOfSpecies>
          <species id="S" compartment="c" initialAmount="4" hasOnlySubstanceUnits="true"
                   boundaryCondition="false" constant="false"/>
        </listOfSpecies>
        <listOfReactions>
          <reaction id="J0" reversible="false">
            <listOfProducts><speciesReference species="S" stoichiometry="1" constant="true"/>
            </listOfProducts>
            <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>{_RATEOF_CSYMBOL}<ci>k</ci></apply></math>
              <listOfLocalParameters><localParameter id="k" value="3"/></listOfLocalParameters>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>"""
    res = _run(bngsim.Model.from_sbml_string(_sbml(body)), t_end=5.0, n_points=6)
    S = _read_value(res, "S")
    np.testing.assert_allclose(S, 4.0, rtol=0, atol=1e-9)


def test_rateof_of_no_math_rate_rule_is_zero():
    # p has a rate rule with no <math> ⇒ dp/dt = 0 (p constant); x := rateOf(p)
    # must bind and evaluate to 0. (Previously load-failed: p stayed a parameter
    # with no rate_of__p.)
    body = f"""
      <model id="m">
        <listOfParameters>
          <parameter id="p" value="2" constant="false"/>
          <parameter id="x" constant="false"/>
        </listOfParameters>
        <listOfRules>
          <rateRule variable="p"/>
          <assignmentRule variable="x"><math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>{_RATEOF_CSYMBOL}<ci>p</ci></apply></math></assignmentRule>
        </listOfRules>
      </model>"""
    res = _run(bngsim.Model.from_sbml_string(_sbml(body)), t_end=3.0, n_points=4)
    p = _read_value(res, "p")
    x = _read_value(res, "x")
    np.testing.assert_allclose(p, 2.0, rtol=0, atol=1e-12)
    assert x is not None
    np.testing.assert_allclose(x, 0.0, rtol=0, atol=1e-12)


# ─── GH #231 sub-cluster 3 / 01463: rateOf of a hasOnlySubstanceUnits=true species
#
# A hOSU=true species's symbol denotes its AMOUNT, so its rateOf csymbol is the
# amount rate d(amount)/dt. The engine stores amount/V_static (rescaling to live
# volume separately), so the rateOf buffer holds d(amount)/dt / V_static and
# ×volume_factor recovers the amount rate — for a CONSTANT-volume compartment
# (01455 / 01457) AND a VARIABLE-volume compartment promoted to a state (rate-rule
# / event-resized, 01463), with no conc·V̇ correction. Off by exactly the factor V
# before the fix.


def _hosu_production_model(rule_xml: str, *, vol: float = 2.0) -> str:
    """A hOSU=true species ``A`` in a constant V=`vol` compartment, produced by a
    constant-rate reaction ``-> A`` at amount rate 2 (so rateOf(A) == 2, while the
    stored d(conc)/dt == 2/vol). ``rule_xml`` adds the rateOf consumer under test."""
    return f"""
  <model id="hosu_rateof">
    <listOfCompartments><compartment id="c" size="{vol}" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="x" value="0" constant="false"/></listOfParameters>
    <listOfRules>{rule_xml}</listOfRules>
    <listOfReactions>
      <reaction id="prod" reversible="false">
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn>2</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""


@pytest.mark.parametrize("codegen", [False, True])
def test_rateof_hosu_amount_rate_in_rhs(codegen):
    """rateOf(A) consumed *inside* the RHS — a rate rule ``B' = rateOf(A)`` (B
    hOSU=false). B integrates the amount rate 2, so ``B(t) = 2t`` exactly (it
    would be ``t`` if rateOf reported the unscaled d(conc)/dt = 1). Exercises the
    interpreted buffer scaling and the codegen rateOf map in lock-step."""
    rule = (
        '<rateRule variable="B"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"<apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply></math></rateRule>"
    )
    model = bngsim.Model.from_sbml_string(_sbml(_hosu_production_model(rule)))
    assert model._core.uses_rateof is True
    res = _run(model, codegen=codegen, t_end=5.0, n_points=11)
    t = np.asarray(res.time)
    B = _series(res, "B")
    np.testing.assert_allclose(B, 2.0 * t, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("codegen", [False, True])
def test_rateof_hosu_amount_rate_as_output(codegen):
    """The 01455 / 01457 pattern: ``x := rateOf(A)`` reported as an output. x must
    read the amount rate 2 (d(amount)/dt) at every row, not 2/V = 1. Pins the
    output recorder — the path the SBML test suite grades."""
    rule = (
        '<assignmentRule variable="x"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"<apply>{_RATEOF_CSYMBOL}<ci>A</ci></apply></math></assignmentRule>"
    )
    model = bngsim.Model.from_sbml_string(_sbml(_hosu_production_model(rule)))
    res = _run(model, codegen=codegen, t_end=10.0, n_points=11)
    x = _read_value(res, "x")
    assert x is not None
    np.testing.assert_allclose(x, 2.0, rtol=1e-6, atol=1e-9)


def test_rateof_hosu_variable_volume_amount_scaled():
    """A hOSU=true species in a VARIABLE-volume rate-ruled compartment IS flagged
    for the amount scaling (01463): the engine stores amount/V_static, so
    ×volume_factor recovers d(amount)/dt with no conc·V̇ term. Pins the gate — both
    the constant-volume and the rate-ruled-compartment hOSU species carry
    report_rateof_amount; an AR-compartment hOSU species (whose volume is a rule
    function, not an integrator state) does not."""
    body = """
  <model id="m">
    <listOfCompartments>
      <compartment id="cc" size="2" constant="true"/>
      <compartment id="cv" size="2" constant="false"/>
      <compartment id="ca" size="2" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="Ac" compartment="cc" initialAmount="1" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="Av" compartment="cv" initialAmount="1" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="Aa" compartment="ca" initialAmount="1" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="q" value="3" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="cv"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <cn>0.1</cn></math></rateRule>
      <assignmentRule variable="ca"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <ci>q</ci></math></assignmentRule>
    </listOfRules>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    flags = {s["name"]: s["report_rateof_amount"] for s in model._core.codegen_data()["species"]}
    assert flags["Ac"] is True  # constant-volume hOSU ⇒ amount-scaled
    assert flags["Av"] is True  # rate-ruled variable-volume hOSU ⇒ amount-scaled (01463)
    assert flags.get("Aa") is False  # AR-compartment hOSU ⇒ left unscaled (not a state)


def test_rateof_hosu_variable_volume_amount_rate_value():
    """01463 end to end: a hOSU species S under a rate rule (d(amount)/dt = 0.4) in
    a rate-ruled compartment C (dV/dt = 0.2, V(0) = 0.5), with ``x := rateOf(S)``.
    The reported amount-rate is the constant 0.4 for all t — NOT 0.4/V(0) = 0.8
    (unscaled) nor a drifting chain-rule value. Pins the ×volume_factor recovery on
    the variable-volume path."""
    body = f"""
  <model id="varvol_rateof">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="0.5" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="true"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="x" constant="false"/></listOfParameters>
    <listOfRules>
      <rateRule variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <cn>0.4</cn></math></rateRule>
      <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <cn>0.2</cn></math></rateRule>
      <assignmentRule variable="x"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>S</ci></apply></math></assignmentRule>
    </listOfRules>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    res = _run(model, t_end=1.0, n_points=11)
    x = _read_value(res, "x")
    assert x is not None
    np.testing.assert_allclose(x, 0.4, rtol=1e-6, atol=1e-9)


# ─── GH #231 sub-cluster 1: rateOf in an initialAssignment ────────────────────
#
# ``p = rateOf(X)`` in an initialAssignment folds to the INITIAL dX/dt: for a
# rate-ruled X the rule's RHS (01250), for a reaction-driven species the signed
# Σ net_stoich·kineticLaw in the species' reporting units (01251/01252/01254). The
# fold reuses the same derivative the runtime rateOf buffer holds; without it the
# IA target silently keeps its declared value.


def test_rateof_in_initial_assignment_rate_rule():
    """01250: ``p2 = rateOf(p1)`` where p1 has rate rule 1 ⇒ p2 folds to 1."""
    body = f"""
  <model id="m">
    <listOfParameters>
      <parameter id="p1" value="2" constant="false"/>
      <parameter id="p2" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p2"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>p1</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <cn>1</cn></math></rateRule>
    </listOfRules>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p2") == pytest.approx(1.0)


def test_rateof_in_initial_assignment_reaction_species():
    """01251 / 01252: ``p2 = rateOf(S)`` folds to the signed reaction rate — negative
    when S is a reactant, positive when a product (kinetic law 1, V=1)."""

    def _model(as_product: bool) -> str:
        ref = "listOfProducts" if as_product else "listOfReactants"
        return f"""
  <model id="m">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="100" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="p2" constant="false"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p2"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>{_RATEOF_CSYMBOL}<ci>S</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <{ref}><speciesReference species="S" stoichiometry="1" constant="true"/></{ref}>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""

    prod = bngsim.Model.from_sbml_string(_sbml(_model(True)))
    reac = bngsim.Model.from_sbml_string(_sbml(_model(False)))
    assert prod.get_param("p2") == pytest.approx(1.0)  # product ⇒ +rate
    assert reac.get_param("p2") == pytest.approx(-1.0)  # reactant ⇒ -rate


# ─── Issue #1045: rateOf of a parameter no rule changes, in an initial value ───
#
# ``p2 = rateOf(p1)`` with p1 a parameter that no rate rule changes is 0 (SBML
# suite 01248: both constant; 01249: both non-constant, but nothing changes p1).
# The §0 fold had that value, but the parameter lift (#732) then lifted p2 into a
# derived parameter reading ``rate_of__p1`` — its rateOf guard fired only when a
# species or assignment-rule substitution was present — and the builder refuses a
# derived parameter that reads a rate accessor (13c9a0e), so the model was
# refused. The species initial-value lift skipped rateOf only when the initial
# value read a state, and had the same defect.


def _rateof_p1_model(
    target: str, p1_constant: str, expr_tail: str = "", *, idiom: bool = False
) -> str:
    """p1 = 1 with no rule; ``target`` (p2 or species S) assigned
    ``rateOf(p1)`` plus ``expr_tail``. ``idiom`` writes the call as COPASI's
    rateOf functionDefinition instead of the csymbol."""
    math = (
        "<apply><ci>rateOf</ci><ci>p1</ci></apply>"
        if idiom
        else f"<apply>{_RATEOF_CSYMBOL}<ci>p1</ci></apply>"
    )
    if expr_tail:
        math = f"<apply><plus/>{math}{expr_tail}</apply>"
    return f"""
  <model id="m">
    {_COPASI_RATEOF_FUNCDEF if idiom else ""}
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="5" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" value="1" constant="{p1_constant}"/>
      <parameter id="p2" value="7" constant="{p1_constant}"/>
      <parameter id="k" value="3" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="{target}"><math xmlns="http://www.w3.org/1998/Math/MathML">
        {math}</math></initialAssignment>
    </listOfInitialAssignments>
  </model>"""


@pytest.mark.parametrize("p1_constant", ["true", "false"], ids=["01248", "01249"])
def test_rateof_of_a_parameter_no_rule_changes_is_zero_in_an_initial_assignment(p1_constant):
    model = bngsim.Model.from_sbml_string(_sbml(_rateof_p1_model("p2", p1_constant)))
    assert model.get_param("p2") == 0.0
    res = _run(model, t_end=1.0, n_points=3)
    assert res.species.shape[0] == 3  # it runs, not just loads


def test_rateof_of_a_parameter_in_a_species_initial_value():
    """The species initial-value lift skipped rateOf only when the initial value
    read a state: ``S = rateOf(p1) + k`` was refused for reading the rate
    accessor; S starts at 0 + 3."""
    model = bngsim.Model.from_sbml_string(_sbml(_rateof_p1_model("S", "true", "<ci>k</ci>")))
    res = _run(model, t_end=1.0, n_points=3)
    np.testing.assert_allclose(_series(res, "S"), 3.0)


def _frozen_warning(caplog) -> str | None:
    """The parameter list the #313 load warning names, or None if it was not
    given."""
    msgs = [r.getMessage() for r in caplog.records if "could not keep symbolic" in r.getMessage()]
    assert len(msgs) <= 1
    if not msgs:
        return None
    # The named list, not a substring of the message ("k" is in "keep").
    named = re.search(r"parameter\(s\) (.+?)\. ", msgs[0])
    assert named
    return named.group(1)


def test_a_folded_rateof_initial_assignment_still_says_what_it_froze(caplog):
    """Not lifted, the assignment keeps its folded value, and the #313 warning
    names what a write to will no longer move it: ``k``, and not ``p1``, whose
    rate is 0 whatever its value (issue #1048)."""
    with caplog.at_level("WARNING", logger="bngsim"):
        model = bngsim.Model.from_sbml_string(_sbml(_rateof_p1_model("p2", "true", "<ci>k</ci>")))
    assert _frozen_warning(caplog) == "k"
    assert model.frozen_params == ["k"]


# ─── Issue #1046: rateOf of an assignmentRule variable ───────────────────────
#
# SBML defines no rate of change for a variable an assignmentRule sets (L3V2
# §3.4.6, validation rule 10224); libsbml and RoadRunner refuse the model. bngsim
# folded ``p2 = rateOf(q)`` in an initialAssignment to 0, a value the model does
# not have, and failed in ExprTk on an unbound ``rate_of__q`` everywhere else.
# No SBML Test Suite case uses the construct.

_TIME_CSYMBOL = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
)
_COPASI_RATEOF_FUNCDEF = """
    <listOfFunctionDefinitions>
      <functionDefinition id="rateOf"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <lambda><bvar><ci>a</ci></bvar><notanumber/></lambda></math></functionDefinition>
    </listOfFunctionDefinitions>"""


def _rateof_q_model(where: str, *, idiom: bool = False, of: str = "q", local_q: bool = False):
    """``q := 2*time`` and ``Y := 3*S`` (assignmentRules) beside ``S -> ∅`` at
    ``k*S``, with ``rateOf(<of>)`` read at ``where``: an initialAssignment,
    assignment or rate rule of p2, the kinetic law of a reaction J1 (``local_q``
    gives J1 its own local q), or an event's trigger or assignment. ``idiom``
    writes the call as COPASI's rateOf functionDefinition."""
    rate = (
        f"<apply><ci>rateOf</ci><ci>{of}</ci></apply>"
        if idiom
        else f"<apply>{_RATEOF_CSYMBOL}<ci>{of}</ci></apply>"
    )
    m = '<math xmlns="http://www.w3.org/1998/Math/MathML">'
    ia = rules = reactions = events = ""
    if where == "initialAssignment:p2":
        ia = (
            f'<listOfInitialAssignments><initialAssignment symbol="p2">{m}{rate}</math>'
            "</initialAssignment></listOfInitialAssignments>"
        )
    elif where == "assignmentRule:p2":
        rules = f'<assignmentRule variable="p2">{m}{rate}</math></assignmentRule>'
    elif where == "rateRule:p2":
        rules = f'<rateRule variable="p2">{m}{rate}</math></rateRule>'
    elif where == "reaction:J1":
        local = (
            '<listOfLocalParameters><localParameter id="q" value="3"/></listOfLocalParameters>'
            if local_q
            else ""
        )
        reactions = (
            '<reaction id="J1" reversible="false"><listOfProducts>'
            '<speciesReference species="S" stoichiometry="1" constant="true"/></listOfProducts>'
            f"<kineticLaw>{m}<apply><plus/>{rate}<cn>1</cn></apply></math>{local}</kineticLaw>"
            "</reaction>"
        )
    elif where in ("event:e:trigger", "event:e:assignment:p2"):
        trigger = f"<apply><gt/>{rate}<cn>1</cn></apply>"
        value = "<cn>1</cn>"
        if where == "event:e:assignment:p2":
            trigger, value = f"<apply><gt/>{_TIME_CSYMBOL}<cn>1</cn></apply>", rate
        events = f"""
    <listOfEvents>
      <event id="e" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">{m}{trigger}</math></trigger>
        <listOfEventAssignments>
          <eventAssignment variable="p2">{m}{value}</math></eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>"""
    else:
        raise ValueError(where)
    return f"""
  <model id="m">
    {_COPASI_RATEOF_FUNCDEF if idiom else ""}
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="Y" compartment="c" initialAmount="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="q" value="0" constant="false"/>
      <parameter id="p2" value="7" constant="false"/>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    {ia}
    <listOfRules>
      <assignmentRule variable="q">{m}<apply><times/><cn>2</cn>{_TIME_CSYMBOL}</apply></math>
      </assignmentRule>
      <assignmentRule variable="Y">{m}<apply><times/><cn>3</cn><ci>S</ci></apply></math>
      </assignmentRule>
      {rules}
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>{m}<apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
      {reactions}
    </listOfReactions>
    {events}
  </model>"""


_Q_RULE = "q is set by the assignmentRule q := 2 * time"


@pytest.mark.parametrize(
    "where",
    [
        "initialAssignment:p2",
        "assignmentRule:p2",
        "rateRule:p2",
        "reaction:J1",
        "event:e:trigger",
        "event:e:assignment:p2",
    ],
)
@pytest.mark.parametrize("idiom", [False, True], ids=["csymbol", "copasi_idiom"])
def test_rateof_of_an_assignment_rule_variable_is_refused_by_name(where, idiom):
    """Each place rateOf can be read is refused with the symbol, the place and
    the rule that sets the symbol. The initialAssignment used to load with
    p2 = 0; the rest failed on an undefined ``rate_of__q``."""
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(_sbml(_rateof_q_model(where, idiom=idiom)))
    msg = str(exc.value)
    assert f"rateOf(q) [{where}]: {_Q_RULE}" in msg
    assert "validation rule 10224" in msg
    assert "rate_of__" not in msg


@pytest.mark.parametrize("where", ["initialAssignment:p2", "rateRule:p2"])
def test_rateof_of_a_species_an_assignment_rule_sets_is_refused(where):
    """A species is no different: ``Y := 3*S`` has no rate of its own in SBML,
    though S has one. Not refused, it was 0 in both places: folded so in the
    initialAssignment, and read so by the rate rule, whose accessor the
    builder binds for a species (p2 stayed 0 where d(3*S)/dt is -15 at t=0)."""
    with pytest.raises(bngsim.ModelError, match=re.escape(f"rateOf(Y) [{where}]: Y")):
        bngsim.Model.from_sbml_string(_sbml(_rateof_q_model(where, of="Y")))


def test_rateof_of_a_kinetic_law_local_shadowing_an_assignment_rule_variable_loads():
    """Inside J1, ``q`` is J1's own constant local parameter, not the global the
    rule sets, so ``rateOf(q)`` there is 0 and J1 produces S at rate 1. (libsbml
    reports rule 10224 here anyway; it does not scope the name.)"""
    model = bngsim.Model.from_sbml_string(_sbml(_rateof_q_model("reaction:J1", local_q=True)))
    res = _run(model, t_end=1.0, n_points=3)
    # dS/dt = 1 - 0.5*S, S(0) = 10  ⇒  S(t) = 2 + 8·exp(-t/2).
    np.testing.assert_allclose(
        _series(res, "S"), 2 + 8 * np.exp(-0.5 * np.asarray(res.time)), rtol=1e-6
    )


# ─── Issue #1048: what a folded rateOf initial value read ─────────────────────
#
# ``rateOf(x)`` does not read x's value. A folded initial value that reads it
# froze x all the same (``set_param`` and the sensitivity request refused it),
# and the #313 warning, which read the expression's names alone, named x and
# missed what the rate is made of. Both now come from one closure: the rate's
# kinetic laws, stoichiometries, compartment or rate rule, and whatever else the
# expression names, x included if it is read outside the rateOf.

_RATE_OF_B = "compartment d = 1; species A in d, B in d; A = 2; B = 0; k1 = 1; p2 = rateOf(B);\n"


@pytest.mark.parametrize("idiom", [False, True], ids=["csymbol", "copasi_idiom"])
@pytest.mark.parametrize("p1_constant", ["true", "false"], ids=["01248", "01249"])
def test_rateof_of_a_parameter_does_not_freeze_the_parameter(p1_constant, idiom, caplog):
    """``p2 = rateOf(p1)`` is 0 whatever ``p1`` is: a write to ``p1`` is
    allowed, ``p2`` stays 0, and no warning names ``p1``. The same written
    with COPASI's idiom, whose body bngsim never evaluates."""
    with caplog.at_level("WARNING", logger="bngsim"):
        model = bngsim.Model.from_sbml_string(
            _sbml(_rateof_p1_model("p2", p1_constant, idiom=idiom))
        )
    assert model.get_param("p2") == 0.0
    assert model.frozen_params == []
    assert _frozen_warning(caplog) is None
    model.set_param("p1", 2.0)
    assert model.get_param("p2") == 0.0
    assert _run(model, t_end=1.0, n_points=3).species.shape[0] == 3


@pytest.mark.parametrize("idiom", [False, True], ids=["csymbol", "copasi_idiom"])
@pytest.mark.parametrize(
    ("target", "tail", "frozen"),
    [
        ("p2", "<ci>k</ci>", ["k"]),
        ("p2", "<ci>p1</ci>", ["p1"]),  # read outside the rateOf too
        ("S", "<ci>k</ci>", ["k"]),
    ],
    ids=["p2=rateOf(p1)+k", "p2=rateOf(p1)+p1", "S=rateOf(p1)+k"],
)
def test_rateof_freezes_what_else_the_initial_value_reads(target, tail, frozen, idiom):
    model = bngsim.Model.from_sbml_string(
        _sbml(_rateof_p1_model(target, "false", tail, idiom=idiom))
    )
    assert model.frozen_params == frozen


def test_the_warning_names_what_a_folded_rate_is_made_of(caplog):
    """``p2 = rateOf(B)`` under ``J0: A -> B; k1*A`` is 2. ``k1`` and B's
    compartment ``d`` were refused already, with no warning at load; the
    warning names them now, the same list ``set_param`` refuses."""
    with caplog.at_level("WARNING", logger="bngsim"):
        model = bngsim.Model.from_antimony_string(_RATE_OF_B + "J0: A -> B; k1*A\n")
    assert model.get_param("p2") == pytest.approx(2.0)
    assert _frozen_warning(caplog) == "d, k1"
    assert sorted(model.frozen_params) == ["d", "k1"]
    with pytest.raises(bngsim.ParameterError, match="initial value of 'p2'"):
        model.set_param("k1", 2.0)


# ─── rateOf of a parameter nothing changes, outside an initial value ──────────
#
# #1045 gave ``p2 = rateOf(p1)`` its 0 in an initialAssignment, with p1 a
# parameter no rule or event changes. Read anywhere else (a rule, a kinetic law,
# an event) the emitter wrote a ``rate_of__p1`` accessor, which the builder binds
# only for a state, so the load failed in ExprTk (ERR239). The rate is 0 at
# every time, and so is that of a compartment nothing resizes.


def _rateof_constant_model(where: str, p1: str) -> str:
    """``rateOf(p1)`` read at ``where`` (the kinetic law also reads the rate of
    the compartment c and of its own speciesReference id). ``p1`` is
    "constant", "variable" (constant="false", nothing changes it), "ia" (set
    only by an initialAssignment) or "event" (an event sets it to 4 at t = 0.5:
    a jump, so its rate is still 0)."""

    def rate(x: str) -> str:
        return f"<apply>{_RATEOF_CSYMBOL}<ci>{x}</ci></apply>"

    m = '<math xmlns="http://www.w3.org/1998/Math/MathML">'
    rules = reactions = events = ""
    if where == "assignmentRule":
        rules = f'<assignmentRule variable="p2">{m}{rate("p1")}</math></assignmentRule>'
    elif where == "rateRule":
        rules = (
            f'<rateRule variable="x">{m}<apply><plus/>{rate("p1")}<cn>1</cn></apply></math>'
            "</rateRule>"
        )
    elif where == "reaction":
        # p1, the compartment and the reaction's own stoichiometry `sr`.
        law = f"<apply><plus/>{rate('p1')}{rate('c')}{rate('sr')}<cn>1</cn></apply>"
        reactions = (
            '<listOfReactions><reaction id="J1" reversible="false"><listOfProducts>'
            '<speciesReference id="sr" species="S" stoichiometry="1" constant="true"/>'
            f"</listOfProducts><kineticLaw>{m}{law}</math></kineticLaw>"
            "</reaction></listOfReactions>"
        )
    elif where == "event":
        events = f"""
    <listOfEvents>
      <event id="e" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">{m}
          <apply><gt/><apply><plus/>{rate("p1")}{_TIME_CSYMBOL}</apply><cn>1</cn></apply>
        </math></trigger>
        <listOfEventAssignments>
          <eventAssignment variable="p3">{m}<apply><plus/>{rate("p1")}<cn>5</cn></apply>
          </math></eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>"""
    else:
        raise ValueError(where)
    if p1 == "event":
        events = (events or "<listOfEvents></listOfEvents>").replace(
            "</listOfEvents>",
            f"""<event id="e1" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">{m}
          <apply><gt/>{_TIME_CSYMBOL}<cn>0.5</cn></apply></math></trigger>
        <listOfEventAssignments>
          <eventAssignment variable="p1">{m}<cn>4</cn></math></eventAssignment>
        </listOfEventAssignments>
      </event></listOfEvents>""",
        )
    ia = (
        '<listOfInitialAssignments><initialAssignment symbol="p1">'
        f"{m}<apply><times/><cn>2</cn><ci>k</ci></apply></math>"
        "</initialAssignment></listOfInitialAssignments>"
        if p1 == "ia"
        else ""
    )
    return f"""
  <model id="m">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" value="1" constant="{"true" if p1 == "constant" else "false"}"/>
      <parameter id="k" value="3" constant="true"/>
      <parameter id="p2" value="7" constant="false"/>
      <parameter id="p3" value="10" constant="false"/>
      <parameter id="x" value="2" constant="false"/>
    </listOfParameters>
    {ia}
    <listOfRules>{rules}</listOfRules>
    {reactions}
    {events}
  </model>"""


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("p1", ["constant", "variable", "ia", "event"])
@pytest.mark.parametrize("where", ["assignmentRule", "rateRule", "reaction", "event"])
def test_rateof_of_something_nothing_changes_is_zero_wherever_it_is_read(where, p1, codegen):
    model = bngsim.Model.from_sbml_string(_sbml(_rateof_constant_model(where, p1)))
    res = _run(model, codegen=codegen, t_end=2.0, n_points=5)
    t = np.asarray(res.time)
    if where == "assignmentRule":
        np.testing.assert_allclose(_read_value(res, "p2"), 0.0, atol=0)
    elif where == "rateRule":
        np.testing.assert_allclose(_read_value(res, "x"), 2.0 + t, rtol=1e-8)
    elif where == "reaction":
        np.testing.assert_allclose(_series(res, "S"), t, rtol=1e-8, atol=1e-10)
    else:
        # rateOf(p1) + time > 1 first holds at t = 1, and p3 := 0 + 5 there.
        p3 = _read_value(res, "p3")
        np.testing.assert_allclose(p3, np.where(t > 1.0 + 1e-9, 5.0, 10.0))


# ─── rateOf the model has no value for, beyond rule 10224 ────────────────────
#
# Each of these loaded with a wrong rate, mostly without a word: a concentration
# in a compartment an assignmentRule sizes (rule 10225), rateOf of anything but a
# symbol (10223), and a call to a functionDefinition whose body reads rateOf.
# None occurs in the semantic suite, the rr_parity BioModels or the events set;
# all are now refused by name, in the same up-front check as #1046.

_SILENT_FUNCS = {
    # f(x) := rateOf(x), the csymbol (libsbml rejects it in a lambda, 20304, and
    # then reports nothing else: only the cases that need f declare it).
    "f": f"<lambda><bvar><ci>x</ci></bvar><apply>{_RATEOF_CSYMBOL}<ci>x</ci></apply></lambda>",
    # h(y) := rateOf(y), a call to COPASI's idiom (valid SBML).
    "h": "<lambda><bvar><ci>y</ci></bvar><apply><ci>rateOf</ci><ci>y</ci></apply></lambda>",
    # g(z) := 2*f(z), rateOf through another function.
    "g": "<lambda><bvar><ci>z</ci></bvar><apply><times/><cn>2</cn><apply><ci>f</ci><ci>z</ci>"
    "</apply></apply></lambda>",
}


def _silent_rateof_model(read: str, where: str, funcs: tuple[str, ...] = ()) -> str:
    """S -> ∅ at 0.5*S (S0 = 10, so rateOf(S) is -5). In compartment
    cv := 1 + time: Sc, a concentration; Sa, an amount; Sr, a concentration
    with rate rule 3; Sk, a constant concentration. ``read`` (MathML) is read
    at ``where``, an initialAssignment or an assignment rule of p2. ``funcs``
    names the functionDefinitions to declare besides COPASI's rateOf idiom."""
    m = '<math xmlns="http://www.w3.org/1998/Math/MathML">'
    defs = "".join(
        f'<functionDefinition id="{fid}">{m}{_SILENT_FUNCS[fid]}</math></functionDefinition>'
        for fid in funcs
    )
    defs += f'<functionDefinition id="rateOf">{m}<lambda><bvar><ci>a</ci></bvar><notanumber/>'
    defs += "</lambda></math></functionDefinition>"
    ia = rule = ""
    if where == "initialAssignment:p2":
        ia = (
            f'<listOfInitialAssignments><initialAssignment symbol="p2">{m}{read}</math>'
            "</initialAssignment></listOfInitialAssignments>"
        )
    else:
        rule = f'<assignmentRule variable="p2">{m}{read}</math></assignmentRule>'
    return f"""
  <model id="m">
    <listOfFunctionDefinitions>{defs}</listOfFunctionDefinitions>
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
      <compartment id="cv" size="1" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="Sc" compartment="cv" initialConcentration="5" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="Sa" compartment="cv" initialAmount="5" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="Sr" compartment="cv" initialConcentration="5" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
      <species id="Sk" compartment="cv" initialConcentration="5" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p2" value="7" constant="false"/>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    {ia}
    <listOfRules>
      <assignmentRule variable="cv">{m}<apply><plus/><cn>1</cn>{_TIME_CSYMBOL}</apply></math>
      </assignmentRule>
      <rateRule variable="Sr">{m}<cn>3</cn></math></rateRule>
      {rule}
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>{m}<apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""


_RATE_2S = "<apply><times/><cn>2</cn><ci>S</ci></apply>"
_FUNC_WHY = "reads rateOf of a parameter of its own"


@pytest.mark.parametrize("where", ["initialAssignment:p2", "assignmentRule:p2"])
@pytest.mark.parametrize(
    ("read", "funcs", "shown", "why"),
    [
        (
            "<apply><ci>f</ci><ci>S</ci></apply>",
            ("f",),
            "f(S)",
            f"functionDefinition f {_FUNC_WHY}",
        ),
        (
            "<apply><ci>h</ci><ci>S</ci></apply>",
            ("h",),
            "h(S)",
            f"functionDefinition h {_FUNC_WHY}",
        ),
        (
            "<apply><ci>g</ci><ci>S</ci></apply>",
            ("f", "g"),
            "g(S)",
            f"functionDefinition g {_FUNC_WHY} (or calls one that does)",
        ),
        (
            f"<apply>{_RATEOF_CSYMBOL}{_RATE_2S}</apply>",
            (),
            "rateOf(2 * S)",
            "not a single symbol (validation rule 10223)",
        ),
        (
            f"<apply><ci>rateOf</ci>{_RATE_2S}</apply>",
            (),
            "rateOf(2 * S)",
            "not a single symbol (validation rule 10223)",
        ),
        (
            f"<apply>{_RATEOF_CSYMBOL}<ci>Sc</ci></apply>",
            (),
            "rateOf(Sc)",
            "Sc is a concentration in compartment cv, which the assignmentRule "
            "cv := 1 + time sizes (validation rule 10225)",
        ),
        # Constant, so SBML holds its amount: its concentration still moves with
        # the size (suite 01117). The fold read 0, a rule -5/(1+t)^2.
        (
            f"<apply>{_RATEOF_CSYMBOL}<ci>Sk</ci></apply>",
            (),
            "rateOf(Sk)",
            "Sk is a concentration in compartment cv, which the assignmentRule "
            "cv := 1 + time sizes (validation rule 10225)",
        ),
    ],
    ids=[
        "csymbol-in-function",
        "idiom-in-function",
        "through-a-function",
        "csymbol-of-2S",
        "idiom-of-2S",
        "10225",
        "10225-constant-species",
    ],
)
def test_a_rateof_with_no_value_is_refused_by_name(read, funcs, shown, why, where):
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(_sbml(_silent_rateof_model(read, where, funcs)))
    msg = str(exc.value)
    assert f"{shown} [{where}]: " in msg
    assert why in msg
    assert "rate_of__" not in msg


@pytest.mark.parametrize(
    ("read", "funcs", "value"),
    [
        # rateOf(S) read directly, beside a function that reads rateOf and that
        # nothing calls: it is left alone.
        (f"<apply>{_RATEOF_CSYMBOL}<ci>S</ci></apply>", ("h",), -5.0),
        # An amount in the rule-sized compartment has a rate of its own: no
        # reaction changes Sa, so 0 (the size's change is #736's business).
        (f"<apply>{_RATEOF_CSYMBOL}<ci>Sa</ci></apply>", (), 0.0),
        # A concentration there whose rate its own rate rule states, whatever
        # the compartment does.
        (f"<apply>{_RATEOF_CSYMBOL}<ci>Sr</ci></apply>", (), 3.0),
    ],
    ids=[
        "uncalled-function",
        "amount-in-a-sized-compartment",
        "rate-ruled-concentration",
    ],
)
def test_a_rateof_with_a_value_still_loads(read, funcs, value):
    model = bngsim.Model.from_sbml_string(
        _sbml(_silent_rateof_model(read, "initialAssignment:p2", funcs))
    )
    assert model.get_param("p2") == pytest.approx(value)


# ─── GH #231 sub-cluster 2: rateOf in an event trigger (atol-robust firing) ────
#
# A trigger gated on rateOf fires when the CVODE root finder brackets the crossing.
# The rising-edge CONFIRMATION in the main loop must refresh the live derivative
# buffer first, else it reads a stale value and the event fires only erratically —
# seen at some solver tolerances, missed at others (01261 / 01293).


def test_rateof_event_trigger_fires_robustly_across_atol():
    """01261 pattern: p1 rate rule 0.01*p1, event ``rateOf(p1) > 0.0105`` sets
    ``p2 := 5``. rateOf(p1) = 0.01*p1 crosses at p1 = 1.05 ⇒ t = ln(1.05)/0.01. The
    fire is deterministic across solver tolerances — before the confirmation refresh
    it was seen only at some atols."""
    body = f"""
  <model id="m">
    <listOfParameters>
      <parameter id="p1" value="1" constant="false"/>
      <parameter id="p2" value="10" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><cn>0.01</cn><ci>p1</ci></apply></math></rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="ev" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><apply>{_RATEOF_CSYMBOL}<ci>p1</ci></apply><cn>0.0105</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="p2"><math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn>5</cn></math></eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>"""
    t_cross = math.log(1.05) / 0.01
    for atol in (1e-8, 1e-10, 1e-12, 1e-14):
        model = bngsim.Model.from_sbml_string(_sbml(body))
        res = bngsim.Simulator(model, method="ode").run(
            t_span=(0.0, 10.0), n_points=1001, rtol=1e-8, atol=atol
        )
        p2 = _read_value(res, "p2")
        assert p2 is not None, f"p2 column missing at atol={atol}"
        t = np.asarray(res.time)
        assert p2[0] == pytest.approx(10.0)
        assert p2[-1] == pytest.approx(5.0), f"event never fired at atol={atol}"
        fire_t = t[int(np.argmax(p2 < 7.5))]
        assert fire_t == pytest.approx(t_cross, abs=3 * (t[1] - t[0])), f"atol={atol}"
