"""Issue #636 — refusing to pickle an extension-backed object, usefully.

``Result``, ``Model`` and ``Simulator`` each hold a handle into
``bngsim._bngsim_core``, which has no pickle support, so none of them can cross
a process boundary. That part is a design boundary rather than a defect. What
was wrong is what the caller was told::

    TypeError: cannot pickle 'bngsim._bngsim_core.ResultCore' object

— the name of an internal C++ type, to a reader who is parallelising a fit
across a process pool and needs to know that the way through is to convert
first. bngsim's own parallelism is threads, so nothing in the library ever hits
this; a consumer hits it on the first return value.

Coverage:
- Each class refuses with a message naming itself, the reason, and a remedy.
- Every remedy the messages name is checked to exist and to work, because an
  error message that recommends a method is a claim about the API.
- ``copy.copy`` is not collateral damage of the refusal. It falls back to
  ``__reduce_ex__``, so a refusal on ``__reduce__`` alone would decide the
  copy question by accident; each class answers it deliberately under
  ``__copy__`` instead. What each one answers is issue #643's call and is
  pinned in test_copy_refusals.py — here we only pin that the pickle refusal
  is not what decided it.
- ``copy.deepcopy`` gets the same clear refusal, where it used to get the
  internal type name.
"""

from __future__ import annotations

import copy
import pickle

import bngsim
import numpy as np
import pytest

DECAY_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="decay">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="death" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@pytest.fixture
def trio():
    """The three extension-backed objects a worker might try to send."""
    model = bngsim.Model.from_sbml_string(DECAY_SBML)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 1), n_points=3)
    return model, sim, result


# ── The refusal names itself, the reason, and a way through ──────────


@pytest.mark.parametrize(
    ("which", "must_mention"),
    [
        ("Result", ["as_roadrunner()", "to_xarray()", ".species", "save(path)"]),
        ("Model", ["Model.from_sbml(path)", "clone()"]),
        # model.clone() is the in-process half: copy.deepcopy does not go through
        # __copy__, so an in-process "give me my own" request lands on THIS message
        # (issue #643). It is a separate sentence, and guarded by "within this
        # process", so a reader parallelising across a pool cannot read it as a way
        # to make a Simulator picklable. Model's message has carried the same clause
        # since #636; this is the half that was missing.
        ("Simulator", ["Model.from_sbml(path)", "Result.save(path)", "model.clone()"]),
    ],
)
def test_the_refusal_says_what_to_do_instead(trio, which, must_mention):
    model, sim, result = trio
    obj = {"Result": result, "Model": model, "Simulator": sim}[which]

    with pytest.raises(TypeError) as excinfo:
        pickle.dumps(obj)

    message = str(excinfo.value)
    assert f"cannot pickle a {which}" in message
    # The reason, in terms the caller can act on rather than a C++ type name.
    assert "compiled extension" in message
    assert "process boundary" in message
    assert "_bngsim_core" not in message
    for remedy in must_mention:
        assert remedy in message, f"{which} message does not name {remedy}"


# ── An error message that names a method is a claim about the API ────


def test_every_remedy_the_messages_name_exists(trio):
    model, _sim, result = trio
    for name in ("as_roadrunner", "to_xarray", "save", "time", "species", "observables"):
        assert hasattr(result, name), f"Result.{name} is named in a message but missing"
    for name in ("from_sbml", "from_bngl", "from_net"):
        assert hasattr(bngsim.Model, name), f"Model.{name} is named in a message but missing"
    assert hasattr(model, "clone")  # named by both the Model and Simulator messages
    assert hasattr(bngsim.Result, "load")


def test_the_conversions_really_do_cross_a_process_boundary(trio):
    """Pickling is the mechanism multiprocessing and joblib move objects by."""
    _model, _sim, result = trio

    arr = pickle.loads(pickle.dumps(result.as_roadrunner()))
    assert arr.colnames == ["time", "[X]"]
    np.testing.assert_array_equal(arr["[X]"], result.as_roadrunner()["[X]"])

    for name in ("time", "species", "observables"):
        block = np.asarray(getattr(result, name))
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(block)), block)


def test_a_result_round_trips_through_a_file(trio, tmp_path):
    """The other remedy: save here, load on the other side."""
    pytest.importorskip("h5py", reason="could not import h5py")
    _model, _sim, result = trio
    path = tmp_path / "result.h5"
    result.save(path)
    back = bngsim.Result.load(path)
    np.testing.assert_allclose(np.asarray(back.species), np.asarray(result.species))


# ── What the refusal does, and does not, decide ──────────────────────


def test_the_pickle_refusal_is_not_what_answers_the_copy_question(trio):
    """copy.copy falls back to __reduce_ex__, which defers to the refusal.

    So without a __copy__ of its own, every one of the three would refuse a
    shallow copy as a side effect of a change about an error message. Each has
    one, and Result's still copies — which is how we can tell the copy question
    is being answered on its own terms (issue #643) rather than inherited here.
    """
    model, sim, result = trio
    duplicate = copy.copy(result)
    assert duplicate is not result
    assert type(duplicate) is type(result)
    np.testing.assert_array_equal(np.asarray(duplicate.species), np.asarray(result.species))

    for obj in (model, sim):
        with pytest.raises(TypeError, match="copy.copy"):
            copy.copy(obj)


def test_a_deep_copy_gets_the_same_clear_refusal(trio):
    """It failed before too — with the internal type name."""
    _model, _sim, result = trio
    with pytest.raises(TypeError, match="cannot pickle a Result"):
        copy.deepcopy(result)


def test_a_steady_state_result_is_still_picklable(trio):
    """Not everything bngsim returns is engine-backed, and a worker can send
    this one home as it is. Pinned so the refusals are not widened by reflex."""
    _model, sim, _result = trio
    ss = sim.steady_state(method="newton")
    back = pickle.loads(pickle.dumps(ss))
    assert back.converged == ss.converged
    np.testing.assert_allclose(np.asarray(back.concentrations), np.asarray(ss.concentrations))
