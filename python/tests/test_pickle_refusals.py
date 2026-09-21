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
- ``copy.copy`` keeps working. It falls back to ``__reduce_ex__``, so the
  refusal would have taken a working shallow copy with it.
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
        ("Simulator", ["Model.from_sbml(path)", "Result.save(path)"]),
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
    assert hasattr(model, "clone")
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


# ── What the refusal must not take with it ───────────────────────────


def test_a_shallow_copy_still_works(trio):
    """copy.copy falls back to __reduce_ex__, which defers to the refusal.

    Without __copy__ this would now raise, taking away an operation that works
    today. The copy shares the extension handle, which is what it always did.
    """
    model, sim, result = trio
    for obj in (result, model, sim):
        duplicate = copy.copy(obj)
        assert duplicate is not obj
        assert type(duplicate) is type(obj)
    np.testing.assert_array_equal(
        np.asarray(copy.copy(result).species), np.asarray(result.species)
    )
    assert copy.copy(model).n_species == model.n_species


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
