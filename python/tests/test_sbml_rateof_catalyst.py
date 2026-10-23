"""A conversion factor over ``rateOf`` of a species a reaction lists but does
not change loads (#1138's review).

``S3 -> S3`` at ``0.1·S3`` lists S3 as reactant and product with a net
stoichiometry of 0, so nothing moves it and its rate is 0. The moving set took
any listed species as changed by the reaction, and ``cf := 1 + rateOf(S3)``
beside it was refused as a factor that changes during the run. libRoadRunner
2.9.2 runs it with cf = 1: S1(2) = e^(-1/2) = 0.606531 with the factor on S1.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

libsbml = pytest.importorskip("libsbml")
import bngsim  # noqa: E402

E_HALF = math.exp(-0.5)


def _model(tmp_path, *, cf_level, s3_reaction, s3_stoich_constant=True):
    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    c = m.createCompartment()
    c.setId("C")
    c.setSize(1)
    c.setConstant(True)
    for sid, ic in (("S1", 1.0), ("S3", 2.0)):
        s = m.createSpecies()
        s.setId(sid)
        s.setCompartment("C")
        s.setInitialConcentration(ic)
        s.setHasOnlySubstanceUnits(False)
        s.setBoundaryCondition(False)
        s.setConstant(False)
        if cf_level == "species" and sid == "S1":
            s.setConversionFactor("cf")
    p = m.createParameter()
    p.setId("cf")
    p.setConstant(False)
    p.setValue(1.0)
    if cf_level == "model":
        m.setConversionFactor("cf")
    r = m.createReaction()
    r.setId("J0")
    r.setReversible(False)
    rr = r.createReactant()
    rr.setSpecies("S1")
    rr.setStoichiometry(1)
    rr.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula("0.25*S1"))
    if s3_reaction == "catalyst":
        r = m.createReaction()
        r.setId("J3")
        r.setReversible(False)
        for make in (r.createReactant, r.createProduct):
            sr = make()
            sr.setSpecies("S3")
            sr.setStoichiometry(1)
            sr.setConstant(s3_stoich_constant)
        r.createKineticLaw().setMath(libsbml.parseL3Formula("0.1*S3"))
    elif s3_reaction == "made":
        r = m.createReaction()
        r.setId("J3")
        r.setReversible(False)
        pr = r.createProduct()
        pr.setSpecies("S3")
        pr.setStoichiometry(1)
        pr.setConstant(True)
        r.createKineticLaw().setMath(libsbml.parseL3Formula("0.1"))
    ar = m.createAssignmentRule()
    ar.setVariable("cf")
    ar.setMath(libsbml.parseL3Formula("1 + rateOf(S3)"))
    path = tmp_path / f"{cf_level}_{s3_reaction}.xml"
    path.write_text(libsbml.writeSBMLToString(doc))
    return bngsim.Model.from_sbml(str(path))


def _s1_at_2(model):
    run = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 2.0), n_points=3, rtol=1e-10, atol=1e-12
    )
    return float(np.asarray(run.species)[-1, list(run.species_names).index("S1")])


@pytest.mark.parametrize("cf_level", ["species", "model"])
def test_a_catalyst_only_species_does_not_move_the_factor(tmp_path, cf_level):
    """libRoadRunner: S1(2) = 0.606531, S3 = 2, cf = 1 (species level; it
    refuses the model level as recursive, which bngsim loads since #1138)."""
    model = _model(tmp_path, cf_level=cf_level, s3_reaction="catalyst")
    assert _s1_at_2(model) == pytest.approx(E_HALF, rel=1e-8)


@pytest.mark.parametrize("cf_level", ["species", "model"])
def test_a_species_a_reaction_makes_still_moves_the_factor(tmp_path, cf_level):
    """Control: S3 made at 0.1 moves, and the factor over its rate is refused
    (libRoadRunner moves cf to 1.1 and gives S1(2) = 0.576950)."""
    with pytest.raises(bngsim.ModelError, match=r"changes during the run"):
        _model(tmp_path, cf_level=cf_level, s3_reaction="made")


def test_a_stoichiometry_that_is_not_constant_keeps_the_species_moving(tmp_path):
    """A speciesReference with constant="false" may be set by a rule, so its
    net change is not known at load: refused as before."""
    with pytest.raises(bngsim.ModelError, match=r"changes during the run"):
        _model(tmp_path, cf_level="species", s3_reaction="catalyst", s3_stoich_constant=False)
