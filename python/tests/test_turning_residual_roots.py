"""Issue #904, part 2: a window whose residual turns within a step.

``if(abs(Aobs-5) < 0.01, k, 0)`` and ``if((Aobs-5)^2 < 0.0001, k, 0)`` are the
window ``4.99 < A < 5.01``. Each is one comparison, and its residual
(``abs(A-5) - 0.01``, ``(A-5)^2 - 1e-4``) falls to zero and comes back up as A
passes 5: within one step it has the same sign at both ends, so a root on it
reports nothing, and the window was stepped over. Y came back 0 for 0.1 at
every tolerance, and with sensitivities ``(A-5)^2`` gave Y = 0 and dY/dk = 0.

Such an atom is now rooted at its edges, ``A - 5`` against ``0.01`` and
``-0.01``, each monotone along a flow that moves A one way, and on the
sensitivity path jumped there as any crossing is.

A = 2t, so A is in the window for 0.01 of time and Y = k * 0.01. With the rate
``k*A`` and the window centred on a parameter c, Y = k * 0.01 * c.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import SensitivityUnsupportedError
from bngsim._switch_sensitivity import (
    _monotone_pieces,
    state_switch_conditions,
    state_switch_reactions,
)

NET = """begin parameters
    1 k 10
    2 two 2
    3 c 5
end parameters
begin functions
    1 f() {cond}
end functions
begin species
    1 A() 0
    2 Y() 0
end species
begin reactions
    1 0 1 two
    2 0 2 f
end reactions
begin groups
    1 Aobs 1
end groups
"""

WINDOWS = [
    "if((Aobs>4.99)&&(Aobs<5.01),k,0)",
    "if(abs(Aobs-5)<0.01,k,0)",
    "if((Aobs-5)^2<0.0001,k,0)",
    "k*(abs(Aobs-5)<0.01)",
]
IDS = ["two-comparisons", "abs", "square", "abs-as-a-number"]


def _model(tmp_path, cond: str) -> bngsim.Model:
    path = tmp_path / "window.net"
    path.write_text(NET.format(cond=cond))
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
@pytest.mark.parametrize("cond", WINDOWS, ids=IDS)
def test_a_window_one_comparison_spells_is_not_stepped_over(tmp_path, cond, rtol):
    run = bngsim.Simulator(_model(tmp_path, cond), method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=rtol, atol=1e-12
    )
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(0.1, rel=1e-5)


@pytest.mark.parametrize("cond", WINDOWS[1:], ids=IDS[1:])
def test_its_edges_are_the_roots(tmp_path, cond):
    roots = _model(tmp_path, cond).state_switch_root_conditions()
    assert len(roots) == 2
    assert all("Aobs-5" in r and "abs" not in r and "^" not in r for r in roots)


@pytest.mark.parametrize(
    "cond",
    ["if((Aobs>c-0.01)&&(Aobs<c+0.01),k*Aobs,0)", "if((Aobs-c)^2<0.0001,k*Aobs,0)"],
    ids=["two-comparisons", "square"],
)
def test_the_sensitivity_jumps_at_each_edge(tmp_path, cond):
    """Y = k * 0.01 * c: dY/dc = 0.1 moves both edges, dY/dk = 0.05."""
    sim = bngsim.Simulator(_model(tmp_path, cond), method="ode", sensitivity_params=["c", "k"])
    run = sim.run(t_span=(0.0, 10.0), n_points=2, rtol=1e-9, atol=1e-12)
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(0.5, rel=1e-6)
    d_c, d_k = np.asarray(run.sensitivities)[-1, 1, :]
    assert d_c == pytest.approx(0.1, rel=1e-5)
    assert d_k == pytest.approx(0.05, rel=1e-6)


@pytest.mark.parametrize("cond", WINDOWS[1:3], ids=IDS[1:3])
def test_each_edge_is_read_by_the_reaction_that_reads_the_window(tmp_path, cond):
    """The jump at an edge is judged on the flux of the reactions that read it
    (issue #763); an edge no reaction is listed for is taken as continuous."""
    core = _model(tmp_path, cond)._core
    conditions = state_switch_conditions(core)
    assert len(conditions) == 2
    assert state_switch_reactions(core, conditions) == [[1], [1]]


@pytest.mark.parametrize(
    ("atom", "pieces"),
    [
        ("abs(Aobs-5)<0.01", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-(0.01))"]),
        ("0.01>abs(Aobs-5)", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-(0.01))"]),
        ("abs(Aobs-c)<=w", ["(Aobs-c)<(w)", "(Aobs-c)>(-(w))"]),
        ("(Aobs-5)^2<0.0001", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-0.01)"]),
        ("Aobs^2>4", ["(Aobs)<(2.0)", "(Aobs)>(-2.0)"]),
    ],
)
def test_the_pieces_of_a_turning_comparison(atom, pieces):
    assert _monotone_pieces(atom) == pieces


@pytest.mark.parametrize(
    "atom",
    [
        "Aobs<5",  # monotone already
        "abs(A)*abs(B)<1",  # not one abs()
        "abs(A)<abs(B)",  # turns on both sides
        "(A-5)^2<w",  # a square against a bound that is not a number
        "(A-5)^2<-1",  # against a negative one: no edge at all
        "(A-5)^2<0",  # against 0: one point, no window
        "abs(A-5)<-1",  # no window either
        "abs(A-5)>0",  # false at one point only
        "A^2.5<1",  # not a square
        "(A-5)^2+(B-5)^2<1",  # a sum: only its last group is squared
        "(A-5)*(B)^2<4",  # a product: likewise
    ],
)
def test_other_comparisons_are_left_as_they_are(atom):
    assert _monotone_pieces(atom) == []


DISK = """begin parameters
    1 k 10
    2 five 5
    3 one 1
    4 c 5
end parameters
begin functions
    1 f() if((Aobs-c)^2+(Bobs-5)^2<0.75,k,0)
end functions
begin species
    1 A() 0
    2 Y() 0
    3 B() 5.25
end species
begin reactions
    1 0 1 five
    2 1 0 one
    3 0 2 f
end reactions
begin groups
    1 Aobs 1
    2 Bobs 3
end groups
"""


def test_a_sum_of_squares_keeps_its_own_root(tmp_path):
    """``(A-c)^2+(B-5)^2`` squares its last group only, so it is no square of
    one side and its comparison is rooted whole, at the crossing it has.

    A = 5(1 - e^-t) enters the disk and stays: with B = 5.25 it is inside once
    25 e^(-2t) < 0.6875, from t* = -ln(0.0275)/2, so Y = k(10 - t*) and
    dY/dc = -k/A'(t*) = -2/sqrt(0.0275). Read as the square of
    ``(A-c)^2+(B-5)``, the disk was rooted at A - c = -0.866, where nothing
    switches, its own crossing was not jumped, and dY/dc came back 0.
    """
    path = tmp_path / "disk.net"
    path.write_text(DISK)
    roots = bngsim.Model.from_net(str(path)).state_switch_root_conditions()
    assert roots == ("(Aobs-c)^2+(Bobs-5)^2<0.75",)
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["c", "k"]
    )
    run = sim.run(t_span=(0.0, 10.0), n_points=2, rtol=1e-9, atol=1e-12)
    t_star = -np.log(0.0275) / 2
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(10 * (10 - t_star), rel=1e-6)
    d_c, d_k = np.asarray(run.sensitivities)[-1, 1, :]
    assert d_c == pytest.approx(-2 / np.sqrt(0.0275), rel=1e-5)
    assert d_k == pytest.approx(10 - t_star, rel=1e-6)


def _sbml_window(tmp_path, condition: str) -> str:
    """A = 2t and Y' = piecewise(k, condition, 0), as an SBML document."""
    import libsbml

    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("window")
    comp = model.createCompartment()
    comp.setId("C")
    comp.setSize(1)
    comp.setConstant(True)
    for sid in ("A", "Y"):
        sp = model.createSpecies()
        sp.setId(sid)
        sp.setCompartment("C")
        sp.setInitialConcentration(0)
        sp.setHasOnlySubstanceUnits(False)
        sp.setBoundaryCondition(False)
        sp.setConstant(False)
    k = model.createParameter()
    k.setId("k")
    k.setValue(10)
    k.setConstant(True)
    for rid, product, law in (("R1", "A", "2"), ("R2", "Y", f"piecewise(k, {condition}, 0)")):
        rxn = model.createReaction()
        rxn.setId(rid)
        rxn.setReversible(False)
        ref = rxn.createProduct()
        ref.setSpecies(product)
        ref.setStoichiometry(1)
        ref.setConstant(True)
        rxn.createKineticLaw().setMath(libsbml.parseL3Formula(law))
    path = tmp_path / "window.xml"
    path.write_text(libsbml.writeSBMLToString(doc))
    return str(path)


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
@pytest.mark.parametrize(
    "condition", ["abs(A - 5) < 0.01", "pow(A - 5, 2) < 0.0001"], ids=["abs", "pow"]
)
def test_an_sbml_kinetic_law_spelled_so_is_not_stepped_over(tmp_path, condition, rtol):
    """The SBML loader roots the whole comparison, which turns within the step
    as the ``.net`` one does; the run roots its edges as well."""
    model = bngsim.Model.from_sbml(_sbml_window(tmp_path, condition))
    run = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=rtol, atol=1e-12
    )
    y = dict(zip(run.species_names, np.asarray(run.species)[-1], strict=True))["Y"]
    assert y == pytest.approx(0.1, rel=1e-5)


COUNTER = """begin parameters
    1 k 10
    2 one 1
    3 c 5
end parameters
begin functions
    1 f() if(abs(Aobs-c)<0.01,k*Aobs,0)
end functions
begin species
    1 A() 0
    2 Y() 0
end species
begin reactions
    1 0 1 one
    2 0 2 f
end reactions
begin groups
    1 Aobs 1
end groups
"""


def test_an_abs_window_on_a_counter_whose_edges_a_column_moves_is_refused(tmp_path):
    """A is a counter: Y = k*c*0.02 = 1, dY/dc = 0.2, dY/dk = 0.1.

    abs() has no analytic derivative, so the columns are CVODES' difference
    quotient. Issue #48 holds a counter's threshold while the quotient is
    taken, but only where it reads the condition, and it does not read this
    one. With c requested the quotient moved the edges the run now stops at,
    and dY/dc came back 7.03. A column that moves neither is right.
    """
    path = tmp_path / "counter.net"
    path.write_text(COUNTER)
    with pytest.raises(SensitivityUnsupportedError, match=r"abs\(Aobs-c\)<0\.01"):
        bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["c"]
        ).run(t_span=(0.0, 10.0), n_points=2, rtol=1e-8, atol=1e-12)
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["k"]
    )
    run = sim.run(t_span=(0.0, 10.0), n_points=2, rtol=1e-8, atol=1e-12)
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(1.0, rel=1e-6)
    assert float(np.asarray(run.sensitivities)[-1, 1, 0]) == pytest.approx(0.1, rel=1e-6)
