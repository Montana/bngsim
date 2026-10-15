"""GH #1062: NFsim refuses a local function that reads a changing observable outside its argument.

NFsim re-evaluates a local function only on the complexes a reaction produces
(RuleWorld/nfsim#104). In ``fA(x) = k*(1+AC(x))*(1+0.01*MAB)`` the ``AC(x)``
term follows the molecule, but the ``MAB`` term keeps, on every molecule off the
firing complex, the value it had at that molecule's last event. On the model in
``tests/data/nfsim/local_function_outside_observable.bngl``, AB at t = 10 was
50.1 against network SSA 55.6 (100 seeds, z = -10), with no error. Until a fix
is vendored, NfsimSession and Simulator(method='nf') refuse such a read by name.

An outside read stays allowed when its observable can never change: every
molecule type in the observable appears in no rule, and either has no
components or the model deletes nothing. The check errs toward refusing, and
the model's generated network (ode / ssa) is always available.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import stale_local_function_reads

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False),
    reason="NFsim not compiled in",
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"

REFUSED = [
    ("local_function_outside_observable.xml", "'fA' reads 'MAB'"),
    ("local_function_outside_species_observable.xml", "'fA' reads 'SpAB'"),
    # Src has a component, and B(a) -> 0 deletes: a deletion could free a bond.
    ("local_function_bondable_outside_observable.xml", "'fA' reads 'NSrc'"),
]


@pytest.mark.parametrize("fixture, says", REFUSED)
def test_the_session_refuses_a_stale_outside_read(fixture, says):
    with pytest.raises(ModelError, match=rf"{says}.*RuleWorld/nfsim#104.*#1062"):
        bngsim.NfsimSession(str(DATA / fixture))


@pytest.mark.parametrize("fixture, says", REFUSED)
def test_the_simulator_refuses_a_stale_outside_read(fixture, says, simple_decay_net):
    m = bngsim.Model.from_net(str(simple_decay_net))
    with pytest.raises(ModelError, match=says):
        bngsim.Simulator(m, method="nf", xml_path=str(DATA / fixture))


@pytest.mark.parametrize(
    "fixture",
    [
        # Src has no components and no rule names it: NSrc is constant.
        "local_function_constant_outside_observable.xml",
        # Both read Obs_Src = Src(), which no rule names, outside the argument.
        "context_symmetry.xml",
        "symmetry_factor_rate_laws.xml",
    ],
)
def test_an_outside_read_of_a_constant_observable_runs(fixture):
    xml = DATA / fixture
    assert stale_local_function_reads(xml) == []
    with bngsim.NfsimSession(str(xml)) as nf:
        nf.initialize(seed=1)
        nf.simulate(0, 1, n_points=2)


def test_a_read_through_the_argument_is_not_an_outside_read(tmp_path):
    # The same function with the MAB term dropped reads only AC(x).
    src = (DATA / "local_function_outside_observable.xml").read_text()
    old = "<Expression> (k*(1+AC(x)))*(1+(0.01*MAB)) </Expression>"
    assert src.count(old) == 1
    xml = tmp_path / "argument_only.xml"
    xml.write_text(src.replace(old, "<Expression> k*(1+AC(x)) </Expression>"))
    assert stale_local_function_reads(xml) == []
    bngsim.NfsimSession(str(xml)).destroy()


def test_a_global_function_reading_the_observable_is_not_refused(tmp_path):
    # Only local functions go stale: a global one is re-evaluated whenever its
    # observables change.
    src = (DATA / "local_function_outside_observable.xml").read_text()
    fa = src.index('<Function id="fA">')
    start = src.index("<ListOfArguments>", fa)
    end = src.index("</ListOfArguments>", fa) + len("</ListOfArguments>")
    xml = tmp_path / "global.xml"
    xml.write_text(src[:start] + src[end:])
    assert stale_local_function_reads(xml) == []
