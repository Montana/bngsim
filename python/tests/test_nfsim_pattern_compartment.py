"""GH #1067: NFsim refuses a compartment model outside the envelope it runs right.

BioNetGen's ``@EC:A()`` prefix puts a compartment on the pattern, written as
``compartment="EC"`` on an observable's ``<Pattern>`` or a rule's
``<ReactantPattern>`` / ``<ProductPattern>``. NFsim reads a compartment only from
a seed species and from a molecule (``A()@EC``), so it ignored the prefix and the
pattern matched every compartment (RuleWorld/nfsim#106), with no error:

* ``Molecules AECp @EC:A()`` counted all 300 A where the network counts the 200
  in EC;
* ``@EC:A() -> 0`` degraded the A in CP too: Atot 2 against 101 at t = 10;
* ``0 -> @EC:A2()`` in a volume-2 compartment ran at k1, not k1*V: A2 9.6
  against 20 at t = 10.

NFsim also applies no volume factor to a rule with two or more reactants
(``A(b)@EC + B(a)@EC`` in a size-2 compartment: AB 133.6 against 100,
RuleWorld/nfsim#108), and gives no compartment to a molecule a rule creates with
none written (``A -> A + C()``: ``C@EC()`` 0 against 86.5, RuleWorld/nfsim#109).

Until fixes are vendored, NfsimSession and Simulator(method='nf') refuse such a
model by name. Inside the envelope (no pattern compartments; created molecules
placed; and, unless every volume factor is 1, only zero- and first-order rules,
with zero-order products in one compartment) NFsim agrees with the network,
#854's zero-order synthesis into a size-2 compartment included.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import nfsim_compartment_problems

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False),
    reason="NFsim not compiled in",
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"

REFUSED = [
    ("pattern_compartment_observable.xml", r"observable 'AECp' carries .*\(@EC:\)"),
    ("pattern_compartment_rule.xml", r"rule '_R1' carries .*\(@EC:\)"),
    ("pattern_compartment_synthesis.xml", r"rule '_R2' carries .*\(@EC:\)"),
    ("compartment_bimolecular_size2.xml", r"rule '_R1' has 2 reactants while EC has size 2"),
    ("compartment_created_molecule.xml", r"rule '_R4' creates C with no compartment written"),
]


@pytest.mark.parametrize("fixture, says", REFUSED)
def test_the_session_refuses_a_pattern_compartment(fixture, says):
    with pytest.raises(ModelError, match=rf"{says}.*RuleWorld/nfsim#106.*#1067"):
        bngsim.NfsimSession(str(DATA / fixture))


@pytest.mark.parametrize("fixture, says", REFUSED)
def test_the_simulator_refuses_a_pattern_compartment(fixture, says, simple_decay_net):
    m = bngsim.Model.from_net(str(simple_decay_net))
    with pytest.raises(ModelError, match=says):
        bngsim.Simulator(m, method="nf", xml_path=str(DATA / fixture))


@pytest.mark.parametrize(
    "fixture",
    [
        # Compartments on molecules and seed species only, all of size 1.
        "molecule_compartment_degradation.xml",
        # #854: zero-order synthesis into one compartment of size 2 is scaled right.
        "zero_order_switchable_rate_compartment.xml",
    ],
)
def test_a_compartment_model_inside_the_envelope_runs(fixture):
    xml = DATA / fixture
    assert nfsim_compartment_problems(xml) == []
    bngsim.NfsimSession(str(xml)).destroy()


def test_the_molecule_level_form_agrees_with_the_network():
    # A()@EC -> 0 degrades only the 200 A in EC; the 100 in CP stay.
    xml = DATA / "molecule_compartment_degradation.xml"
    net = DATA / "molecule_compartment_degradation.net"
    n = 20
    nf = []
    for seed in range(1, n + 1):
        with bngsim.NfsimSession(str(xml)) as s:
            s.initialize(seed=seed)
            nf.append(np.asarray(s.simulate(0, 4, n_points=3).observables)[-1])
    ode = np.asarray(
        bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode")
        .run(t_span=(0, 4), n_points=3)
        .observables
    )[-1]
    a = np.array(nf)
    # Atot and AEC: binomial survivors of 200 plus the 100 in CP, so the spread
    # is that of the survivors; five standard errors.
    sem = np.sqrt(np.maximum(a.var(axis=0, ddof=1), 1.0) / n)
    np.testing.assert_array_less(np.abs(a.mean(axis=0) - ode), 5 * sem)
    assert ode[0] > 100  # the CP molecules were not degraded
