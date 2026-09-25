"""Difference-quotient sensitivities through a chain of derived parameters.

Every RHS call of a sensitivity run syncs the model's parameters from CVODES'
parameter array and re-derives the derived ones. Most calls land on the same
nominal point, so after the first the sync copies a snapshot of it; a probe
copies it too and re-derives only the derived parameters that read the probed
one, directly or through another derived parameter. A dependency the probe
path missed would leave a derived rate constant at its nominal value while its
primary moved: a sensitivity column that is silently wrong, not an error.

Here A -> B has the function rate constant ``d2 * abs(Atot)`` (a .net function
rate multiplies the reactant, so the flux is ``d2 * A**2``), with ``d2 = 3*d1``
and ``d1 = 2*k``: ``abs()`` puts the run on CVODES' difference quotient, and
``d2`` reads ``k`` only through ``d1``. ``e = kb`` is a derived parameter that
does NOT read ``k``, and must not move when ``k`` is probed. So
A = A0 / (1 + 6 k A0 t), dA/dk = -6 A0**2 t / (1 + 6 k A0 t)**2, and A does not
depend on kb at all.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import bngsim
import numpy as np

_NET = """
begin parameters
    1 k   0.5
    2 kb  0.3
    3 A0  10
    4 d1  2*k
    5 d2  3*d1
    6 e   kb
end parameters
begin functions
    1 rf() d2*abs(Atot)
end functions
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 rf
    2 2 0 e
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""


def test_a_probe_reaches_a_derived_parameter_two_links_down(tmp_path: Path):
    net = tmp_path / "chain.net"
    net.write_text(textwrap.dedent(_NET).strip() + "\n")
    m = bngsim.Model.from_net(str(net))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # abs() declines the analytic sensitivity RHS, on purpose
        sim = bngsim.Simulator(m, sensitivity_params=["k", "kb"])
        for _ in range(2):  # a second run starts from what the first left behind
            m.reset()
            r = sim.run((0.0, 1.0), 6, rtol=1e-9, atol=1e-12)
    t = np.asarray(r.time)
    s = np.asarray(r.sensitivities)[:, 0, :]  # species A
    exact = -6.0 * 10.0**2 * t / (1.0 + 30.0 * t) ** 2
    np.testing.assert_allclose(s[:, 0], exact, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(s[:, 1], 0.0, atol=1e-8)
    assert m.get_param("d2") == 3.0 and m.get_param("e") == 0.3
