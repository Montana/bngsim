"""The ``.net`` writer spells the natural logarithm ``ln`` (issue #1273).

bngsim's expressions write it ``log``, and the writer copied that out. BioNetGen's
``run_network`` has no ``log``: it refuses the file, ``Could not find variable
'log'``. Ten of the 214 SBML event models in ``benchmarks/sbml_events`` were
written that way. ``ln`` is the name BNGL and ``run_network`` give it, and
bngsim's reader takes it as the same function: the trajectory, the analytic
Jacobian, the sensitivities and the compiled code are what ``log`` gives.

``log10`` and ``log2`` are names ``run_network`` has, and are left alone, as is
any name that only holds ``log``: ``log_k``, or a function ``catalog()``.
"""

from __future__ import annotations

import re
import subprocess

import bngsim
import numpy as np
import pytest
from bngsim._bngpath import resolve_bng
from bngsim.convert import write_net

_NET = """begin parameters
    1 k     0.5
    2 c     2
    3 log_k 0.25
    4 k2    k*log(c+1)
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin groups
    1 Aobs 1
end groups
begin functions
    1 f() k*log(Aobs+c)+log_k*log10(Aobs+1)
    2 g() k2*log2(Aobs+1)/log(c)+catalog()
    3 catalog() 0.1*Aobs
end functions
begin reactions
    1 1 2 f
    2 2 0 g
end reactions
"""

# A derived rate constant, as BNG2.pl writes one: no function, no flux helper.
_RATE = """begin parameters
    1 k  0.5
    2 c  2
    3 k2 k*log(c+1)
end parameters
begin species
    1 A() 10
end species
begin reactions
    1 1 0 k2
end reactions
"""

TIMES = list(np.linspace(0.0, 5.0, 21))


def _load(tmp_path, text, name="m.net"):
    path = tmp_path / name
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _expressions(net):
    """The expression field of each parameters and functions line."""
    block = None
    for line in net.splitlines():
        s = line.strip()
        if s.startswith("begin "):
            block = s.split()[1]
        elif s.startswith("end "):
            block = None
        elif s and not s.startswith("#") and block in ("parameters", "functions"):
            yield s.split(None, 2)[2]


def _run(model, params=()):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params) or None)
    return sim, sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


def test_the_natural_log_is_written_ln(tmp_path):
    net = write_net(_load(tmp_path, _NET))
    fields = " ".join(_expressions(net))
    assert not re.search(r"(?<![\w.])log\s*\(", fields), net
    assert fields.count("ln(") == 3
    # names run_network has, and a name that only holds "log", are left alone
    assert "log10(" in fields and "log2(" in fields and "log_k" in fields
    # a call to a function is written by its name alone, so only a builtin is
    # followed by "(" here, and the left boundary is not reached; the name survives
    assert "+catalog " in fields + " "


def test_an_expression_written_as_it_is_is_spelled_ln_too(tmp_path):
    """An expression the writer cannot give as one field (issue #1249) is
    written as it is under strict=False, and its logarithm is still ``ln``."""
    text = _NET.replace("k*log(Aobs+c)+", "if(Aobs > 1 xor k > 1, log(Aobs), 0)+")
    with pytest.warns(bngsim.ConversionWarning, match="xor"):
        net = write_net(_load(tmp_path, text), strict=False)
    assert "if(Aobs > 1 xor k > 1, ln(Aobs), 0)+" in net
    assert not re.search(r"(?<![\w.])log\s*\(", " ".join(_expressions(net))), net


def test_the_written_file_is_the_model(tmp_path):
    """Read back, the same trajectory and sensitivities, the analytic Jacobian
    and sensitivity right-hand side kept."""
    source = _load(tmp_path, _NET)
    again = _load(tmp_path, write_net(source), "again.net")
    again.prepare_analytical_jacobian()
    assert again.analytical_jacobian_status == "complete"
    sim, back = _run(again, ["k", "c"])
    assert sim.sens_rhs_decline_reason is None
    _, ref = _run(_load(tmp_path, _NET, "ref.net"), ["k", "c"])
    np.testing.assert_allclose(back.species, ref.species, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(back.sensitivities, ref.sensitivities, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("codegen", [True, False])
def test_ln_is_log_on_every_path(tmp_path, codegen):
    """The reader's ln is its log, bit for bit, compiled or interpreted."""
    a = _load(tmp_path, _NET, "log.net")
    b = _load(tmp_path, _NET.replace("log(", "ln("), "ln.net")
    runs = [
        bngsim.Simulator(m, method="ode", codegen=codegen).run(
            sample_times=TIMES, rtol=1e-10, atol=1e-12
        )
        for m in (a, b)
    ]
    np.testing.assert_array_equal(runs[0].species, runs[1].species)


def _run_network():
    res = resolve_bng()
    if not res.ok or res.root is None:
        return None
    path = res.root / "bin" / "run_network"
    return path if path.is_file() else None


@pytest.mark.skipif(
    _run_network() is None, reason="no BioNetGen install (BNG2.pl and its run_network) found"
)
def test_run_network_runs_the_written_file(tmp_path):
    """The file the writer gives, run by run_network: bngsim's trajectory. Before,
    'Could not find variable 'log''. The log is in a derived rate constant, a
    line the writer gives as one field."""
    path = tmp_path / "written.net"
    write_net(_load(tmp_path, _RATE), path)
    out = tmp_path / "rn"
    run = subprocess.run(
        [str(_run_network()), "-o", str(out), "-p", "cvode", "-a", "1e-12", "-r", "1e-10"]
        + [str(path), "0.25", "20"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout[-600:] + run.stderr[-600:]
    rows = np.loadtxt(str(out) + ".cdat", comments="#")
    expected = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
        sample_times=list(rows[:, 0]), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(rows[:, 1:], expected.species, rtol=1e-6, atol=1e-9)
    # and the rate is the log's: A = 10 exp(-k ln(c + 1) t)
    np.testing.assert_allclose(rows[:, 1], 10 * np.exp(-0.5 * np.log(3) * rows[:, 0]), rtol=1e-6)
