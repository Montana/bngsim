"""Every expression the ``.net`` writer emits is one token (issue #1249).

BioNetGen's ``run_network`` reads a line as whitespace-separated fields. Since
#497 the writer keeps a derived parameter as its expression, written as sympy
prints it, ``k / (c1 / _V0_c1)``; ``run_network`` refuses the parameters block
("parameters list not read because of errors"). A function line it does not
refuse: it reads the expression up to the first space and runs that, with no
message, so ``k * Aobs / (1 + Aobs)`` ran as ``k`` (A(1) = 6.065 for 6.414).
The writer's own helpers and wrappers were spaced the same way.

Each expression is now written with its whitespace removed. Where whitespace
separates two words, removing it would join them (#498: ``a and b`` is not
``aandb``): ``and`` and ``or`` are written ``&&`` and ``||``, which both
readers take with the same meaning and precedence, and any other such operator
(``xor``) has no one-token spelling and is refused, as an unrepresentable
construct is.
"""

from __future__ import annotations

import subprocess
import warnings

import bngsim
import numpy as np
import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim._bngpath import resolve_bng
from bngsim.convert import write_net

_SPACED_NET = """begin parameters
    1 k    0.5
    2 thr  5.0
    3 k2   2 * k
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin functions
    1 f() k * Aobs / (1 + Aobs)
    2 g() if(Aobs > thr and k > 0.1, k2, 0)
    3 h() if(Aobs < 1 or Bobs > 8, 1, 0)
end functions
begin reactions
    1 1 2 f
    2 2 0 g
    3 0 1 h
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
end groups
"""


def _load(tmp_path, text, name="m.net"):
    path = tmp_path / name
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _expression_lines(net: str):
    """(block, line) for each parameters and functions line of *net*."""
    block = None
    for line in net.splitlines():
        s = line.strip()
        if s.startswith("begin "):
            block = s.split()[1]
        elif s.startswith("end "):
            block = None
        elif s and not s.startswith("#") and block in ("parameters", "functions"):
            yield block, s


def _trajectory(model, times):
    r = bngsim.Simulator(model, method="ode").run(sample_times=list(times), rtol=1e-10, atol=1e-12)
    return np.asarray(r.species)


def test_a_derived_parameter_is_one_token(tmp_path):
    """The issue's model: `_rateLaw_J0` was `k / (c1 / _V0_c1)`."""
    import antimony

    antimony.loadAntimonyString(
        "model m\n  compartment c1 = 2\n  species A in c1 = 10\n  k = 0.5\n  J0: A -> ; k*A\nend"
    )
    model = bngsim.Model.from_sbml_string(antimony.getSBMLString("m"))
    net = write_net(model)
    lines = dict(
        line.split()[1:3] for block, line in _expression_lines(net) if block == "parameters"
    )
    assert lines["_rateLaw_J0"] == "k/(c1/_V0_c1)"


def test_every_expression_written_is_one_token(tmp_path):
    """Index, name and one expression on each line, the functions the model
    reads included, and the trajectory is the model's."""
    model = _load(tmp_path, _SPACED_NET)
    net = write_net(model)
    seen = 0
    for _, line in _expression_lines(net):
        assert len(line.split()) == 3, line
        seen += 1
    # 3 parameters, 3 functions, and the flux helpers the writer adds
    assert seen >= 6
    again = _load(tmp_path, net, "again.net")
    times = np.linspace(0, 5, 21)
    np.testing.assert_allclose(_trajectory(again, times), _trajectory(model, times), rtol=1e-9)


def test_and_and_or_are_written_with_the_same_meaning(tmp_path):
    net = write_net(_load(tmp_path, _SPACED_NET))
    functions = dict(
        line.split()[1:3] for block, line in _expression_lines(net) if block == "functions"
    )
    assert functions["g()"] == "if(Aobs>thr&&k>0.1,k2,0)"
    assert functions["h()"] == "if(Aobs<1||Bobs>8,1,0)"
    assert functions["f()"] == "k*Aobs/(1+Aobs)"


def test_precedence_is_kept_where_and_and_or_meet(tmp_path):
    """`a or b and c` is `a or (b and c)` in ExprTk, and `a||b&&c` is the same
    in both readers: true here, where the left-to-right reading is false."""
    text = _SPACED_NET.replace(
        "if(Aobs < 1 or Bobs > 8, 1, 0)", "if(Aobs > 1 or Bobs > 8 and Bobs > 9, 1, 0)"
    )
    model = _load(tmp_path, text)
    net = write_net(model)
    assert "if(Aobs>1||Bobs>8&&Bobs>9,1,0)" in net
    again = _load(tmp_path, net, "again.net")
    times = np.linspace(0, 5, 21)
    np.testing.assert_allclose(_trajectory(again, times), _trajectory(model, times), rtol=1e-9)


def test_an_operator_with_no_one_token_spelling_is_refused(tmp_path):
    """`xor` between two words: joined it is another name, and run_network has
    no spelling of it. Refused under strict; written as it is, with a
    warning, otherwise."""
    text = _SPACED_NET.replace("Aobs > thr and k > 0.1", "Aobs > thr xor k > 0.1")
    model = _load(tmp_path, text)
    with pytest.raises(ConversionError, match="xor"):
        write_net(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net = write_net(model, strict=False)
    assert any(
        issubclass(w.category, ConversionWarning) and "xor" in str(w.message) for w in caught
    )
    assert "Aobs > thr xor k > 0.1" in net


def _run_network():
    res = resolve_bng()
    if not res.ok or res.root is None:
        return None
    path = res.root / "bin" / "run_network"
    return path if path.is_file() else None


@pytest.mark.skipif(
    _run_network() is None, reason="no BioNetGen install (BNG2.pl and its run_network) found"
)
def test_run_network_runs_the_written_net_as_bngsim_does(tmp_path):
    """The written file on run_network: bngsim's trajectory, to its tolerance.
    Before, the parameters block was refused, and a spaced function ran as its
    first token."""
    import antimony

    antimony.loadAntimonyString(
        "model m\n  compartment c1 = 2\n  species A in c1 = 10\n  k = 0.5\n  J0: A -> ; k*A\nend"
    )
    for model in (
        bngsim.Model.from_sbml_string(antimony.getSBMLString("m")),
        _load(tmp_path, _SPACED_NET),
    ):
        path = tmp_path / "written.net"
        write_net(model, path)
        out = tmp_path / "rn"
        run = subprocess.run(
            # -g: read the groups block too, which the functions read
            [str(_run_network()), "-o", str(out), "-p", "cvode", "-a", "1e-12", "-r", "1e-10"]
            + ["-g", str(path), str(path), "0.25", "20"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stdout[-600:] + run.stderr[-600:]
        rows = np.loadtxt(str(out) + ".cdat", comments="#")
        times = rows[:, 0]
        expected = _trajectory(bngsim.Model.from_net(str(path)), times)
        np.testing.assert_allclose(rows[:, 1:], expected, rtol=1e-6, atol=1e-9)
