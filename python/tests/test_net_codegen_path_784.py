"""The #784 family on both codegen paths: the model path answers correctly today;
the .net path is quarantined until #803 step 3 routes .net models to it.

Each case is a reproduction from its issue, run through the public
``Model.from_net`` + ``Simulator`` on two paths and checked against a closed form
or scipy's ``solve_ivp``:

* ``net``   -- the path ``from_net``/``from_bngl`` models take today: codegen
  re-reads the ``.net`` text with its own parser (``_codegen._parse_net_file``);
* ``model`` -- the same model with ``_net_path`` cleared, the one switch
  ``Simulator`` reads, so C is emitted from the model bngsim built.

The ``net`` parameter of every case is ``xfail(strict=True, raises=...)`` naming
its issue. When step 3 routes ``.net`` models to the model path the ``net`` and
``model`` parameters run the same code, every quarantined case passes, and strict
xfail turns that into a failure -- so the markers have to come out in that PR,
not whenever someone remembers.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest
from scipy.integrate import solve_ivp


def _quarantine(issue: int, raises: type[BaseException], why: str):
    return pytest.mark.xfail(
        strict=True,
        raises=raises,
        reason=f"#{issue}: {why} on the .net codegen path; retires when .net "
        "models route to the model path (#803 step 3)",
    )


def _paths(issue: int, raises: type[BaseException], why: str) -> list:
    return [pytest.param("net", marks=_quarantine(issue, raises, why)), "model"]


def _refused(issue: int, why: str) -> list:
    """(sens, path) cases for a file the .net path refuses and the model path runs:
    its parser's ValueError reaches a codegen=True run as it is, and a
    sensitivity run wrapped in the RuntimeError that says the sensitivity RHS
    could not be built."""
    return [
        pytest.param(False, "net", marks=_quarantine(issue, ValueError, why), id="codegen-net"),
        pytest.param(False, "model", id="codegen-model"),
        pytest.param(
            True, "net", marks=_quarantine(issue, RuntimeError, why), id="sensitivity-net"
        ),
        pytest.param(True, "model", id="sensitivity-model"),
    ]


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _run(net, path, *, sens=None, t_end=1.0, n=3, override=None, model=None):
    m = model if model is not None else bngsim.Model.from_net(net)
    for name, value in (override or {}).items():
        m.set_param(name, value)
    if path == "model":
        m._net_path = ""  # the switch Simulator reads (#803)
    kw = {"sensitivity_params": sens} if sens else {"codegen": True}
    sim = bngsim.Simulator(m, method="ode", **kw)
    if not sens:
        # Compiled code on the path under test, not a quiet ExprTk run that would
        # pass the model leg trivially.
        assert sim.codegen_backend in ("cc", "mir")
    return sim.run(t_span=(0.0, t_end), n_points=n, rtol=1e-10, atol=1e-12)


def _write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text).lstrip())
    return p


# ── #689: stat-factor spellings and the legacy Sat/Hill rate laws ────────────

DECAY = """\
begin parameters
    1 k K
end parameters
begin species
    1 A() 10
end species
begin reactions
    1 1 0 RATE
end reactions
"""


@pytest.mark.parametrize("sens", [False, True], ids=["codegen", "sensitivity"])
@pytest.mark.parametrize(
    "tok",
    [".5*k", "5e-01*k", "1.6605779e-09*k", "1e+24*k"],
    ids=["leading-dot", "exponent", "cbngl-volume", "1e24"],
)
@pytest.mark.parametrize(
    "path",
    _paths(689, AssertionError, "a stat factor with an exponent or a leading dot is a zero rate"),
)
def test_689_stat_factor_spellings(tmp_path, path, tok, sens):
    """A -> 0 at sf*k with k = 1/sf: A = 10 e^-t, dA/dk = -sf t A. BNG2.pl prints a
    cBNGL volume conversion with %.8g, so e-notation is what it writes below 1e-4."""
    sf = float(tok.split("*")[0])
    net = _write(tmp_path, "d.net", DECAY.replace("RATE", tok).replace("K", repr(1.0 / sf)))
    r = _run(net, path, sens=["k"] if sens else None)
    t = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(r.species[:, 0], 10 * np.exp(-t), rtol=1e-7)
    if sens:
        np.testing.assert_allclose(
            r.sensitivities[:, 0, 0], -sf * t * 10 * np.exp(-t), rtol=1e-6, atol=1e-12
        )


LEGACY = {
    "Sat": (
        "begin parameters\n    1 k3  1.52\n    2 K4  114.418\nend parameters\n"
        "begin species\n    1 S() 100\n    2 P() 0\nend species\n"
        "begin reactions\n    1 1 2 Sat k3 K4\nend reactions\n",
        [1.52, 114.418],
        lambda S, p: -p[0] * S / (p[1] + S),
    ),
    "Hill": (
        "begin parameters\n    1 V   1.0\n    2 K   50.0\n    3 n   2.0\nend parameters\n"
        "begin species\n    1 S() 100\n    2 P() 0\nend species\n"
        "begin reactions\n    1 1 2 Hill V K n\nend reactions\n",
        [1.0, 50.0, 2.0],
        lambda S, p: -p[0] * S ** p[2] / (p[1] ** p[2] + S ** p[2]),
    ),
}


def _s10(dsdt, p):
    return solve_ivp(lambda t, y: [dsdt(y[0], p)], (0, 10), [100.0], rtol=1e-12, atol=1e-12).y[
        0, -1
    ]


@pytest.mark.filterwarnings("ignore:Legacy/deprecated BioNetGen")
@pytest.mark.parametrize("sens", [False, True], ids=["codegen", "sensitivity"])
@pytest.mark.parametrize("law", ["Sat", "Hill"])
@pytest.mark.parametrize(
    "path", _paths(689, AssertionError, "a legacy Sat/Hill rate law is a zero rate")
)
def test_689_legacy_sat_and_hill(tmp_path, path, law, sens):
    text, p0, dsdt = LEGACY[law]
    net = _write(tmp_path, f"{law}.net", text)
    first = "k3" if law == "Sat" else "V"
    r = _run(net, path, sens=[first] if sens else None, t_end=10.0, n=2)
    np.testing.assert_allclose(r.species[-1, 0], _s10(dsdt, p0), rtol=1e-7)
    if sens:
        h = 1e-6
        d = (_s10(dsdt, [p0[0] + h, *p0[1:]]) - _s10(dsdt, [p0[0] - h, *p0[1:]])) / (2 * h)
        np.testing.assert_allclose(r.sensitivities[-1, 0, 0], d, rtol=1e-5)


# ── #694: a set_param override of a derived parameter ────────────────────────

ELEM = """\
begin parameters
    1 k1   1     # Constant
    2 k2   2*k1  # ConstantExpression
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 k2
end reactions
"""
FUNC = """\
begin parameters
    1 k1   1     # Constant
    2 k2   2*k1  # ConstantExpression
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin groups
    1 Atot 1
end groups
begin functions
    1 frate() k2*Atot
end functions
begin reactions
    1 1 2 frate
end reactions
"""


@pytest.mark.parametrize(
    "path",
    _paths(694, AssertionError, "a pinned derived rate constant keeps its stale chain rule"),
)
def test_694_elementary_override_drops_the_chain_rule(tmp_path, path):
    """k2 pinned to 5: A = 10 e^-5t does not depend on k1."""
    r = _run(_write(tmp_path, "e.net", ELEM), path, sens=["k1"], override={"k2": 5.0})
    assert r.species[-1, 0] == pytest.approx(10 * math.exp(-5), rel=1e-7)
    assert r.sensitivities[-1, 0, 0] == pytest.approx(0.0, abs=1e-10)


_CHILD = """
import json, sys, warnings
warnings.simplefilter("ignore")
import bngsim
net, mode, path = sys.argv[1:4]
m = bngsim.Model.from_net(net)
if mode == "override":
    m.set_param("k2", 5.0)
if path == "model":
    m._net_path = ""
r = bngsim.Simulator(m, method="ode", sensitivity_params=["k1"]).run(
    t_span=(0.0, 1.0), n_points=3, rtol=1e-10, atol=1e-12)
print("ROW" + json.dumps([float(r.species[-1, 0]), float(r.sensitivities[-1, 0, 0])]))
"""


@pytest.mark.parametrize("order", [("override", "pristine"), ("pristine", "override")])
@pytest.mark.parametrize(
    "path",
    _paths(
        694,
        AssertionError,
        "the .so cache key cannot tell a pinned derived parameter from a pristine one",
    ),
)
def test_694_cache_serves_each_attachment_state_its_own_code(tmp_path, path, order):
    """dA/dt = -k2 A^2: A = 10/(1 + 10 k2 t). Pristine, dA/dk1 = 2 dA/dk2; pinned,
    0. Two processes share one codegen cache, in both orders."""
    net = _write(tmp_path, "f.net", FUNC)
    want = {"pristine": [10 / 21, -200 / 21**2], "override": [10 / 51, 0.0]}
    env = {**os.environ, "BNGSIM_CODEGEN_CACHE_DIR": str(tmp_path / "shared")}
    for mode in order:
        out = subprocess.run(
            [sys.executable, "-c", _CHILD, str(net), mode, path],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        got = json.loads([ln for ln in out.stdout.splitlines() if ln.startswith("ROW")][-1][3:])
        np.testing.assert_allclose(got, want[mode], rtol=1e-7, atol=1e-10)


# ── #699: a function that reads one declared after it ───────────────────────

FWD = """\
begin parameters
    1 k 2
end parameters
begin species
    1 A() 10
end species
begin reactions
    1 1 0 y
end reactions
begin groups
    1 Atot 1
end groups
begin functions
    1 y() x*3
    2 x() k*0.5
end functions
"""


def _fwd_expect(r, sens):
    t = np.array([0.0, 0.5, 1.0])  # y = 1.5 k = 3: A = 10 e^-3t, dA/dk = -15 t e^-3t
    np.testing.assert_allclose(r.species[:, 0], 10 * np.exp(-3 * t), rtol=1e-7)
    if sens:
        np.testing.assert_allclose(
            r.sensitivities[:, 0, 0], -15 * t * np.exp(-3 * t), rtol=1e-6, atol=1e-12
        )


@pytest.mark.parametrize("sens", [False, True], ids=["codegen", "sensitivity"])
@pytest.mark.parametrize(
    "path", _paths(699, RuntimeError, "a forward function reference fails to compile")
)
def test_699_forward_reference_flat(tmp_path, monkeypatch, path, sens):
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "off")
    _fwd_expect(_run(_write(tmp_path, "fwd.net", FWD), path, sens=["k"] if sens else None), sens)


def _reads_before_written(src: str) -> list[str]:
    """``func[]`` slots the emitted C reads before any line has assigned them."""
    written: set[str] = set()
    early = []
    for line in src.splitlines():
        m = re.match(r"\s*func\[(\d+)\]\s*=(.*);", line)
        if m:
            early += [
                f"func[{j}] in `{line.strip()}`"
                for j in sorted(set(re.findall(r"func\[(\d+)\]", m.group(2))) - written)
            ]
            written.add(m.group(1))
    return early


def _source(net, path) -> str:
    m = bngsim.Model.from_net(net)
    if path == "model":
        m._net_path = ""
        return cg.prepare_model_codegen_source(m)
    return cg.prepare_codegen_source(net, m)


@pytest.mark.parametrize("sens", [False, True], ids=["codegen", "sensitivity"])
@pytest.mark.parametrize(
    "path",
    _paths(699, AssertionError, "a chunked build reads a forward function's uninitialised slot"),
)
def test_699_forward_reference_chunked(tmp_path, monkeypatch, path, sens):
    """The defect is a read of an uninitialised stack slot, and what that slot
    holds is up to the platform: on Linux CI it held the right value, so the run
    passed and the strict xfail failed. The emitted C's order is checked first,
    because that is the same everywhere."""
    monkeypatch.setenv("BNGSIM_CODEGEN_CHUNK", "on")
    net = _write(tmp_path, "fwd.net", FWD)
    early = _reads_before_written(_source(net, path))
    assert not early, f"read before written: {early}"
    _fwd_expect(_run(net, path, sens=["k"] if sens else None), sens)


# ── #721: a tfun time index spelled T ────────────────────────────────────────

TFUN_NET = """\
begin parameters
    1 T   0.5
    2 k   0.1
end parameters
begin functions
    1 drv()  tfun('drive.tfun', T)
end functions
begin species
    1 A() 1
    2 B() 0
end species
begin reactions
    1 1 1,2 drv #R1
end reactions
"""
TFUN = "# T  drv\n0       1.0\n1       2.0\n2       4.0\n5       10.0\n"


def _tfun_model(tmp_path, with_param_t: bool):
    (tmp_path / "drive.tfun").write_text(TFUN)
    text = (
        TFUN_NET if with_param_t else TFUN_NET.replace("    1 T   0.5\n", "").replace("2 k", "1 k")
    )
    return _write(tmp_path, "m.net", text)


@pytest.mark.parametrize("sens", [False, True], ids=["codegen", "sensitivity"])
@pytest.mark.parametrize(
    "path", _paths(721, AssertionError, "a tfun index T reads a parameter named T, not the clock")
)
def test_721_tfun_index_T_is_the_clock_beside_a_parameter_T(tmp_path, path, sens):
    """B' = drv(t) A with A = 1: B(3) = the table's integral over [0, 3] = 9.5."""
    r = _run(_tfun_model(tmp_path, True), path, sens=["k"] if sens else None, t_end=3.0)
    assert r.species[-1, 1] == pytest.approx(9.5, rel=1e-7)


@pytest.mark.parametrize(
    ("sens", "path"), _refused(721, "a tfun index T with no symbol named T is refused")
)
def test_721_tfun_index_T_with_no_symbol_T(tmp_path, path, sens):
    r = _run(_tfun_model(tmp_path, False), path, sens=["k"] if sens else None, t_end=3.0)
    assert r.species[-1, 1] == pytest.approx(9.5, rel=1e-7)


# ── #730 / #731: the file changes under a loaded model ───────────────────────

REWRITE = """\
begin parameters
    1 k1 1
    2 k2 5
end parameters
begin species
    1 X() 10
    2 Y() 0
end species
begin reactions
    1 1 2 RATE #_R1
end reactions
"""


@pytest.mark.parametrize("sens", [False, True], ids=["codegen", "sensitivity"])
@pytest.mark.parametrize(
    "path", _paths(730, AssertionError, "codegen compiles whatever file is at _net_path now")
)
def test_730_codegen_compiles_the_loaded_model_not_the_file(tmp_path, path, sens):
    """Loaded with rate k1 = 1, file then rewritten to k2: X = 10 e^-t."""
    net = _write(tmp_path, "m.net", REWRITE.replace("RATE", "k1"))
    m = bngsim.Model.from_net(net)
    net.write_text(REWRITE.replace("RATE", "k2"))
    r = _run(net, path, sens=["k1"] if sens else None, model=m)
    assert r.species[-1, 0] == pytest.approx(10 * math.exp(-1), rel=1e-7)
    if sens:
        assert r.sensitivities[-1, 0, 0] == pytest.approx(-10 * math.exp(-1), rel=1e-6)


@pytest.mark.parametrize(
    "path", _paths(731, AssertionError, "the in-process memo keys the .net by mtime only")
)
def test_731_same_mtime_rewrite_is_not_served_the_old_network(tmp_path, path):
    base = (
        "begin parameters\n    1 k 1\nend parameters\n"
        "begin species\n    1 A() 10\n    2 B() 0\n    3 C() 0\nend species\n"
    )
    v1 = _write(tmp_path, "v1.net", base + "begin reactions\n    1 1 2 k\nend reactions\n")
    v2 = _write(tmp_path, "v2.net", base + "begin reactions\n    1 1 3 k\nend reactions\n")
    st = os.stat(v1)
    os.utime(v2, ns=(st.st_atime_ns, st.st_mtime_ns))
    target = tmp_path / "model.net"
    shutil.copy2(v1, target)
    _run(target, path)
    shutil.copy2(v2, target)  # same bytes length, same mtime: cp -p / tar x
    r = _run(target, path)
    np.testing.assert_allclose(
        r.species[-1], [10 * math.exp(-1), 0.0, 10 * (1 - math.exp(-1))], rtol=1e-7, atol=1e-9
    )


# ── #608: the unindexed blocks the documented loaders read ───────────────────

UNINDEXED = """\
begin parameters
    kcat 0.3
end parameters
begin species
    A() 100
    B() 0
end species
begin groups
    Atot 1
end groups
begin functions
    1 fA() kcat*Atot
end functions
begin reactions
    1 1 2 fA
end reactions
"""


@pytest.mark.parametrize(
    ("sens", "path"), _refused(608, "an unindexed parameters/species/groups block is refused")
)
def test_608_unindexed_blocks(tmp_path, path, sens):
    """dA/dt = -kcat A^2: A = 100/(1 + 100 kcat t), dA/dkcat = -1e4 t/(1 + 100 kcat t)^2."""
    r = _run(_write(tmp_path, "u.net", UNINDEXED), path, sens=["kcat"] if sens else None)
    assert r.species[-1, 0] == pytest.approx(100 / 31, rel=1e-7)
    if sens:
        assert r.sensitivities[-1, 0, 0] == pytest.approx(-1e4 / 31**2, rel=1e-6)
