"""Numbers in a .net file read the same under a decimal-comma LC_NUMERIC
(issue #1235).

The C++ side read numbers with strtod, std::stod and atof, which follow the C
locale's decimal point. In a process that had set LC_NUMERIC to de_DE or fr_FR,
"0.5" read as 0:

* a homodimer's stat factor ``0.5*kf1``, which BNG2.pl writes for every
  homodimer (321 of 1,732 local .net files), was refused as an unknown
  parameter;
* a group factor ``0.5*1,2`` read 0, with no error;
* an inline table's 2.5 and 7.5 read 2 and 7, with no error, and breakpoints
  ``[0,0.5,1.5,3]`` were refused as not monotonic;
* a literal ExprTk misreads (#1014) was left to it, one ulp off;
* a parameter ``1.5`` became an expression and a species ``100.5`` gained a
  synthetic initial-concentration parameter;
* the SSA propensity source bngsim's C++ compiles in-process wrote ``2,5``
  into its C, which reads as the comma operator: A(1) 45 for 57.5;
* and libantimony, which reads and writes numbers in the locale too, wrote
  ``k = 0.1`` into its SBML as 0, so the model had no rate.

The code bngsim runs in-process but did not write reads numbers the same way:

* RuleMonkey's loader read a TFUN's ``0.5,0.5`` as 0, 0, so the rule it drives
  never fired (AX 0 for 48), with no error;
* NFsim's TFUN parser refused a table with ``0.5`` in it;
* MIR's C front end refused every decimal literal, so no ODE model compiled.

Each runs in a subprocess, so the locale it sets is its own, and every value
is held against the same run under "C". Where no decimal-comma locale is
installed (a bare CI image), the module is skipped.
"""

from __future__ import annotations

import importlib.util
import json
import locale
import subprocess
import sys
import textwrap
from pathlib import Path

import bngsim
import pytest

_CANDIDATES = (
    "de_DE.UTF-8",
    "de_DE.utf8",
    "fr_FR.UTF-8",
    "fr_FR.utf8",
    "nl_NL.UTF-8",
    "nl_NL.utf8",
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"


def _comma_locale() -> str | None:
    saved = locale.setlocale(locale.LC_NUMERIC)
    try:
        for name in _CANDIDATES:
            try:
                locale.setlocale(locale.LC_NUMERIC, name)
            except locale.Error:
                continue
            if locale.localeconv()["decimal_point"] == ",":
                return name
        return None
    finally:
        locale.setlocale(locale.LC_NUMERIC, saved)


COMMA = _comma_locale()
pytestmark = pytest.mark.skipif(COMMA is None, reason="no decimal-comma locale is installed")

HAS_ANTIMONY = importlib.util.find_spec("antimony") is not None
_needs_antimony = pytest.mark.skipif(not HAS_ANTIMONY, reason="antimony is not installed")
_needs_nfsim = pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not compiled in")
_needs_rulemonkey = pytest.mark.skipif(
    not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not compiled in"
)
_needs_mir = pytest.mark.skipif(
    not bngsim.capabilities()["features"].get("mir", False), reason="built without the MIR backend"
)

_PROBE = textwrap.dedent(
    """
    import json, locale, os, pathlib, sys, tempfile, threading, warnings
    warnings.filterwarnings("ignore")
    try:
        import antimony
    except ImportError:
        antimony = None

    # Built under "C", before the switch, so the case below reads only
    # bngsim's own literals and not Antimony's (which the next case reads).
    SBML = None
    if antimony is not None:
        antimony.clearPreviousLoads()
        antimony.loadAntimonyString(
            "compartment c = 0.4; species A in c, B in c, C in c; A = 100; B = 100; C = 0;"
            " k = 0.003; R: A + B -> C; k*A*B"
        )
        SBML = antimony.getSBMLString(antimony.getMainModuleName())

    locale.setlocale(locale.LC_NUMERIC, sys.argv[1])
    import bngsim
    import numpy as np

    d = pathlib.Path(tempfile.mkdtemp())
    data = pathlib.Path(sys.argv[2])

    def net(name, params, species, reactions, groups=" 1 Aobs 1", functions=""):
        text = f"begin parameters\\n{params}\\nend parameters\\n"
        text += f"begin species\\n{species}\\nend species\\n"
        if functions:
            text += f"begin functions\\n{functions}\\nend functions\\n"
        text += f"begin reactions\\n{reactions}\\nend reactions\\n"
        text += f"begin groups\\n{groups}\\nend groups\\n"
        path = d / f"{name}.net"
        path.write_text(text)
        return bngsim.Model.from_net(str(path))

    def last(model, **kw):
        result = bngsim.Simulator(model, method="ode").run(**kw)
        return np.asarray(result.species)[-1].tolist(), result

    def edited(src, old, new):
        text = (data / src).read_text(encoding="utf-8")
        assert old in text, old
        path = d / pathlib.Path(src).name
        path.write_text(text.replace(old, new), encoding="utf-8")
        return str(path)

    def session(cls, xml):
        with cls(xml) as s:
            s.initialize(seed=3)
            r = s.simulate(0, 10, n_points=3)
        obs = np.asarray(r.observables)
        return {n: obs[:, i].tolist() for i, n in enumerate(r.observable_names)}

    class env:
        # Set for the case and put back after, as the caller had it.
        def __init__(self, name, value):
            self.name, self.value = name, value

        def __enter__(self):
            self.saved = os.environ.get(self.name)
            os.environ[self.name] = self.value

        def __exit__(self, *exc):
            if self.saved is None:
                del os.environ[self.name]
            else:
                os.environ[self.name] = self.saved

    out = {}

    def case(key):
        # Each case on its own, so one refused load (as on main) is one failure.
        def wrap(fn):
            try:
                out[key] = fn()
            except Exception as exc:
                out[key] = "raised: " + str(exc).splitlines()[0][:200]
        return wrap

    @case("stat factor")
    def _():
        m = net("dimer", " 1 kf1 0.1", " 1 A() 10\\n 2 B() 0", " 1 1,1 2 0.5*kf1")
        return last(m, t_span=(0, 1), n_points=2)[0]

    @case("group factor")
    def _():
        m = net(
            "half", " 1 k 1.5", " 1 A() 100.5\\n 2 B() 0", " 1 1 2 k", groups=" 1 Half 0.5*1,2"
        )
        return float(np.asarray(last(m, t_span=(0, 2), n_points=3)[1].observables["Half"])[0])

    for key, xs, ys in (
        ("table values", "0,1,2,3", "0,2.5,7.5,15"),
        ("table breakpoints", "0,0.5,1.5,3", "0,1,2,3"),
    ):
        @case(key)
        def _(key=key, xs=xs, ys=ys):
            m = net(key.replace(" ", "_"), " 1 k 1", " 1 A() 0", " 1 0 1 f #_FUNC",
                    functions=f" 1 f() tfun([{xs}],[{ys}],time())")
            return last(m, t_span=(0, 3), n_points=2)[0]

    def literals_model():
        params = " 1 a 7.77e-05*1\\n 2 b 1.14*1\\n 3 k 1.5"
        return net("literals", params, " 1 A() 100.5", " 1 1 0 k")

    @case("literals")
    def _():
        m = literals_model()
        return [float(m.get_param("a")).hex(), float(m.get_param("b")).hex()]

    @case("plain values")
    def _():
        m = literals_model()
        return {
            "species": list(m.species_names),
            "k is an expression": dict(zip(m.param_names, m.param_is_expression))["k"],
        }

    @case("ssa")
    def _():
        m = net("ssa", " 1 kf1 0.1", " 1 A() 50\\n 2 B() 0", " 1 1,1 2 0.5*kf1")
        r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 1), n_points=2, seed=7)
        return np.asarray(r.species)[-1].tolist()

    # The propensity source bngsim's C++ writes and compiles in-process: the
    # 1/V = 2.5 of a second-order reaction in a 0.4 compartment went in as 2,5.
    if SBML is not None:
        @case("ssa source literal")
        def _():
            with env("BNGSIM_SSA_PROP_CC", "1"):
                r = bngsim.Simulator(bngsim.Model.from_sbml_string(SBML), method="ssa").run(
                    t_span=(0, 1), n_points=2, seed=7
                )
            return np.asarray(r.species)[-1].tolist()

        # Antimony's own parse and writer, called under "C" by the loader.
        @case("antimony")
        def _():
            m = bngsim.Model.from_antimony_string("species A; A = 50; k = 0.1; R: A -> ; k*A")
            return [m.get_param("k"), last(m, t_span=(0, 1), n_points=2)[0]]

        # ... and "C" on this thread only: another thread, and the process's own
        # setting, keep the caller's locale (a process-wide switch did not).
        @case("antimony thread")
        def _():
            from bngsim._sbml_loader import _antimony_numbers_in_c_locale

            seen = {}
            with _antimony_numbers_in_c_locale():
                seen["this thread"] = locale.localeconv()["decimal_point"]
                other = threading.Thread(
                    target=lambda: seen.update(other=locale.localeconv()["decimal_point"])
                )
                other.start()
                other.join()
                seen["process"] = locale.setlocale(locale.LC_NUMERIC)
            seen["after"] = locale.localeconv()["decimal_point"]
            return seen

    # RuleMonkey's loader: the 0.5 of a time table read 0, so the rule it
    # drives never fired.
    if bngsim.HAS_RULEMONKEY:
        @case("rulemonkey table")
        def _():
            xml = edited("rate_law_time_table.xml", 'yData="0.1,5"', 'yData="0.5,0.5"')
            return session(bngsim.RuleMonkeySession, xml)

    # NFsim's TFUN parser: refused the model.
    if bngsim.HAS_NFSIM:
        @case("nfsim table")
        def _():
            xml = edited(
                "tfun_new_format/valid_inline_step_time.xml",
                'yData="0,10,20"',
                'yData="0.5,10.5,20.5"',
            )
            return session(bngsim.NfsimSession, xml)

    # MIR's C front end: every decimal literal was "wrong number".
    if bngsim.capabilities()["features"].get("mir", False):
        @case("mir")
        def _():
            with env("BNGSIM_CODEGEN_JIT", "mir"):
                m = net("mir", " 1 kf1 0.1", " 1 A() 10\\n 2 B() 0", " 1 1 2 f #_FUNC",
                        functions=" 1 f() 0.25*kf1")
                sim = bngsim.Simulator(m, method="ode", codegen=True)
                r = sim.run(t_span=(0, 1), n_points=2)
            return [sim.codegen_backend, np.asarray(r.species)[-1].tolist()]

    print(json.dumps(out))
    """
)


def _probe(name: str) -> dict:
    run = subprocess.run(
        [sys.executable, "-c", _PROBE, name, str(DATA)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def under_c() -> dict:
    return _probe("C")


@pytest.fixture(scope="module")
def under_comma() -> dict:
    return _probe(COMMA)


@pytest.mark.parametrize(
    "case",
    [
        "stat factor",
        "group factor",
        "table values",
        "table breakpoints",
        "literals",
        "plain values",
        "ssa",
        pytest.param("ssa source literal", marks=_needs_antimony),
        pytest.param("antimony", marks=_needs_antimony),
        pytest.param("rulemonkey table", marks=_needs_rulemonkey),
        pytest.param("nfsim table", marks=_needs_nfsim),
        pytest.param("mir", marks=_needs_mir),
    ],
)
def test_a_comma_locale_reads_the_same_model(under_c, under_comma, case):
    assert under_comma[case] == under_c[case]


def test_the_values_under_c_are_the_right_ones(under_c):
    """So the comparison above is against the right answer, not a shared wrong one,
    and not two copies of one error."""
    raised = {k: v for k, v in under_c.items() if isinstance(v, str)}
    assert not raised
    assert under_c["group factor"] == pytest.approx(50.25)
    assert under_c["table values"][0] == pytest.approx(17.5, rel=1e-6)
    assert under_c["literals"] == [(7.77e-05).hex(), (1.14).hex()]
    assert under_c["plain values"] == {"species": ["A()"], "k is an expression": False}
    if HAS_ANTIMONY:
        assert under_c["antimony"][0] == 0.1
    if bngsim.HAS_RULEMONKEY:
        assert under_c["rulemonkey table"]["AX"][-1] > 0
    if "mir" in under_c:
        assert under_c["mir"][0] == "mir"


@_needs_antimony
def test_the_antimony_switch_is_this_threads_alone(under_comma):
    seen = under_comma["antimony thread"]
    assert seen == {"this thread": ".", "other": ",", "process": COMMA, "after": ","}
