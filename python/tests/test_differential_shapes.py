"""Shape models through every path that should give the same answer (issue #1257).

``parity_checks/differential/_differential.py`` runs one model through the
paths that should agree (a sensitivity run's state and a plain run's, analytic
and FD Jacobians, KLU and dense, compiled and interpreted, cc and MIR) and three
rules for bugs whose answers are right (a sensitivity run refused, a sparsity
pattern that misses entries, a backend other than the one asked for). The nightly
runs it over the corpus (``parity_checks/differential``); this runs it over the
shapes the corpus rarely holds, on every pull request:

* triangle pulses over the clock written with ``max``/``abs``, ``clamp`` and
  ``if()``, and a ``clamp`` ramp, each wide (w = 0.5) and narrow (w = 0.01):
  only the narrow ones, at several tolerances, found #1227;
* an equality the state sits on, ``if(Bobs==0, kb, 0)`` (#1228, #1254);
* a derived rate constant over ``log10`` (#1214, fixed by #1224);
* a state trigger that crosses exactly at a sample time (#1182);
* a compartment an assignment rule moves, so its species are diluted;
* ``rateOf`` across compartments (#1266);
* a daily pulse whose width an event sets, between pulses and inside one
  (#1244: the grid check, 8x the output points read at the base's own, found
  it 7.5x the trajectory's peak off before #1263).

Each check is asked to pass or to say why it does not apply. A known bug that is
still open is a strict xfail naming its issue, so its fix turns it into a
failure to be removed. Each check is also shown to fail on a defect it is there
to see (the negative controls below), so a check that cannot fail does not
read as a pass.

The locale check is the nightly's: it needs a comma-decimal locale, which the
nightly generates and this suite's runners do not have.
"""

from __future__ import annotations

import atexit
import functools
import json
import shutil
import sys
import tempfile
from pathlib import Path

import bngsim
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "parity_checks" / "differential"))

import _differential as D  # noqa: E402

antimony = pytest.importorskip("antimony")

# The ODE checks: the SSA arm (#1310) runs on SSA jobs, which the shapes are not.
CHECKS = tuple(c for c in D.CHECKS if c != "locale" and c not in D.SSA_CHECKS)
_DIR = Path(tempfile.mkdtemp(prefix="bngsim-differential-shapes-"))
atexit.register(shutil.rmtree, _DIR, ignore_errors=True)


def _net(name, params, species, functions, reactions, groups) -> Path:
    path = _DIR / f"{name}.net"
    path.write_text(
        f"begin parameters\n{params}\nend parameters\nbegin species\n{species}\nend species\n"
        f"begin functions\n{functions}\nend functions\n"
        f"begin reactions\n{reactions}\nend reactions\nbegin groups\n{groups}\nend groups\n"
    )
    return path


def _sbml(name, text) -> Path:
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(f"model m\n{text}\nend") >= 0, antimony.getLastError()
    path = _DIR / f"{name}.xml"
    path.write_text(antimony.getSBMLString("m"))
    return path


_PULSE = " 1 kx 1\n 2 k_clock 2\n 3 c 5\n 4 w {w}"
_LAWS = {
    "triangle-maxabs": "kx*(1/w)*max(0,1-abs(Tobs-c)/(w/2))",
    "triangle-clamp": "kx*(1/w)*(clamp(0,(Tobs-c+w/2)/(w/2),1)-clamp(0,(Tobs-c)/(w/2),1))",
    "triangle-if": "kx*(1/w)*if(Tobs<c-w/2,0,if(Tobs<c,(Tobs-c+w/2)/(w/2),"
    "if(Tobs<c+w/2,(c+w/2-Tobs)/(w/2),0)))",
    "ramp-clamp": "kx*clamp(0,(Tobs-c)/w,1)",
}


def _shapes() -> dict[str, D.Job]:
    jobs = {}
    for name, law in _LAWS.items():
        for suffix, w in (("", "0.5"), ("-narrow", "0.01")):
            path = _net(
                name + suffix,
                _PULSE.format(w=w),
                " 1 T() 0\n 2 X() 10\n 3 Y() 0",
                f" 1 f() {law}",
                " 1 0 1 k_clock\n 2 2 3 f",
                " 1 Tobs 1",
            )
            jobs[name + suffix] = D.Job(name + suffix, path, 0.0, 10.0, 21)
    jobs["equality-dwell"] = D.Job(
        "equality-dwell",
        _net(
            "equality-dwell",
            " 1 a 0.5\n 2 q 2.0\n 3 kb 1.0\n 4 kp 1.0",
            " 1 A() 10\n 2 B() 0\n 3 Y() 0",
            " 1 fB() kp*if(q-Aobs>0,q-Aobs,0)\n 2 fY() if(Bobs==0,kb,0)",
            " 1 1 0 a\n 2 0 2 fB\n 3 0 3 fY",
            " 1 Aobs 1\n 2 Bobs 2",
        ),
        0.0,
        10.0,
        21,
    )
    jobs["derived-log10"] = D.Job(
        "derived-log10",
        _net(
            "derived-log10",
            " 1 p 100.0\n 2 k log10(p)",
            " 1 A() 1.0\n 2 B() 0",
            "",
            " 1 1 2 k",
            " 1 Aobs 1",
        ),
        0.0,
        10.0,
        21,
    )
    sbml = {
        # A = 10·exp(-t·ln2/5) is 5 at t = 5, a sample time
        "event-at-a-sample": (
            "A = 10; k = 0.1386294361119891; J: A -> ; k*A; B = 0; E: at (A < 5): B = B + 1;",
            10.0,
        ),
        "moving-compartment": (
            "compartment cv; cv := 1 + 0.1*time; species X in cv = 4; k = 0.25; J: X -> ; k*X*cv;",
            10.0,
        ),
        "rateof-across": (
            "compartment C1 = 2; compartment C2 = 3; k = 0.3; g = 0.7; species A in C1 = 1;"
            " species B in C2 = 0; species S in C1 = 0; J: A -> B; k*A*C1; S' = g*rateOf(B);",
            20.0,
        ),
        "pulse-width-event": (
            "compartment C = 1; species X in C; X = 0; w = 1; a = 1;"
            " J0: -> X; a*piecewise(1, (time - 24*floor(time/24)) < w, 0);"
            " E1: at (time >= 50): w = 2;",
            230.0,
        ),
        # the same, the width set inside a pulse (t = 72.5, in the one from 72)
        "pulse-width-event-midpulse": (
            "compartment C = 1; species X in C; X = 0; w = 1; a = 1;"
            " J0: -> X; a*piecewise(1, (time - 24*floor(time/24)) < w, 0);"
            " E1: at (time >= 72.5): w = 2;",
            230.0,
        ),
    }
    for name, (text, t_end) in sbml.items():
        jobs[name] = D.Job(name, _sbml(name, text), 0.0, t_end, 21)
    return jobs


SHAPES = _shapes()

# (shape, check) -> what is expected instead of a pass: a declared refusal is
# one by name; an open bug is a strict xfail on its issue.
REFUSED = {
    # #1240: refused by name, the state on the equality's surface
    ("equality-dwell", "sens_runs"): "unsupported",
}
OPEN = {
    ("equality-dwell", "klu_vs_dense"): "#1254: KLU with the FD Jacobian on an exact equality",
}


@functools.cache
def _verdicts(shape: str) -> dict[str, D.Verdict]:
    return D.check_all(SHAPES[shape], None, CHECKS)


def _cases():
    for shape in SHAPES:
        for check in CHECKS:
            marks = ()
            if (shape, check) in OPEN:
                # Only where the path runs: a build without KLU skips the KLU
                # check, which is right, and is no pass of #1254's (the MIR leg
                # builds without it).
                marks = pytest.mark.xfail(
                    condition=check != "klu_vs_dense" or bool(bngsim._bngsim_core.HAS_KLU),
                    strict=True,
                    reason=OPEN[shape, check],
                )
            yield pytest.param(shape, check, id=f"{shape}-{check}", marks=marks)


@pytest.mark.parametrize(("shape", "check"), list(_cases()))
def test_each_path_gives_the_same_answer(shape, check):
    v = _verdicts(shape)[check]
    expected = REFUSED.get((shape, check))
    if expected is not None:
        assert v.status == expected, v
    else:
        assert v.status in ("pass", "skip"), f"{v.status}: {v.note}"


@pytest.mark.parametrize("shape", [s for s in SHAPES if (s, "sens_runs") not in REFUSED])
def test_the_sensitivity_rhs_is_analytic(shape):
    """A slow fallback with right answers is the #1214 class: the answers agree,
    so only the path it took shows it. derived-log10 declined until #1224."""
    v = _verdicts(shape)["sens_runs"]
    assert v.extra.get("sens_rhs") == "analytic", v.extra


# ── Negative controls: each check fails on the defect it is there to see ──


def test_a_path_off_by_more_than_the_bound_is_a_finding():
    base = D.Base(SHAPES["triangle-maxabs"])
    assert base.compare(base.x).status == "pass"
    off = base.x * (1 + 10 * base.limit)
    v = base.compare(off)
    assert v.status == "diff" and v.value > base.limit, v


def test_a_path_within_atol_is_not_a_finding():
    """A cell is judged past an absolute floor of the run's atol, scaled as the
    comparison scales it: a species at 1e-14 in a model of 10 is not one."""
    base = D.Base(SHAPES["triangle-maxabs"])
    x = base.x.copy()
    x[:, -1] += 0.5 * D.PATH_RTOL_FACTOR * base.job.atol
    assert base.compare(x).status == "pass"


def test_a_trajectory_smaller_than_atol_is_judged_past_it():
    """A nanomolar model's every cell is within atol of another path's: no cell
    of a run held to atol is asked to agree closer. Judged on the peak alone,
    a difference of 5e-11 in a state of 1e-9 read as 5%."""
    ref = np.full((3, 2), 1e-9)
    assert D.gap(ref + 5e-11, ref, 1e-10) == 0.0
    assert D.gap(ref + 5e-10, ref, 1e-10) > 0.0


def test_a_shape_or_a_non_finite_cell_is_a_finding():
    ref = np.ones((3, 2))
    assert D.gap(np.ones((3, 3)), ref, 1e-10) == float("inf")
    bad = ref.copy()
    bad[1, 1] = np.nan
    assert D.gap(bad, ref, 1e-10) == float("inf")


class _FakeCore:
    def __init__(self, pattern):
        self.jacobian_sparsity = pattern


class _FakeModel:
    """dA/dt = -k·A·B, dB/dt = -A: every entry but d(dB/dt)/dB is nonzero."""

    def __init__(self, pattern):
        self._core = _FakeCore(pattern)

    @staticmethod
    def rhs(y, t=0.0):
        a, b = y
        return np.array([-2.0 * a * b, -a])


def _csc(dense):
    dense = np.asarray(dense, dtype=bool)
    n = dense.shape[0]
    cols, rows = [0], []
    for j in range(n):
        rows += [i for i in range(n) if dense[i, j]]
        cols.append(len(rows))
    return {"n": n, "col_ptrs": np.array(cols), "row_indices": np.array(rows)}


def test_a_sparsity_pattern_that_misses_an_entry_is_a_finding():
    y = [np.array([1.0, 2.0])]
    full = _FakeModel(_csc([[1, 1], [1, 0]]))
    assert D.sparsity_misses(full, y) == []
    missing = _FakeModel(_csc([[1, 0], [1, 0]]))  # no slot for d(dA/dt)/dB
    assert D.sparsity_misses(missing, y) == [(0, 1)]


def test_an_entry_the_differences_cannot_resolve_is_not_asked_of_the_pattern():
    """An entry under the floor of the largest contribution is no entry."""

    class Tiny(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            a, b = y
            return np.array([-a + 1e-14 * b, -a])

    assert D.sparsity_misses(Tiny(_csc([[1, 0], [1, 0]])), [np.array([1.0, 2.0])]) == []


class _JacModel(_FakeModel):
    """_FakeModel with an analytic Jacobian, ``wrong`` written over its entries."""

    def __init__(self, wrong=None, source="analytical"):
        super().__init__(_csc([[1, 1], [1, 0]]))
        self.wrong, self.source = wrong or {}, source

    def jacobian(self, y, t=0.0):
        from bngsim._evaluators import JacobianMatrix

        a, b = y
        J = np.array([[-2.0 * b, -2.0 * a], [-1.0, 0.0]])
        for (i, j), v in self.wrong.items():
            J[i, j] = v
        return JacobianMatrix(J, self.source)


def test_a_jacobian_with_one_wrong_entry_is_a_finding():
    """#1308's control: d(dA/dt)/dB is -2a = -2; written as -2.2 (10% off), the
    check names that entry and no other."""
    y = [np.array([1.0, 2.0])]
    assert D.jacobian_misfits(_JacModel(), y) == []
    found = D.jacobian_misfits(_JacModel({(0, 1): -2.2}), y)
    assert [m["entry"] for m in found] == [(0, 1)]
    assert found[0]["off"] == pytest.approx(0.2 / 2.2, rel=1e-3)


def test_a_one_percent_error_is_a_finding():
    found = D.jacobian_misfits(_JacModel({(1, 0): -1.01}), [np.array([1.0, 2.0])])
    assert [m["entry"] for m in found] == [(1, 0)]


def test_a_spurious_entry_where_the_rhs_reads_nothing_is_a_finding():
    """d(dB/dt)/dB is 0 in the RHS; an analytic -0.5 there is wrong."""
    found = D.jacobian_misfits(_JacModel({(1, 1): -0.5}), [np.array([1.0, 2.0])])
    assert [m["entry"] for m in found] == [(1, 1)]


def test_a_non_finite_analytic_entry_is_a_finding():
    found = D.jacobian_misfits(_JacModel({(0, 0): float("nan")}), [np.array([1.0, 2.0])])
    assert [m["entry"] for m in found] == [(0, 0)] and found[0]["off"] == float("inf")


def test_a_differenced_jacobian_is_not_judged():
    """Model.jacobian differences the RHS where the closed form is incomplete:
    nothing analytic to judge."""
    y = [np.array([1.0, 2.0])]
    assert D.jacobian_misfits(_JacModel(source="finite-difference"), y) is None


def test_a_model_whose_difference_quotient_jacobian_refuses_is_skipped(monkeypatch):
    """Model.jacobian() refuses a difference quotient whose step decides an
    entry (#1002), where it returned one: a model without the closed form is
    SKIP, as the baseline holds it, not an EXCEPTION. BIOMD0000000053 and six
    more BioModels read EXCEPTION once main refused."""
    base = _base("derived-log10")

    def refuses(self, y, t=0.0, **kw):
        raise bngsim.SimulationError(
            "Model.jacobian() does not return this difference quotient (issue #1002)"
        )

    monkeypatch.setattr(bngsim.Model, "prepare_analytical_jacobian", lambda self: False)
    monkeypatch.setattr(bngsim.Model, "jacobian", refuses)
    v = D.check_jacobian_values(base)
    assert v.status == "skip" and "no analytic Jacobian" in v.note, v


def test_rounding_in_a_large_rhs_is_not_a_finding():
    """blbr's false alarm: f is 1e11, so a difference over a small step rounds by
    eps·|f|/h. At a step of 1e-6 its differences agreed on 2.0345 at two steps
    where the entry is 2. The gaps between neighbouring steps bound that
    rounding, so it does not read as a wrong entry even on a ladder of small
    steps only, while on the check's own ladder a 10% error still reads as one.

    (Rounding is the limit of what this check can judge: an entry no step
    resolves is not judged at all.)"""

    class Big(_JacModel):
        @staticmethod
        def rhs(y, t=0.0):
            a, b = y
            return np.array([1e9 - 2.0 * a * b, -a])

    y = [np.array([1.0, 2.0])]
    assert D.jacobian_misfits(Big(), y) == []
    assert D.jacobian_misfits(Big(), y, steps=(1e-6, 5e-7, 2.5e-7)) == []
    assert [m["entry"] for m in D.jacobian_misfits(Big({(0, 1): -2.2}), y)] == [(0, 1)]


def test_an_entry_on_a_kink_is_not_judged():
    """The #368 trap: dA/dt = -|b - 2| at b = 2. Every central difference reads
    0, the mean of the slopes -1 and +1 either side; the analytic entry is one
    side. Neither is the derivative, so the entry is not judged."""
    from bngsim._evaluators import JacobianMatrix

    class Kink(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            return np.array([-abs(y[1] - 2.0), -y[0]])

        def jacobian(self, y, t=0.0):
            return JacobianMatrix(np.array([[0.0, -1.0], [-1.0, 0.0]]), "analytical")

    assert D.jacobian_misfits(Kink(_csc([[0, 1], [1, 0]])), [np.array([1.0, 2.0])]) == []


def test_a_feature_narrower_than_the_coarse_steps_is_not_a_finding():
    """BIOMD0000000498's shape: dA/dt = 5 - 2b + 1e-9·tanh(b/1e-9) at b = 0. Its
    slope at the point is -1, the analytic entry; at steps far over 1e-9 the
    differences settle, step after step, on the secant -2, and with the
    constant's rounding at the fine steps that plateau is the steps' tightest.
    A finding needs the analytic entry contradicted at every step that
    resolves it, and the fine steps agree with it; judged at the tightest step
    alone, this read as a wrong entry.

    (A feature narrower still than any step rounding lets the ladder resolve
    is past what the check can tell from a wrong entry: at 1e-10 under a
    constant of 10 it reads as one.)"""
    from bngsim._evaluators import JacobianMatrix

    class Narrow(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            return np.array([5.0 - 2.0 * y[1] + 1e-9 * np.tanh(y[1] / 1e-9), -y[0]])

        def jacobian(self, y, t=0.0):
            return JacobianMatrix(np.array([[0.0, -1.0], [-1.0, 0.0]]), "analytical")

    y = [np.array([1.0, 0.0])]
    assert D.jacobian_misfits(Narrow(_csc([[0, 1], [1, 0]])), y) == []
    # and with the analytic entry the coarse secant's -2, which no fine step
    # bears out, it is no finding either: the steps disagree with each other.
    wrong = Narrow(_csc([[0, 1], [1, 0]]))
    wrong.jacobian = lambda yy, t=0.0: JacobianMatrix(
        np.array([[0.0, -3.0], [-1.0, 0.0]]), "analytical"
    )
    assert [m["entry"] for m in D.jacobian_misfits(wrong, y)] == [(0, 1)]


def test_a_slope_rounded_away_at_the_fine_steps_is_not_a_finding():
    """BIOMD0000000393's shape (#533): dA/dt = (N/2)(sqrt(1 + 4b/N) - 1)
    + 100(b - 1)^3 - 0.3b at b = 1, with 4b/N = 1e-10. Under a step of about
    2e-6 of b, 1 + 4b/N rounds to what it was, so the root's slope of 1 is
    lost and the fine steps agree, step after step, on -0.3; the analytic
    entry is 0.7. The coarse steps, a few percent apart from each other for
    the cubic, read 0.7: too noisy to resolve the entry, which the fine steps
    seem to, but enough to bear the analytic entry out, so it is no finding.

    (Rounding lost deeper inside a law than the coarse steps can speak to is
    past what the check can tell from a wrong entry: at 4b/N = 1e-11 with no
    cubic it reads as one.)"""
    from bngsim._evaluators import JacobianMatrix

    N = 4.0 / 1e-10

    class Plateau(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            b = y[1]
            root = 0.5 * N * (np.sqrt(1.0 + 4.0 * b / N) - 1.0)
            return np.array([root + 100.0 * (b - 1.0) ** 3 - 0.3 * b, -y[0]])

        def jacobian(self, y, t=0.0):
            slope = 1.0 / np.sqrt(1.0 + 4.0 * y[1] / N) - 0.3
            return JacobianMatrix(np.array([[0.0, slope], [-1.0, 0.0]]), "analytical")

    assert D.jacobian_misfits(Plateau(_csc([[0, 1], [1, 0]])), [np.array([1.0, 1.0])]) == []


def test_a_column_of_noise_is_not_judged_against_itself():
    """mm_tqssa_stiff's false alarm: a column whose analytic entries are 2e-19
    and whose differences are 0, beside entries of order 1. On its own peak it
    reads 100% off; against the matrix it is noise."""
    from bngsim._evaluators import JacobianMatrix

    class Noise(_FakeModel):
        """dA/dt = 1 - a, dB/dt = 0, at its steady state a = 1: nothing reads
        B, so column B is 0, and the analytic side carries 2e-19 there. f is 0
        at the state, so the differences round to nothing either, and only the
        matrix's own scale says 2e-19 is noise."""

        @staticmethod
        def rhs(y, t=0.0):
            return np.array([1.0 - y[0], 0.0])

        def jacobian(self, y, t=0.0):
            return JacobianMatrix(np.array([[-1.0, 2e-19], [0.0, 0.0]]), "analytical")

    assert D.jacobian_misfits(Noise(_csc([[1, 1], [0, 0]])), [np.array([1.0, 2.0])]) == []


def test_jacobian_values_on_a_real_model_with_a_wrong_entry_is_a_diff(monkeypatch):
    """The check end to end: a real shape's analytic Jacobian, its largest entry
    10% off. derived-log10's entries are of order 2 at every state; the
    triangle pulses' are all 0 (their rates read the clock only), where a
    scaled entry changes nothing."""
    from bngsim._evaluators import JacobianMatrix

    real = bngsim.Model.jacobian

    def wrong(self, y, t=0.0, **kw):
        J = real(self, y, t, **kw)
        out = np.array(J, dtype=float)
        k = np.unravel_index(np.argmax(np.abs(out)), out.shape)
        assert out[k] != 0.0, "the corruption must change the matrix"
        out[k] *= 1.1
        return JacobianMatrix(out, J.source)

    base = _base("derived-log10")
    assert D.check_jacobian_values(base).status == "pass"
    monkeypatch.setattr(bngsim.Model, "jacobian", wrong)
    v = D.check_jacobian_values(base)
    assert v.status == "diff", v


def test_a_backend_other_than_the_one_asked_is_a_finding(monkeypatch):
    """#1213: codegen=False was ignored on MIR builds. Here the property lies."""
    monkeypatch.setattr(bngsim.Simulator, "codegen_backend", property(lambda self: "mir"))
    v = D.check_backend(_base("triangle-maxabs"))
    assert v.status == "diff" and "codegen=False ran mir" in v.note, v


def test_a_sensitivity_run_refused_is_unsupported_not_a_pass():
    v = _verdicts("equality-dwell")["sens_runs"]
    assert v.status == "unsupported" and "SensitivityUnsupportedError" in v.note


@functools.cache
def _base(shape):
    return D.Base(SHAPES[shape])


def test_the_conditioning_gate_keeps_a_chaotic_model_from_a_false_finding(monkeypatch):
    """Lorenz over t = 25 amplifies every difference: the FD Jacobian's path is
    3.4e-4 from the base, past the 1e-5 that 1e3·rtol allows, and both are
    right, for the base moves by 0.79 at a tenfold tighter tolerance itself. Such a
    base is no reference at all, and nothing is judged against it; with the
    drift as the bound alone, it passes; with neither, it is a finding."""
    job = D.Job(
        "lorenz",
        _sbml(
            "lorenz",
            "s = 10; r = 28; b = 2.6666666666666665; X = 1; Y = 1; Z = 1;"
            " X' = s*(Y - X); Y' = X*(r - Z) - Y; Z' = X*Y - b*Z;",
        ),
        0.0,
        25.0,
        21,
        1e-8,
        1e-10,
    )
    gated = D.check_ajac(D.Base(job))
    assert gated.status == "skip" and "no reference" in gated.note, gated
    monkeypatch.setattr(D, "UNJUDGEABLE_DRIFT", float("inf"))
    bounded = D.check_ajac(D.Base(job))
    assert bounded.status == "pass", bounded
    monkeypatch.setattr(D, "DRIFT_FACTOR", 0.0)
    ungated = D.check_ajac(D.Base(job))
    assert ungated.status == "diff", ungated


def test_codegen_true_that_runs_the_interpreter_is_a_finding(monkeypatch):
    """#1213's rule, its other half: codegen=True builds compiled code or raises,
    so a run on the interpreter is a finding, not a machine without a compiler."""
    monkeypatch.setattr(bngsim.Simulator, "codegen_backend", property(lambda self: "exprtk"))
    base = D.Base(SHAPES["triangle-maxabs"])
    v = D.check_backend(base)
    assert v.status == "diff" and "codegen=True ran exprtk" in v.note, v
    # and cc_vs_interp does not pass the interpreter against itself
    assert D.check_cc_vs_interp(base).status == "skip"


@pytest.mark.parametrize(
    "text",
    [
        # Windows' compiler lookup, a missing cc, and macOS's shim without the CLT
        "No C compiler found. Install Visual Studio Build Tools (cl.exe) or MinGW (gcc).",
        "C compiler not found. Install Xcode Command Line Tools (macOS), gcc (Linux), ...",
        "Codegen compilation failed:\nxcrun: error: invalid active developer path "
        "(/Library/Developer/CommandLineTools), missing xcrun at: ...",
    ],
)
def test_a_machine_without_a_compiler_skips_the_compiled_paths(monkeypatch, text):
    base = D.Base(SHAPES["triangle-maxabs"])
    real = D.run

    def no_cc(job, **kw):
        if kw.get("codegen") is True:
            raise RuntimeError(
                f"codegen=True requested, but the codegen build failed (RuntimeError: {text})."
            )
        return real(job, **kw)

    monkeypatch.setattr(D, "run", no_cc)
    for check in (D.check_cc_vs_interp, D.check_backend):
        v = check(base)
        assert v.status == "skip" and "no C compiler" in v.note, v


def _chain(n=60) -> D.Job:
    """A60 -> ... in a line: 60 species and a sparse Jacobian, which the
    automatic rule sends to KLU."""
    params = "\n".join(f" {i + 1} k{i} {0.5 + 0.01 * i}" for i in range(n))
    species = "\n".join(f" {i + 1} A{i}() {10.0 if i == 0 else 0.0}" for i in range(n))
    reactions = "\n".join(f" {i + 1} {i + 1} {i + 2} k{i}" for i in range(n - 1))
    return D.Job("chain", _net("chain", params, species, "", reactions, f" 1 Aend {n}"))


@pytest.mark.skipif(not bngsim.HAS_KLU, reason="KLU not compiled")
def test_klu_is_compared_with_the_dense_solver_where_the_auto_rule_picks_klu(monkeypatch):
    """The base of a sparse model of 50 species or more runs on KLU: a reference
    left to the automatic rule is KLU against KLU."""
    base = D.Base(_chain())
    solvers = []
    real = D.run

    def spy(job, **kw):
        result, sim = real(job, **kw)
        solvers.append(result.solver_stats["linear_solver"])
        return result, sim

    monkeypatch.setattr(D, "run", spy)
    v = D.check_klu(base)
    assert v.status == "pass", v
    assert len(solvers) == 2 and solvers.count(1) == 1, solvers  # 1 is KLU


def test_a_tighter_run_that_fails_leaves_the_rules_that_compare_nothing(monkeypatch):
    """A base whose tenfold tighter run does not complete is no reference, and
    its comparisons are SKIP; the model still ran, and the rules that compare
    no answers (sparsity, backend) still apply."""

    def fails(job, x):
        raise bngsim.SimulationError("too much work")

    monkeypatch.setattr(D, "own_drift", fails)
    v = D.check_all(SHAPES["triangle-maxabs"], None, ("ajac_vs_fdjac", "sparsity_covers_jacobian"))
    assert v["ajac_vs_fdjac"].status == "skip" and "did not complete" in v["ajac_vs_fdjac"].note
    assert v["sparsity_covers_jacobian"].status == "pass", v


def test_a_nanomolar_state_is_stepped_on_its_own_scale():
    """A step floor of 1e-3 absolute moved a state of 2e-9 by half its size, and
    a saturating law's two steps then disagreed past 10%: the missing slot read
    as unresolvable."""

    class Nano(_FakeModel):
        @staticmethod
        def rhs(y, t=0.0):
            a, b = y
            return np.array([-a * b / (1e-10 + b), -a])

    y = [np.array([1e-9, 2e-9])]
    assert D.sparsity_misses(Nano(_csc([[1, 1], [1, 0]])), y) == []
    assert D.sparsity_misses(Nano(_csc([[1, 0], [1, 0]])), y) == [(0, 1)]


def test_a_trajectory_wholly_under_the_floor_is_not_a_pass():
    """A BioModel in molar units peaks under 1e3·atol: every path inside the
    floor read PASS, a path of all zeros included. It is SKIP, said why; a path
    that clears the floor is still judged."""
    base = D.Base(SHAPES["triangle-maxabs"])
    tiny = base.against(base.x * 1e-12)  # a peak of ~1e-11 under a floor of 1e-7
    v = tiny.compare(np.zeros_like(base.x))
    assert v.status == "skip" and "under the absolute floor" in v.note, v
    assert tiny.compare(tiny.x + 1.0).status == "diff"


def test_a_state_the_rule_cannot_reach_is_left_out_not_an_exception(monkeypatch):
    """The sparsity rule reads states after a one-interval run to the middle and
    the end of the window. Where that run fails and the base's grid does not
    (MODEL1112050001), the pattern is asked at the states that were reached."""
    base = D.Base(SHAPES["triangle-maxabs"])
    real = bngsim.Simulator.run

    def fails_on_one_interval(self, *a, **kw):
        if kw.get("n_points") == 2:
            raise bngsim.SimulationError("CVODE integration failed (CV_CONV_FAILURE)")
        return real(self, *a, **kw)

    monkeypatch.setattr(bngsim.Simulator, "run", fails_on_one_interval)
    v = D.check_sparsity(base)
    assert v.status == "pass" and "at 1 states" in v.note, v


def test_a_run_past_the_models_deadline_does_not_start(monkeypatch):
    import time

    monkeypatch.setattr(D, "_DEADLINE", time.monotonic() - 1.0)
    with pytest.raises(bngsim.SimulationTimeout):
        D.run(SHAPES["triangle-maxabs"])


def test_a_cap_spent_after_the_base_times_out_each_check_alone(monkeypatch):
    """The pool kills a model's worker at its cap, and every row with it: on
    the nightly MODEL1006230049, 0053 and 0077 take 716-751 s of 900, so a check
    added to every model turned all their rows TIMEOUT. Past the deadline each
    check is its own TIMEOUT, said why, and none is an exception or a pass."""
    import time

    real = D.Base

    def base_then_spent(job):
        b = real(job)
        D._DEADLINE = time.monotonic() - 1.0
        return b

    monkeypatch.setattr(D, "Base", base_then_spent)
    v = D.check_all(SHAPES["triangle-maxabs"], None, CHECKS, deadline=time.monotonic() + 1e3)
    assert {x.status for x in v.values()} == {"timeout"}, v
    assert v["grid"].status == "timeout", v["grid"]
    assert "not finished within the model's cap" in v["grid"].note
    assert v["sens_runs"].status == "timeout", v["sens_runs"]
    assert D._DEADLINE is None


def test_no_check_starts_past_the_deadline(monkeypatch):
    """backend_as_asked builds and compiles Simulators without running one, so no
    run's timeout bounds it: on MODEL1006230049 it started past a 180 s
    deadline and took the worker to 203 s, past a 200 s cap."""
    import time

    real = D.Base
    started = []

    def base_then_spent(job):
        b = real(job)
        D._DEADLINE = time.monotonic() - 1.0
        return b

    monkeypatch.setattr(D, "Base", base_then_spent)
    monkeypatch.setattr(D, "check_backend", lambda base: started.append(1) or D.Verdict("pass"))
    v = D.check_all(SHAPES["triangle-maxabs"], None, CHECKS, deadline=time.monotonic() + 1e3)
    assert not started and v["backend_as_asked"].status == "timeout", v["backend_as_asked"]


def test_a_base_past_the_deadline_is_a_timeout_on_every_row():
    import time

    v = D.check_all(SHAPES["triangle-maxabs"], None, CHECKS, deadline=time.monotonic() - 1.0)
    assert {x.status for x in v.values()} == {"timeout"}, v


def test_no_deadline_bounds_nothing():
    assert D._DEADLINE is None
    v = D.check_all(SHAPES["triangle-maxabs"], None, ("grid",))
    assert v["grid"].status == "pass", v


def test_the_runner_hands_each_model_its_cap(tmp_path):
    """The worker's deadline is the model's cap, short of it: with no time left
    every row is a TIMEOUT of its own, not a model the pool killed."""
    import queue

    import differential_run as RUN

    job = SHAPES["triangle-maxabs"]
    spec = {
        "key": job.model_id,
        "job": job.as_dict(),
        "checks": ["grid"],
        "locale": None,
        "max_species": 200,
        "timeout": 1e-6,
    }
    q = queue.Queue()
    RUN._worker(spec, q)
    res = q.get_nowait()
    assert res["status"] == "done", res
    assert res["rows"]["grid"]["status"] == "timeout", res


def test_a_model_killed_at_the_cap_keeps_the_rows_it_finished(tmp_path):
    """A check compiling C past the deadline has no run to bound, and the pool
    kills its worker at the cap: the verdicts it reached before stand, and only
    the rest are TIMEOUT."""
    import differential_run as RUN

    job = SHAPES["triangle-maxabs"]
    spec = {
        "key": job.model_id,
        "job": job.as_dict(),
        "checks": ["grid", "backend_as_asked"],
        "timeout": 900.0,
        "partial": str(tmp_path / "0.json"),
    }
    RUN._keep(spec, "grid", D.Verdict("pass", 0.0, "agrees"))
    rows = {r.method: r for r in RUN.to_results(spec, {"status": "timeout"}, {})}
    assert rows["grid"].outcome == "PASS", rows["grid"]
    assert rows["backend_as_asked"].outcome == "TIMEOUT"
    assert rows["backend_as_asked"].exception == "cap of 900 s"


@pytest.mark.parametrize(
    ("r", "note"),
    [
        ({"status": "dead", "exitcode": -11}, "the worker died (exit code -11)"),
        ({"status": "exception", "why": "ValueError: boom"}, "ValueError: boom"),
    ],
    ids=["died", "raised"],
)
def test_a_model_that_died_or_raised_keeps_the_rows_it_finished(tmp_path, r, note):
    """A worker that segfaults, or raises past check_all, has finished what it
    kept as surely as one killed at the cap. The rest say what happened to the
    worker: one that died did not reach the cap."""
    import differential_run as RUN

    job = SHAPES["triangle-maxabs"]
    spec = {
        "key": job.model_id,
        "job": job.as_dict(),
        "checks": ["grid", "backend_as_asked"],
        "timeout": 900.0,
        "partial": str(tmp_path / "0.json"),
    }
    RUN._keep(spec, "grid", D.Verdict("pass", 0.0, "agrees"))
    rows = {x.method: x for x in RUN.to_results(spec, r, {})}
    assert rows["grid"].outcome == "PASS", rows["grid"]
    assert rows["backend_as_asked"].outcome == "EXCEPTION"
    assert rows["backend_as_asked"].exception == note


def _locale_runs_out(monkeypatch, left):
    """``check_locale`` on a child that never finishes, with ``left`` seconds of
    the model's cap (None for no cap): the timeout the child was given, and the
    verdict."""
    import subprocess
    import time

    base = D.Base(SHAPES["triangle-maxabs"])
    given = []

    def never_finishes(cmd, timeout=None, **kw):
        given.append(timeout)
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(D.subprocess, "run", never_finishes)
    monkeypatch.setattr(D, "_DEADLINE", None if left is None else time.monotonic() + left)
    return given, D.check_locale(base, "de_DE.UTF-8", timeout=600.0)


def test_a_locale_run_past_the_models_cap_is_a_timeout(monkeypatch):
    given, v = _locale_runs_out(monkeypatch, left=30.0)
    assert given[0] <= 30.0, given
    assert v.status == "timeout" and "the model's cap" in v.note, v


@pytest.mark.parametrize("left", [None, 1e4], ids=["no-cap", "cap-to-spare"])
def test_a_locale_run_past_its_own_limit_is_not_the_caps(monkeypatch, left):
    """The run under C finished; the same run under a comma-decimal locale
    did not, in 600 s. That stays an EXCEPTION, as before the cap, and not a
    TIMEOUT the baseline reads as a changed state: a locale that stalls a run is
    the check's finding (#1235's class)."""
    given, v = _locale_runs_out(monkeypatch, left=left)
    assert given == [600.0], given
    assert v.status == "exception" and "TimeoutExpired" in v.note, v


def test_each_verdict_is_heard_as_it_is_reached():
    heard = {}
    v = D.check_all(SHAPES["triangle-maxabs"], None, CHECKS, on_verdict=heard.__setitem__)
    assert heard.keys() == v.keys()
    assert all(heard[c] is v[c] for c in v)


@pytest.mark.parametrize(
    "error",
    [
        # codegen=True's refusal: the compile's timeout in its text, no cause
        RuntimeError(
            "codegen=True requested, but the codegen build failed (RuntimeError: Codegen "
            "compilation timed out after 600 s (4.2 MB C source at -O1). The RHS, ..."
        ),
        # the sensitivity build's: chained to the compile's TimeoutExpired
        "chained",
    ],
    ids=["in-text", "chained"],
)
def test_a_compile_past_bngsims_limit_is_a_timeout_not_a_crash(error):
    """BIOMD0000000385 compiles 4.2 MB of C, past bngsim's 600 s limit: given
    the time to reach that compile, it read TIMEOUT -> EXCEPTION, a "new crash"
    the nightly alerts on. A timeout, whatever wraps it."""
    import subprocess

    if error == "chained":
        try:
            try:
                raise subprocess.TimeoutExpired(["cc"], 600)
            except subprocess.TimeoutExpired as inner:
                raise RuntimeError("Codegen compilation timed out after 600 s") from inner
        except RuntimeError as compile_error:
            try:
                raise RuntimeError("Failed to build the analytical sensitivity RHS") from (
                    compile_error
                )
            except RuntimeError as outer:
                error = outer
    v = D._verdict_of(error)
    assert v.status == "timeout", v
    assert D._verdict_of(RuntimeError("Codegen compilation failed: syntax")).status == "exception"


# ── The SSA arm (#1310): compiled propensities against interpreted ones ──

_DIMER = """begin parameters
    1 k 0.01
    2 kd 0.5
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1,1 2 k
    2 2 1,1 kd
end reactions
"""


def _ssa_job(n_rep=10):
    path = _DIR / "dimer_ssa.net"
    path.write_text(_DIMER)
    return D.Job("dimer", path, 0.0, 5.0, 11, method="ssa", n_rep=n_rep, seed=1)


def _compiled_here():
    r = bngsim.Simulator(D.load(_ssa_job().path), method="ssa").run(
        t_span=(0, 1), n_points=2, seed=1
    )
    return r.ssa_diagnostics["propensity_backend"] != "interpreted"


def test_an_ssa_job_runs_the_ssa_checks_and_an_ode_job_does_not():
    assert set(D.checks_for(_ssa_job())) == set(D.SSA_CHECKS)
    assert not set(D.checks_for(SHAPES["equality-dwell"])) & set(D.SSA_CHECKS)


def test_compiled_and_interpreted_mass_action_agree_draw_for_draw():
    if not _compiled_here():
        pytest.skip("no C compiler: the default SSA runs interpreted too")
    v = D.check_all(_ssa_job(), None)
    assert v["ssa_cc_vs_interp"].status == "pass" and "draw for draw" in v["ssa_cc_vs_interp"].note
    assert v["ssa_backend_as_asked"].status == "pass", v["ssa_backend_as_asked"]


def _perturbed_compiled(monkeypatch, factor):
    """One propensity perturbed, on the compiled side only: its rate constant
    k scaled by ``factor`` in the models the compiled runs load."""
    real_runs, real_load = D._ssa_runs, D.load
    on = []

    def load(path):
        model = real_load(path)
        if on:
            model.set_param("k", factor * float(model.get_param("k")))
        return model

    def runs(job, compiled, n=None):
        on[:] = [True] if compiled else []
        try:
            return real_runs(job, compiled, n)
        finally:
            on[:] = []

    monkeypatch.setattr(D, "load", load)
    monkeypatch.setattr(D, "_ssa_runs", runs)


def test_one_perturbed_propensity_is_a_finding(monkeypatch):
    """The issue's negative control: mass-action laws alone, so one draw off
    is a DIFF. 1% off one rate constant moves draws in every seed here (a
    change of 1e-9 moves none: it shifts each event time by as little, which
    crosses no sample)."""
    if not _compiled_here():
        pytest.skip("no C compiler: the default SSA runs interpreted too")
    _perturbed_compiled(monkeypatch, 1.01)
    v = D.check_all(_ssa_job(), None)["ssa_cc_vs_interp"]
    assert v.status == "diff" and "mass-action" in v.note, v


def test_functional_laws_are_judged_in_distribution(monkeypatch):
    """Where rounding in a functional law may change a selection, the seeds
    may differ and the ensembles must not: the same ensemble at other seeds
    passes, one with a propensity doubled does not."""
    if not _compiled_here():
        pytest.skip("no C compiler: the default SSA runs interpreted too")
    monkeypatch.setattr(D, "_all_elementary", lambda job: False)
    real = D._ssa_runs

    def other_seeds(job, compiled, n=None):
        if compiled:
            job = D.Job(**{**job.__dict__, "seed": job.seed + 1000})
        return real(job, compiled, n)

    monkeypatch.setattr(D, "_ssa_runs", other_seeds)
    same = D.check_all(_ssa_job(n_rep=30), None)["ssa_cc_vs_interp"]
    assert same.status == "pass" and "in distribution" in same.note, same
    monkeypatch.setattr(D, "_ssa_runs", real)
    _perturbed_compiled(monkeypatch, 3.0)
    moved = D.check_all(_ssa_job(n_rep=30), None)["ssa_cc_vs_interp"]
    assert moved.status == "diff" and "in distribution" in moved.note, moved


def test_codegen_false_that_runs_compiled_propensities_is_a_finding(monkeypatch):
    real = D._ssa_runs

    def lies(job, compiled, n=None):
        out, backends = real(job, compiled, n)
        return out, ({"cc"} if not compiled else backends)

    monkeypatch.setattr(D, "_ssa_runs", lies)
    v = D.check_all(_ssa_job(), None)["ssa_backend_as_asked"]
    assert v.status == "diff" and "codegen=False ran" in v.note, v


def test_a_model_without_a_propensity_library_runs_no_replicate(monkeypatch):
    """Most of the corpus is not eligible for compiled propensities, which the
    Simulator decides before any run: there is nothing to tell apart, and the
    corpus's slowest take one replicate past the nightly's cap."""
    import bngsim._codegen

    asked = []
    monkeypatch.setattr(bngsim._codegen, "prepare_ssa_propensity_lib", lambda model: None)
    monkeypatch.setattr(D, "_ssa_runs", lambda *a, **kw: asked.append(a) or ([], set()))
    v = D.check_all(_ssa_job(n_rep=10), None)
    assert asked == [], asked
    assert {c: x.status for c, x in v.items()} == dict.fromkeys(D.SSA_CHECKS, "skip"), v


def test_a_library_the_engine_declines_takes_one_replicate_each_way(monkeypatch):
    """Where the Simulator built a library and the engine still runs the
    interpreter (it declines events, functional laws and large networks),
    one replicate of each decides both checks, not an ensemble of each."""
    if not _compiled_here():
        pytest.skip("no C compiler: no propensity library is built")
    real, asked = D._ssa_runs, []

    def interpreted(job, compiled, n=None):
        asked.append((compiled, job.n_rep if n is None else n))
        out, _ = real(job, False, n)
        return out, {"interpreted"}

    monkeypatch.setattr(D, "_ssa_runs", interpreted)
    v = D.check_all(_ssa_job(n_rep=10), None)
    assert asked == [(True, 1), (False, 1)], asked
    assert {c: x.status for c, x in v.items()} == dict.fromkeys(D.SSA_CHECKS, "skip"), v


def test_an_ssa_job_with_no_replicates_is_skipped_not_judged(tmp_path):
    """Job's n_rep defaults to 0: two empty ensembles read PASS, "all 0 seeds
    identical", and no backend read DIFF, "codegen=False ran []". Nothing was
    compared, so both are SKIP, said why; the runner refuses --max-reps 0."""
    import differential_run as RUN

    v = D.check_all(_ssa_job(n_rep=0), None)
    assert {c: x.status for c, x in v.items()} == dict.fromkeys(D.SSA_CHECKS, "skip"), v
    assert all("no replicates" in x.note for x in v.values()), v
    # An empty corpus, so a runner that took --max-reps 0 returns at once.
    empty = ["--fd-sens-jobs", "", "--rr-jobs", "", "--ssa-jobs", "", "--no-locale"]
    with pytest.raises(SystemExit):
        RUN.main([*empty, "--max-reps", "0", "--out", str(tmp_path / "r.json")])


def test_the_runner_reads_rr_ssas_jobs_as_ssa_jobs(tmp_path):
    import differential_run as RUN

    (tmp_path / "m.net").write_text(_DIMER)
    manifest = tmp_path / "ssa_jobs.json"
    params = {"t_start": 0.0, "t_end": 5.0, "n_points": 1001, "seed": 7, "n_rep": 50}
    manifest.write_text(
        json.dumps({"jobs": [{"model_id": "m", "model": "m.net", "params": params}]})
    )
    (job,) = RUN.ssa_jobs(manifest, 0, 101, 20)
    assert (job.method, job.n_points, job.n_rep, job.seed) == ("ssa", 101, 20, 7)
