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

import functools
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

CHECKS = tuple(c for c in D.CHECKS if c != "locale")
_DIR = Path(tempfile.mkdtemp(prefix="bngsim-differential-shapes-"))


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


# ── State leaks (#1307): a model mutated between runs reads DIFF ──

# equality-dwell is a .net whose parameters all move it; event-at-a-sample an
# SBML model with an event, which restart crosses leg by leg.
LEAKED_ON = ("equality-dwell", "event-at-a-sample")


def _double_first_knob(model):
    knob = D.sens_params(model)[0]
    model.set_param(knob, 2.0 * float(model.get_param(knob)))


@pytest.mark.parametrize("shape", LEAKED_ON)
def test_a_model_changed_between_the_legs_of_a_restart_is_a_finding(monkeypatch, shape):
    real = bngsim.Simulator.run_until
    legs = []

    def leaky(self, t, **kw):
        if legs:
            _double_first_knob(self._model)
        legs.append(t)
        return real(self, t, **kw)

    monkeypatch.setattr(bngsim.Simulator, "run_until", leaky)
    v = D.check_restart(_base(shape))
    assert v.status == "diff" and "run_until" in v.note, v


def test_a_fresh_model_given_a_state_it_does_not_hold_is_a_finding(monkeypatch):
    """The second restart: a fresh model from the middle's saved state, which
    a model without events takes (equality-dwell has none)."""
    real = bngsim.Simulator.set_state

    def leaky(self, state, **kw):
        real(self, state, **kw)
        _double_first_knob(self._model)

    monkeypatch.setattr(bngsim.Simulator, "set_state", leaky)
    v = D.check_restart(_base("equality-dwell"))
    assert v.status == "diff" and "fresh model" in v.note, v


@pytest.mark.parametrize("shape", LEAKED_ON)
def test_a_restore_that_leaves_a_parameter_changed_is_a_finding(monkeypatch, shape):
    """Reuse rests on snapshot()/restore(): one that leaves the first run's
    mark reads as the leak it is."""
    real = bngsim.Simulator.restore

    def leaky(self, snapshot=None):
        real(self, snapshot)
        _double_first_knob(self._model)

    monkeypatch.setattr(bngsim.Simulator, "restore", leaky)
    v = D.check_reuse(_base(shape))
    assert v.status == "diff", v


@pytest.mark.parametrize("shape", LEAKED_ON)
def test_a_batch_point_run_on_another_parameter_is_a_finding(monkeypatch, shape):
    """One threaded point run on a parameter other than its own, as a clone
    that shared its parent's state would be."""
    real_batch, real_one = bngsim.Simulator.run_batch, bngsim.Simulator._run_single_batch
    threaded = []

    def batch(self, *a, num_processors=None, **kw):
        threaded.append(bool(num_processors and num_processors > 1))
        try:
            return real_batch(self, *a, num_processors=num_processors, **kw)
        finally:
            threaded.pop()

    def one(self, i, p, *a, **kw):
        if threaded and threaded[-1] and i == 1:
            p = {k: 2.0 * v for k, v in p.items()}
        return real_one(self, i, p, *a, **kw)

    monkeypatch.setattr(bngsim.Simulator, "run_batch", batch)
    monkeypatch.setattr(bngsim.Simulator, "_run_single_batch", one)
    v = D.check_batch(_base(shape))
    assert v.status == "diff" and "point 1" in v.note, v


def _failing_point(monkeypatch, only_threaded):
    """Point 1 of the batch raises, on threads only or always."""
    real_batch, real_one = bngsim.Simulator.run_batch, bngsim.Simulator._run_single_batch
    threaded = []

    def batch(self, *a, num_processors=None, **kw):
        threaded.append(bool(num_processors and num_processors > 1))
        try:
            return real_batch(self, *a, num_processors=num_processors, **kw)
        finally:
            threaded.pop()

    def one(self, i, p, *a, **kw):
        if i == 1 and (threaded[-1] or not only_threaded):
            raise bngsim.SimulationError(f"Batch simulation {i} failed: CVODE (planted)")
        return real_one(self, i, p, *a, **kw)

    monkeypatch.setattr(bngsim.Simulator, "run_batch", batch)
    monkeypatch.setattr(bngsim.Simulator, "_run_single_batch", one)


def test_a_sweep_point_that_does_not_run_alone_is_not_the_batchs_to_judge(monkeypatch):
    """BIOMD0000000763's: a point 10% off its parameter fails to integrate one
    at a time as on threads, which is the model, not a leak."""
    _failing_point(monkeypatch, only_threaded=False)
    v = D.check_batch(_base("equality-dwell"))
    assert v.status == "skip" and "one at a time" in v.note, v


def test_a_point_that_fails_only_on_threads_is_a_finding(monkeypatch):
    _failing_point(monkeypatch, only_threaded=True)
    v = D.check_batch(_base("equality-dwell"))
    assert v.status == "exception", v
