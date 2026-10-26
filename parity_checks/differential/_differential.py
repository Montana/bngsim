"""Paths that should give the same answer, run and compared (issue #1257).

A recurring kind of bngsim bug is two code paths that should agree and do not:
a plain run and a sensitivity run recording an event's row on different sides
of it (#1182), the same ``.net`` file under ``LC_NUMERIC=C`` and ``de_DE``
(#1235), KLU and the dense solver on an equality the state sits on (#1254).
Each was found by a one-off probe. This runs those probes for any model.

One model, one plain run (the *base*), and each check against it:

=====================  ========================================  =====================
check                  path                                      reference
=====================  ========================================  =====================
``sens_vs_plain``      the state of a sensitivity run            the base
``ajac_vs_fdjac``      ``jacobian="fd"``                         the base (``"auto"``)
``klu_vs_dense``       ``force_sparse_linear_solver=True``       the base (dense)
``cc_vs_interp``       ``codegen=True``                          ``codegen=False``
``cc_vs_mir``          ``BNGSIM_CODEGEN_JIT=mir``                ``codegen=True`` (cc)
``locale``             the run under a comma-decimal locale      the base, under ``C``
=====================  ========================================  =====================

and three rules for bugs whose answers are right, which no comparison of
answers sees:

* ``sens_runs``: the sensitivity run is refused while the plain run completes
  (#1229 refused a bend for a jump that did not exist). A refusal is
  UNSUPPORTED, not a finding: many are deliberate, by name. The nightly's
  baseline is what makes one a finding, when a model that ran is refused.
* ``sparsity_covers_jacobian``: every entry central differences of the
  right-hand side find nonzero has a slot in ``jacobian_sparsity`` (#1239: KLU
  factored a pattern that missed entries, and still agreed within tolerance).
* ``backend_as_asked``: ``codegen=False`` runs the interpreter, and
  ``codegen=True`` compiled code (#1213: ``codegen=False`` was ignored on MIR
  builds; same answers either way).

Each returns a :class:`Verdict`: ``pass``, ``diff`` (a finding, with its size),
``unsupported`` (the path declined by type: not a wrong answer), ``skip`` (the
path does not apply here, said why), ``exception`` (a path raised where the base
ran: a finding) or ``no_base`` (the model does not load or its plain run does
not complete, so there is nothing to compare against: the parity suites own
that).

The oracle has to be careful
----------------------------
Every false positive three rounds of triage found came from the checking, not
from bngsim. So a disagreement between two paths counts only past how far the
reference moves by itself at a tenfold tighter tolerance (:func:`own_drift`):
a model that amplifies every difference, a fast oscillator, makes two correct
paths disagree by that much, and so does one path against a tighter self
(``LV``: 3e-6 apart at rtol 1e-12, 76% at 1e-8). A base that moves by more
than :data:`UNJUDGEABLE_DRIFT` that way is no reference at all, and its paths
are not judged (SKIP, said why). And a cell is judged on the
trajectory's own scale, with the run's ``atol`` as its floor, so a state of 1e6
is not held to an absolute 1e-3, nor one of 1e-9 let off by it.

Sensitivities against central differences are ``fd_sens``'s, not this module's:
one oracle with one set of rules (``parity_checks/fd_sens``).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: How many times the run's rtol two correct paths may differ by, before the
#: reference's own drift is counted: their errors are each about rtol, scaled by
#: the problem's conditioning, which a tolerance does not bound.
PATH_RTOL_FACTOR = 1e3
#: And how far past the reference's own drift at a tenfold tighter tolerance.
DRIFT_FACTOR = 10.0
#: A base that moves by more than this at a tenfold tighter tolerance is not
#: converged at its own: no path is judged against it. MODEL2310250001 drives a
#: species through a pole of its uptake law; its plain run moves 2.6% at a
#: tighter tolerance, libRoadRunner's -1.4 to -4.6 across three, and its
#: sensitivity run 359 times the trajectory's peak.
UNJUDGEABLE_DRIFT = 1e-3
#: The parameters a sensitivity run takes, at most.
MAX_SENS_PARAMS = 3
#: Comma-decimal locales, in the order tried.
COMMA_LOCALES = ("de_DE.UTF-8", "de_DE.utf8", "de_DE", "fr_FR.UTF-8", "fr_FR.utf8", "fr_FR")

CHECKS = (
    "sens_runs",
    "sens_vs_plain",
    "ajac_vs_fdjac",
    "klu_vs_dense",
    "cc_vs_interp",
    "cc_vs_mir",
    "locale",
    "sparsity_covers_jacobian",
    "backend_as_asked",
)


@dataclass
class Verdict:
    status: str  # pass | diff | unsupported | skip | exception | no_base
    value: float | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Job:
    """One model and how to run it: its window, grid and tolerances, and any
    Simulator options every run of it takes (``sim``)."""

    model_id: str
    path: Path
    t_start: float = 0.0
    t_end: float = 10.0
    n_points: int = 21
    rtol: float = 1e-8
    atol: float = 1e-10
    max_steps: int = 100_000
    sim: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "t_start": self.t_start,
            "t_end": self.t_end,
            "n_points": self.n_points,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
            "sim": self.sim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        return cls(**{**d, "path": Path(d["path"])})


def load(path: Path):
    import bngsim

    path = Path(path)
    if path.suffix == ".net":
        return bngsim.Model.from_net(str(path))
    return bngsim.Model.from_sbml(str(path))


def sens_params(model) -> list[str]:
    """The first few of the model's knobs, less those a sensitivity is refused for
    by construction (compartment sizes, frozen parameters): fd_sens's choice."""
    excluded = set(getattr(model, "compartment_size_params", None) or ())
    excluded |= set(getattr(model, "frozen_params", None) or ())
    return [p for p in model.primary_param_names if p not in excluded][:MAX_SENS_PARAMS]


def run(job: Job, **kw):
    """A run of a fresh model (a run leaves its state on the Model it was given):
    the Result and the Simulator."""
    import bngsim

    sim = bngsim.Simulator(load(job.path), method="ode", **{**job.sim, **kw})
    result = sim.run(
        t_span=(job.t_start, job.t_end),
        n_points=job.n_points,
        rtol=job.rtol,
        atol=job.atol,
        max_steps=job.max_steps,
    )
    return result, sim


def own_drift(job: Job, base: np.ndarray) -> float:
    """How far the base moves at a tenfold tighter tolerance, on its own scale."""
    tight = Job(
        **{**job.__dict__, "rtol": job.rtol / 10.0, "atol": job.atol / 10.0, "sim": job.sim}
    )
    x, _ = run(tight)
    return gap(np.asarray(x.species, dtype=float), base, job.atol)


def gap(x: np.ndarray, ref: np.ndarray, atol: float) -> float:
    """``max |x - ref|`` over the reference's peak, past an absolute floor of
    ``atol``: 0 where every cell is within atol. A shape or finiteness mismatch
    is ``inf``."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    if x.shape != ref.shape or not np.array_equal(np.isfinite(x), np.isfinite(ref)):
        return float("inf")
    finite = np.isfinite(ref)
    if not finite.any():
        return 0.0
    over = np.abs(x - ref)[finite] - atol
    peak = float(np.abs(ref[finite]).max())
    return float(max(over.max(), 0.0) / max(peak, atol))


class Base:
    """The base run, and the bound two paths may differ by, computed once."""

    def __init__(self, job: Job):
        result, sim = run(job)
        self.job = job
        self.x = np.asarray(result.species, dtype=float)
        self.backend = sim.codegen_backend
        self.drift = own_drift(job, self.x)
        self.limit = max(PATH_RTOL_FACTOR * job.rtol, DRIFT_FACTOR * self.drift)

    def compare(self, x, label: str = "") -> Verdict:
        g = gap(x, self.x, PATH_RTOL_FACTOR * self.job.atol)
        note = f"{g:.2e} of the trajectory's peak (bound {self.limit:.1e}, own drift {self.drift:.1e})"
        if not self.drift <= UNJUDGEABLE_DRIFT:
            return Verdict(
                "skip",
                g,
                f"{label}no reference: the base itself moves by {self.drift:.1e} at a tenfold tighter tolerance; {note}",
            )
        return Verdict("pass" if g <= self.limit else "diff", g, f"{label}{note}")


def _declined(e: Exception) -> bool:
    """A refusal by type, or a sensitivity run's refusal by name at run time: a
    SimulationError whose message says forward sensitivity is what it declines
    (an event not smooth in its parameter, a trigger that starts on its
    threshold, a delay; #1229's class). A crash says no such thing."""
    import bngsim

    if isinstance(e, bngsim.SensitivityUnsupportedError):
        return True
    text = str(e).lower()
    return isinstance(e, bngsim.SimulationError) and (
        "forward sensitivity" in text or "forward-sensitivity run" in text
    )


def _verdict_of(e: Exception) -> Verdict:
    text = str(e).splitlines()[0][:300] if str(e) else ""
    status = "unsupported" if _declined(e) or isinstance(e, _Declined) else "exception"
    return Verdict(status, None, f"{type(e).__name__}: {text}")


def _guard(fn) -> Verdict:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - a path's failure is that check's verdict
        return _verdict_of(e)


def check_sens(base: Base) -> dict[str, Verdict]:
    """``sens_runs`` and ``sens_vs_plain``: a sensitivity run's state is the plain
    run's. Its sensitivity RHS (analytic or CVODES' difference quotient) is
    recorded, the #1214 class: a slow fallback with right answers."""
    params = sens_params(load(base.job.path))
    if not params:
        skip = Verdict("skip", note="no parameter to differentiate")
        return {"sens_runs": skip, "sens_vs_plain": skip}
    try:
        result, sim = run(base.job, sensitivity_params=params)
    except Exception as e:  # noqa: BLE001 - that it did not complete is the verdict
        # Where the plain run completes and the sensitivity run does not: a
        # refusal by name, or a derivative that does not exist at the start
        # (a power of a species at 0), or a crash. The rule records it; the
        # baseline makes a new one a finding. Telling them apart by the text
        # of the message would be guesswork.
        skip = Verdict("skip", note="the sensitivity run did not complete")
        why = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
        return {"sens_runs": Verdict("unsupported", None, why), "sens_vs_plain": skip}
    rhs = "analytic" if sim.has_analytic_sens_rhs else "difference_quotient"
    extra = {"sens_rhs": rhs, "params": params}
    if not sim.has_analytic_sens_rhs:
        extra["sens_rhs_decline_reason"] = str(sim.sens_rhs_decline_reason)[:300]
    v = base.compare(result.species)
    v.extra = extra
    return {"sens_runs": Verdict("pass", note=rhs, extra=extra), "sens_vs_plain": v}


def check_ajac(base: Base) -> Verdict:
    if base.job.sim.get("jacobian") == "fd":
        return Verdict("skip", note='the model runs on jacobian="fd"')
    return _guard(lambda: base.compare(run(base.job, jacobian="fd")[0].species))


def _sparse_declined(e: Exception) -> bool:
    # bngsim's own refusal of the option, by name: a model with no sparse
    # Jacobian to hand KLU under the asked strategy (a SimulationError, not a
    # type of its own).
    return "force_sparse_linear_solver" in str(e) and "cannot supply a sparse Jacobian" in str(e)


def check_klu(base: Base) -> Verdict:
    """KLU against the dense solver on the same Jacobian, so the one thing that
    differs is the linear solver and the Jacobian it is handed (KLU's FD
    Jacobian is colored over ``jacobian_sparsity``; the dense one differences
    every column). The FD Jacobian first, where #1254 was found; the analytic
    one where bngsim declines KLU with the FD one."""
    import bngsim

    if not bngsim._bngsim_core.HAS_KLU:
        return Verdict("skip", note="a build without KLU")

    def go():
        for jac in ("fd", "auto"):
            try:
                x, _ = run(base.job, force_sparse_linear_solver=True, jacobian=jac)
            except bngsim.SimulationError as e:
                if _sparse_declined(e):
                    continue
                raise
            ref = base
            if jac != "auto":
                dense, _ = run(base.job, jacobian=jac)
                ref = Base.__new__(Base)
                ref.__dict__.update(base.__dict__, x=np.asarray(dense.species, dtype=float))
            return ref.compare(x.species, f'jacobian="{jac}", KLU against dense: ')
        return Verdict("unsupported", None, "bngsim declines KLU on either Jacobian, by name")

    return _guard(go)


def _compiled(job: Job):
    try:
        result, sim = run(job, codegen=True)
    except RuntimeError as e:
        if "codegen declined this model" in str(e):
            raise _Declined(str(e).splitlines()[0][:300]) from e
        raise
    if sim.codegen_backend == "exprtk":
        raise _NoCompiler("codegen=True ran the interpreter: no codegen backend")
    return result, sim


class _NoCompiler(RuntimeError):
    pass


class _Declined(RuntimeError):
    """Codegen's refusal of a model, by name ("codegen declined this model:
    ..."), as against a build that failed."""


def check_cc_vs_interp(base: Base) -> Verdict:
    def go():
        interp, _ = run(base.job, codegen=False)
        try:
            compiled, sim = _compiled(base.job)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e))
        ref = Base.__new__(Base)
        ref.__dict__.update(base.__dict__, x=np.asarray(interp.species, dtype=float))
        return ref.compare(compiled.species, f"{sim.codegen_backend} against exprtk: ")

    return _guard(go)


@contextlib.contextmanager
def _env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def check_cc_vs_mir(base: Base) -> Verdict:
    import bngsim

    if not bngsim.HAS_MIR:
        return Verdict("skip", note="a build without MIR")

    def go():
        try:
            cc, _ = _compiled(base.job)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e))
        with _env("BNGSIM_CODEGEN_JIT", "mir"):
            mir, sim = run(base.job, codegen=True)
        if sim.codegen_backend != "mir":
            return Verdict("diff", None, f"BNGSIM_CODEGEN_JIT=mir ran {sim.codegen_backend}")
        ref = Base.__new__(Base)
        ref.__dict__.update(base.__dict__, x=np.asarray(cc.species, dtype=float))
        return ref.compare(mir.species, "mir against cc: ")

    return _guard(go)


def check_backend(base: Base) -> Verdict:
    """codegen=False runs the interpreter; codegen=True compiled code."""

    def go():
        _, interp = run(base.job, codegen=False)
        asked = {"codegen=False": interp.codegen_backend}
        if interp.codegen_backend != "exprtk":
            return Verdict("diff", None, f"codegen=False ran {interp.codegen_backend}", asked)
        try:
            _, sim = _compiled(base.job)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e), extra=asked)
        asked["codegen=True"] = sim.codegen_backend
        return Verdict("pass", note=", ".join(f"{k}: {v}" for k, v in asked.items()), extra=asked)

    return _guard(go)


def sparsity_misses(model, states, t=0.0, rel=1e-6, floor=1e-8) -> list[tuple[int, int]]:
    """The (row, column) entries central differences of the right-hand side find
    nonzero at any of ``states``, which ``jacobian_sparsity`` has no slot for.

    An entry is nonzero when, at both steps ``h`` and ``h/2`` (relative to the
    state's own size, or to ``rel`` of its largest where it is 0), its
    contribution ``|J_ij|·scale_j`` clears ``floor`` of the largest one, and the
    two steps agree to 10%: an entry the differences cannot resolve is not
    asked of the pattern.
    """
    pat = model._core.jacobian_sparsity
    n = int(pat["n"])
    if n == 0:
        return []
    cols, rows = np.asarray(pat["col_ptrs"]), np.asarray(pat["row_indices"])
    slots = np.zeros((n, n), dtype=bool)
    for j in range(n):
        slots[rows[cols[j] : cols[j + 1]], j] = True
    missed: set[tuple[int, int]] = set()
    for y in states:
        y = np.asarray(y, dtype=float)
        big = float(np.abs(y).max()) if y.size else 0.0
        scale = np.maximum(np.abs(y), max(big, 1.0) * 1e-3)
        jac = []
        for h_rel in (rel, rel / 2.0):
            cols_ = []
            for j in range(n):
                h = h_rel * scale[j]
                up, dn = y.copy(), y.copy()
                up[j] += h
                dn[j] -= h
                cols_.append(
                    (np.asarray(model.rhs(up, t)) - np.asarray(model.rhs(dn, t))) / (2 * h)
                )
            jac.append(np.stack(cols_, axis=1) * scale[np.newaxis, :])
        c1, c2 = np.abs(jac[0]), np.abs(jac[1])
        top = max(float(c1.max()), float(c2.max()))
        if top == 0.0 or not np.isfinite(top):
            continue
        sure = (c1 > floor * top) & (c2 > floor * top) & (np.abs(c1 - c2) <= 0.1 * c2)
        missed |= {(int(i), int(j)) for i, j in zip(*np.nonzero(sure & ~slots), strict=True)}
    return sorted(missed)


def _states(job: Job) -> list[tuple[float, np.ndarray]]:
    """The engine's whole state, at the start, the middle and the end of the
    window: ``Model.get_state()`` after a run to each, which holds what the
    right-hand side reads, a parameter an event promotes included (a Result's
    species do not)."""
    import bngsim

    out = []
    for t in (job.t_start, 0.5 * (job.t_start + job.t_end), job.t_end):
        model = load(job.path)
        if t > job.t_start:
            bngsim.Simulator(model, method="ode", **job.sim).run(
                t_span=(job.t_start, t), n_points=2, rtol=job.rtol, atol=job.atol
            )
        y = np.asarray(model.get_state(), dtype=float)
        if np.all(np.isfinite(y)):
            out.append((t, y))
    return out


def check_sparsity(base: Base) -> Verdict:
    def go():
        model = load(base.job.path)
        states = _states(base.job)
        misses = sorted({m for t, y in states for m in sparsity_misses(model, [y], t=t)})
        if misses:
            names = list(model.species_names)
            shown = ", ".join(f"d{names[i]}/d{names[j]}" for i, j in misses[:5])
            return Verdict(
                "diff",
                float(len(misses)),
                f"{len(misses)} entries the differences find nonzero have no slot: {shown}",
                {"missed": misses[:50]},
            )
        return Verdict("pass", 0.0, f"pattern covers every entry at {len(states)} states")

    return _guard(go)


def comma_locale() -> str | None:
    """A comma-decimal LC_NUMERIC this machine has, or None. Asked of a child
    process, so this one's locale is never set."""
    probe = (
        "import locale, sys\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        locale.setlocale(locale.LC_NUMERIC, name)\n"
        "    except locale.Error:\n"
        "        continue\n"
        "    if locale.localeconv()['decimal_point'] == ',':\n"
        "        print(name); break\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe, *COMMA_LOCALES], capture_output=True, text=True, timeout=60
    )
    return out.stdout.strip() or None


_LOCALE_PROBE = """
import json, locale, logging, sys, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
locale.setlocale(locale.LC_NUMERIC, sys.argv[1])
sys.path.insert(0, sys.argv[3])
import numpy as np
from _differential import Job, run
r, _ = run(Job.from_dict(json.loads(sys.argv[2])))
print(json.dumps(np.asarray(r.species, dtype=float).tolist()))
"""


def check_locale(base: Base, name: str | None, timeout: float = 600.0) -> Verdict:
    """The run in a child process under the comma-decimal locale ``name``."""
    if name is None:
        return Verdict("skip", note="no comma-decimal locale on this machine")

    def go():
        here = str(Path(__file__).resolve().parent)
        out = subprocess.run(
            [sys.executable, "-c", _LOCALE_PROBE, name, json.dumps(base.job.as_dict()), here],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if out.returncode != 0:
            last = (out.stderr.strip().splitlines() or ["?"])[-1][:300]
            return Verdict("diff", None, f"runs under C, fails under {name}: {last}")
        return base.compare(json.loads(out.stdout.strip().splitlines()[-1]), f"{name}: ")

    return _guard(go)


def check_all(job: Job, locale_name: str | None, checks=CHECKS) -> dict[str, Verdict]:
    """Every check of ``checks`` on one model. A base that does not run is no
    reference: each check is then an exception on the base."""
    try:
        base = Base(job)
    except Exception as e:  # noqa: BLE001
        # No base, no reference: not a finding of this suite's (the parity suites
        # own a model that does not load or run), and not a pass either.
        v = Verdict("no_base", None, f"the base run: {type(e).__name__}: {str(e)[:300]}")
        return {c: v for c in checks}
    out: dict[str, Verdict] = {}
    if {"sens_runs", "sens_vs_plain"} & set(checks):
        out.update(check_sens(base))
    one = {
        "ajac_vs_fdjac": check_ajac,
        "klu_vs_dense": check_klu,
        "cc_vs_interp": check_cc_vs_interp,
        "cc_vs_mir": check_cc_vs_mir,
        "sparsity_covers_jacobian": check_sparsity,
        "backend_as_asked": check_backend,
    }
    for name, fn in one.items():
        if name in checks:
            out[name] = fn(base)
    if "locale" in checks:
        out["locale"] = check_locale(base, locale_name)
    return {c: out[c] for c in checks if c in out}
