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
``grid``               the run at 8x the output points, read     the base
                       at the base's own sample times
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
trajectory's own scale, past an absolute floor of the run's ``atol`` scaled as
``rtol`` is (:data:`PATH_RTOL_FACTOR`), so a state of 1e6 is not held to an
absolute 1e-3. A trajectory wholly under that floor (a BioModel in molar units,
1e-6 at its atol of 1e-9) shows no difference inside it, so a path that stays
inside is SKIP, said why, not PASS; one that clears it is still judged.

Sensitivities against central differences are ``fd_sens``'s, not this module's:
one oracle with one set of rules (``parity_checks/fd_sens``).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
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
#: The grid check's output points, as a multiple of the base's intervals.
GRID_FACTOR = 8
#: Comma-decimal locales, in the order tried.
COMMA_LOCALES = ("de_DE.UTF-8", "de_DE.utf8", "de_DE", "fr_FR.UTF-8", "fr_FR.utf8", "fr_FR")

#: The ``time.monotonic()`` every run of the model being checked must finish
#: by, which :func:`check_all` sets: None outside it, or with no cap.
_DEADLINE: float | None = None

CHECKS = (
    "sens_runs",
    "sens_vs_plain",
    "ajac_vs_fdjac",
    "klu_vs_dense",
    "cc_vs_interp",
    "cc_vs_mir",
    "locale",
    "grid",
    "sparsity_covers_jacobian",
    "backend_as_asked",
    # State leaks (#1307): a restarted run, a reused Model or Simulator, a batch.
    "restart",
    "reuse",
    "batch",
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


def _left() -> float | None:
    """The seconds to :data:`_DEADLINE`, a run's ``timeout``: None for none. Past
    it no run starts (``SimulationTimeout``)."""
    if _DEADLINE is None:
        return None
    left = _DEADLINE - time.monotonic()
    if left <= 0:
        import bngsim

        raise bngsim.SimulationTimeout("the model's cap is spent")
    return left


def run(job: Job, **kw):
    """A run of a fresh model (a run leaves its state on the Model it was given):
    the Result and the Simulator. It has until :data:`_DEADLINE`."""
    import bngsim

    sim = bngsim.Simulator(load(job.path), method="ode", **{**job.sim, **kw})
    result = sim.run(
        t_span=(job.t_start, job.t_end),
        n_points=job.n_points,
        rtol=job.rtol,
        atol=job.atol,
        max_steps=job.max_steps,
        timeout=_left(),
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
        # The columns of x, by name: a Result's species, which leave out a
        # parameter an event promotes into the state (Model.species_names has it).
        self.species_names = list(result.species_names)
        self.backend = sim.codegen_backend
        # A tenfold tighter run that does not complete leaves the base no
        # reference, not the model unrun: the checks that compare no answers
        # (sparsity, backend, a sensitivity run's refusal) still apply.
        self.drift_error = ""
        try:
            self.drift = own_drift(job, self.x)
        except Exception as e:  # noqa: BLE001 - recorded on every comparison's note
            self.drift = float("inf")
            self.drift_error = f"{type(e).__name__}: {_first_line(e)}"
        self.limit = max(PATH_RTOL_FACTOR * job.rtol, DRIFT_FACTOR * self.drift)

    def against(self, x) -> Base:
        """This base's bound and drift, with ``x`` as the reference: another path
        of the same model (the dense solver, the interpreter, cc)."""
        ref = Base.__new__(Base)
        ref.__dict__.update(self.__dict__, x=np.asarray(x, dtype=float))
        return ref

    def compare(self, x, label: str = "") -> Verdict:
        floor = PATH_RTOL_FACTOR * self.job.atol
        g = gap(x, self.x, floor)
        note = f"{g:.2e} of the trajectory's peak (bound {self.limit:.1e}, own drift {self.drift:.1e})"
        if self.drift_error:
            return Verdict(
                "skip",
                g,
                f"{label}no reference: the base at a tenfold tighter tolerance did not "
                f"complete ({self.drift_error}); {note}",
            )
        if not self.drift <= UNJUDGEABLE_DRIFT:
            return Verdict(
                "skip",
                g,
                f"{label}no reference: the base itself moves by {self.drift:.1e} at a tenfold tighter tolerance; {note}",
            )
        finite = np.abs(self.x[np.isfinite(self.x)])
        peak = float(finite.max()) if finite.size else 0.0
        if g == 0.0 and not peak > floor:
            return Verdict(
                "skip",
                g,
                f"{label}nothing could count: the trajectory's peak {peak:.1e} is under the "
                f"absolute floor {floor:.1e} ({PATH_RTOL_FACTOR:g}·atol); {note}",
            )
        return Verdict("pass" if g <= self.limit else "diff", g, f"{label}{note}")


def _first_line(e: BaseException) -> str:
    return str(e).splitlines()[0][:300] if str(e) else ""


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


def _timed_out(e: BaseException) -> bool:
    import bngsim

    return isinstance(e, bngsim.SimulationTimeout | subprocess.TimeoutExpired)


def _verdict_of(e: Exception) -> Verdict:
    if _timed_out(e):
        # The model's cap, reached in this check: its own row, not every row.
        return Verdict("timeout", None, f"not finished within the model's cap: {_first_line(e)}")
    status = "unsupported" if _declined(e) or isinstance(e, _Declined) else "exception"
    return Verdict(status, None, f"{type(e).__name__}: {_first_line(e)}")


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
        if _timed_out(e):
            v = _verdict_of(e)
            return {"sens_runs": v, "sens_vs_plain": v}
        # Where the plain run completes and the sensitivity run does not: a
        # refusal by name, or a derivative that does not exist at the start
        # (a power of a species at 0), or a crash. The rule records it; the
        # baseline makes a new one a finding. Telling them apart by the text
        # of the message would be guesswork.
        skip = Verdict("skip", note="the sensitivity run did not complete")
        why = f"{type(e).__name__}: {_first_line(e)}"
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
    one where bngsim declines KLU with the FD one.

    The reference is forced dense, not left to the automatic rule: that rule
    sends a model of 50 species or more with a sparse Jacobian to KLU, and the
    base with it, so an unforced reference there is KLU against KLU."""
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
            dense, _ = run(base.job, force_dense_linear_solver=True, jacobian=jac)
            return base.against(dense.species).compare(
                x.species, f'jacobian="{jac}", KLU against dense: '
            )
        return Verdict("unsupported", None, "bngsim declines KLU on either Jacobian, by name")

    return _guard(go)


#: How a machine without a C compiler shows through bngsim's build: Windows'
#: compiler lookup, a ``cc`` that is not there (FileNotFoundError), and macOS's
#: ``cc`` shim without the Command Line Tools.
_NO_COMPILER = ("No C compiler found", "C compiler not found", "invalid active developer path")


def _compiled(job: Job):
    """A ``codegen=True`` run: the Result and the Simulator. Which backend ran is
    the caller's to judge: ``codegen=True`` builds compiled code or raises, so
    one that ran the interpreter is ``backend_as_asked``'s finding, not a
    machine without a compiler."""
    try:
        return run(job, codegen=True)
    except RuntimeError as e:
        if "codegen declined this model" in str(e):
            raise _Declined(_first_line(e)) from e
        if any(s in str(e) for s in _NO_COMPILER):  # a machine without cc: no cc path here
            raise _NoCompiler("codegen=True found no C compiler") from e
        raise


def _paths(base: Base, key: str, fn):
    """``fn()`` once per base: ``cc_vs_interp`` and ``backend_as_asked`` ask for
    the same two runs. A path that raised raises again."""
    memo = base.__dict__.setdefault("_paths", {})
    if key not in memo:
        try:
            memo[key] = (fn(), None)
        except Exception as e:  # noqa: BLE001 - re-raised to each caller
            memo[key] = (None, e)
    value, err = memo[key]
    if err is not None:
        raise err
    return value


def _interpreted(base: Base):
    return _paths(base, "codegen=False", lambda: run(base.job, codegen=False))


def _compiled_of(base: Base):
    return _paths(base, "codegen=True", lambda: _compiled(base.job))


class _NoCompiler(RuntimeError):
    pass


class _Declined(RuntimeError):
    """Codegen's refusal of a model, by name ("codegen declined this model:
    ..."), as against a build that failed."""


def check_cc_vs_interp(base: Base) -> Verdict:
    def go():
        interp, _ = _interpreted(base)
        try:
            compiled, sim = _compiled_of(base)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e))
        if sim.codegen_backend == "exprtk":
            # backend_as_asked's finding; here, the interpreter against itself
            return Verdict("skip", note="codegen=True ran exprtk: no compiled path to compare")
        return base.against(interp.species).compare(
            compiled.species, f"{sim.codegen_backend} against exprtk: "
        )

    return _guard(go)


@contextlib.contextmanager
def _env(name: str, value: str | None):
    """``name`` set to ``value`` (unset for None) for the block."""
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
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
        # cc with the selector cleared: a run under BNGSIM_CODEGEN_JIT=mir (the
        # MIR leg sets it for its whole suite) would compare MIR with itself.
        try:
            with _env("BNGSIM_CODEGEN_JIT", None):
                cc, cc_sim = _compiled(base.job)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e))
        if cc_sim.codegen_backend != "cc":
            return Verdict(
                "diff",
                None,
                f"codegen=True without BNGSIM_CODEGEN_JIT ran {cc_sim.codegen_backend}",
            )
        with _env("BNGSIM_CODEGEN_JIT", "mir"):
            mir, sim = run(base.job, codegen=True)
        if sim.codegen_backend != "mir":
            return Verdict("diff", None, f"BNGSIM_CODEGEN_JIT=mir ran {sim.codegen_backend}")
        return base.against(cc.species).compare(mir.species, "mir against cc: ")

    return _guard(go)


def check_grid(base: Base) -> Verdict:
    """The answer must not depend on where it is read: a run asked for 8x the
    output points, read at the base's own sample times, is the base. #1244 was a
    pulse train stepped over at one grid and caught at another: each output
    point is a stop the integrator lands on, and a schedule's stops moved with
    them."""

    def go():
        fine = Job(**{**base.job.__dict__, "n_points": (base.job.n_points - 1) * GRID_FACTOR + 1})
        x, _ = run(fine)
        return base.compare(
            np.asarray(x.species, dtype=float)[::GRID_FACTOR], f"{GRID_FACTOR}x the points: "
        )

    return _guard(go)


def check_backend(base: Base) -> Verdict:
    """codegen=False runs the interpreter; codegen=True compiled code."""

    def go():
        _, interp = _interpreted(base)
        asked = {"codegen=False": interp.codegen_backend}
        if interp.codegen_backend != "exprtk":
            return Verdict("diff", None, f"codegen=False ran {interp.codegen_backend}", asked)
        try:
            _, sim = _compiled_of(base)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e), extra=asked)
        asked["codegen=True"] = sim.codegen_backend
        if sim.codegen_backend == "exprtk":
            return Verdict("diff", None, "codegen=True ran exprtk", asked)
        return Verdict("pass", note=", ".join(f"{k}: {v}" for k, v in asked.items()), extra=asked)

    return _guard(go)


def sparsity_misses(model, states, t=0.0, rel=1e-6, floor=1e-8) -> list[tuple[int, int]]:
    """The (row, column) entries central differences of the right-hand side find
    nonzero at any of ``states``, which ``jacobian_sparsity`` has no slot for.

    An entry is nonzero when, at both steps ``h`` and ``h/2`` (``rel`` of the
    state's own size, or of 1e-3 of the largest state where it is smaller), its
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
        # On the state's own scale: a floor of 1e-3 absolute stepped a nanomolar
        # state by its whole size, past what a nonlinear law's two steps agree on
        scale = np.maximum(np.abs(y), (big if big > 0.0 else 1.0) * 1e-3)
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
            try:
                bngsim.Simulator(model, method="ode", **job.sim).run(
                    t_span=(job.t_start, t),
                    n_points=2,
                    rtol=job.rtol,
                    atol=job.atol,
                    timeout=_left(),
                )
            except bngsim.SimulationError:
                # A state this one-interval run does not reach, where the base's
                # grid does (MODEL1112050001: CV_CONV_FAILURE before t = 50), is
                # not asked of the pattern, as a non-finite one is not: whether a
                # run completes is the comparisons' question, not the pattern's.
                continue
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


# State-leak checks (issue #1307). Every other check runs a fresh model, by
# design; callers reuse one, and "the solver leaves scratch state in the
# user's Model, or drops state it should carry" is a family of its own (#786;
# #748, #705, #693). Each run here is compared with the base as every path is.

BATCH_FACTORS = (1.0, 1.1, 0.9, 1.05)
BATCH_THREADS = 4


def _species(result, base: Base) -> np.ndarray:
    """A Result's species in the base's column order."""
    names = list(result.species_names)
    x = np.asarray(result.species, dtype=float)
    return x[:, [names.index(n) for n in base.species_names]]


def _worst(verdicts: list[tuple[str, Verdict]]) -> Verdict:
    """The verdict of several comparisons: a DIFF if any is, else the first."""
    for _label, v in verdicts:
        if v.status != "pass":
            return v
    return Verdict(
        "pass", max(v.value or 0.0 for _, v in verdicts), "; ".join(v.note for _, v in verdicts)
    )


def _solver(job: Job) -> dict:
    return {"rtol": job.rtol, "atol": job.atol, "max_steps": job.max_steps}


def _timeout() -> dict:
    """What is left of the model's cap, for a ``run`` or ``run_batch`` (``run_until``
    takes none: a kill at the cap still leaves the model's other rows)."""
    return {"timeout": _left()}


def check_restart(base: Base) -> Verdict:
    """A run to the middle sample, continued to the end, is the base. Leg by
    leg with ``run_until`` on one Simulator, which carries the events' trigger
    state and pending delays across the stop (#693); and, for a model without
    events, a fresh Model given the state saved at the middle, which cannot
    know which triggers were already true and so is not asked to."""
    import bngsim

    def go():
        job = base.job
        n = job.n_points
        m = (n - 1) // 2
        t_mid = job.t_start + (job.t_end - job.t_start) * m / (n - 1)
        out = []

        sim = bngsim.Simulator(load(job.path), method="ode", **job.sim)
        if job.t_start != 0.0:
            sim.set_time(job.t_start)
        first = sim.run_until(t_mid, n_points=m + 1, **_solver(job))
        second = sim.run_until(job.t_end, n_points=n - m, **_solver(job))
        legs = np.vstack([_species(first, base), _species(second, base)[1:]])
        out.append(("run_until", base.compare(legs, "run_until to the middle and on: ")))

        model = load(job.path)
        if model._core.n_events == 0:
            sim = bngsim.Simulator(model, method="ode", **job.sim)
            head = sim.run(
                t_span=(job.t_start, t_mid), n_points=m + 1, **_solver(job), **_timeout()
            )
            state = np.asarray(sim.get_state(), dtype=float)
            fresh = bngsim.Simulator(load(job.path), method="ode", **job.sim)
            fresh.set_state(state, time=t_mid)
            tail = fresh.run(
                t_span=(t_mid, job.t_end), n_points=n - m, **_solver(job), **_timeout()
            )
            joined = np.vstack([_species(head, base), _species(tail, base)[1:]])
            out.append(("fresh", base.compare(joined, "a fresh model from the middle's state: ")))
        return _worst(out)

    return _guard(go)


def check_reuse(base: Base) -> Verdict:
    """A second run is the base: a second ``run()`` on the same Simulator, and
    a first one on a second Simulator over the same Model, each after the
    first run is undone with ``snapshot()``/``restore()`` (species, parameters,
    and the events' trigger state, #693). ``Model.reset()`` restores species
    only, so an event that writes a parameter would leave it written; restore
    is the documented way back."""
    import bngsim

    def go():
        job = base.job
        span = {"t_span": (job.t_start, job.t_end), "n_points": job.n_points, **_solver(job)}
        out = []

        sim = bngsim.Simulator(load(job.path), method="ode", **job.sim)
        snap = sim.snapshot()
        sim.run(**span, **_timeout())
        sim.restore(snap)
        again = sim.run(**span, **_timeout())
        out.append(
            ("run", base.compare(_species(again, base), "a second run() on one Simulator: "))
        )

        model = load(job.path)
        first = bngsim.Simulator(model, method="ode", **job.sim)
        snap = first.snapshot()
        first.run(**span, **_timeout())
        first.restore(snap)
        second = bngsim.Simulator(model, method="ode", **job.sim).run(**span, **_timeout())
        out.append(
            ("model", base.compare(_species(second, base), "a second Simulator on one Model: "))
        )
        return _worst(out)

    return _guard(go)


def check_batch(base: Base) -> Verdict:
    """``run_batch`` on ``BATCH_THREADS`` threads is the same points run one
    at a time, and its unchanged point is the base. The points scale the
    first parameter a sensitivity run would take (``sens_params``); a model
    with none runs the same point four times, which still asks whether the
    threads share anything."""
    import bngsim

    def go():
        job = base.job
        model = load(job.path)
        knobs = sens_params(model)
        if knobs:
            p, v = knobs[0], float(model.get_param(knobs[0]))
            points = [{p: v * f} for f in BATCH_FACTORS]
        else:
            points = [{} for _ in BATCH_FACTORS]
        kw = {
            "t_span": (job.t_start, job.t_end),
            "n_points": job.n_points,
            "params": points,
            **_solver(job),
        }
        # One at a time first: a point that does not integrate alone (a model
        # at the edge of its domain, moved 10% off its own parameter) is not
        # the batch's to judge. Threads failing where it runs is.
        try:
            serial = bngsim.Simulator(load(job.path), method="ode", **job.sim).run_batch(
                **kw, **_timeout()
            )
        except bngsim.SimulationError as e:
            first = str(e).splitlines()[0][:200]
            return Verdict(
                "skip", note=f"a point of the sweep does not run one at a time: {first}"
            )
        sim = bngsim.Simulator(model, method="ode", **job.sim)
        threaded = sim.run_batch(num_processors=BATCH_THREADS, **kw, **_timeout())
        out = [("point 0", base.compare(_species(threaded[0], base), "the batch's own point: "))]
        for i, (a, b) in enumerate(zip(threaded, serial, strict=True)):
            g = gap(_species(a, base), _species(b, base), PATH_RTOL_FACTOR * job.atol)
            note = (
                f"point {i} on {BATCH_THREADS} threads against one at a time: {g:.2e} of its peak"
            )
            out.append((f"point {i}", Verdict("pass" if g <= base.limit else "diff", g, note)))
        return _worst(out)

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
            timeout=timeout if _DEADLINE is None else min(timeout, _left()),
        )
        if out.returncode != 0:
            last = (out.stderr.strip().splitlines() or ["?"])[-1][:300]
            return Verdict("diff", None, f"runs under C, fails under {name}: {last}")
        return base.compare(json.loads(out.stdout.strip().splitlines()[-1]), f"{name}: ")

    return _guard(go)


def check_all(
    job: Job,
    locale_name: str | None,
    checks=CHECKS,
    deadline: float | None = None,
    on_verdict=None,
) -> dict[str, Verdict]:
    """Every check of ``checks`` on one model. A base that does not run is no
    reference: each check is then an exception on the base. Every run has
    until ``deadline`` (a ``time.monotonic()``): a check that has not finished
    by then is TIMEOUT, its own row only, where the runner's cap would take
    every row of the model (MODEL1006230049, 0053 and 0077 take 716-751 s of
    the nightly's 900 s, so any check added to every model tipped them over).
    ``on_verdict(check, verdict)`` hears each as it is reached, so a caller
    killed at the cap still has those."""
    global _DEADLINE
    _DEADLINE = deadline
    try:
        return _check_all(job, locale_name, checks, on_verdict or (lambda c, v: None))
    finally:
        _DEADLINE = None


class _Heard(dict):
    """The verdicts so far, each told to ``on_verdict`` as it is set."""

    def __init__(self, on_verdict):
        super().__init__()
        self.on_verdict = on_verdict

    def __setitem__(self, check, verdict):
        super().__setitem__(check, verdict)
        self.on_verdict(check, verdict)

    def update(self, other):
        for check, verdict in other.items():
            self[check] = verdict


def _check_all(job: Job, locale_name: str | None, checks, on_verdict) -> dict[str, Verdict]:
    try:
        base = Base(job)
    except Exception as e:  # noqa: BLE001
        # No base, no reference: not a finding of this suite's (the parity suites
        # own a model that does not load or run), and not a pass either.
        v = Verdict("no_base", None, f"the base run: {type(e).__name__}: {str(e)[:300]}")
        return {c: _verdict_of(e) if _timed_out(e) else v for c in checks}
    out: dict[str, Verdict] = _Heard(on_verdict)

    def spent() -> Verdict | None:
        # No check starts past the deadline: one that builds and compiles
        # Simulators without running them (backend_as_asked) has no run to bound.
        try:
            _left()
        except Exception as e:  # noqa: BLE001 - SimulationTimeout, the cap spent
            return _verdict_of(e)
        return None

    if {"sens_runs", "sens_vs_plain"} & set(checks):
        v = spent()
        out.update({"sens_runs": v, "sens_vs_plain": v} if v else check_sens(base))
    one = {
        "ajac_vs_fdjac": check_ajac,
        "klu_vs_dense": check_klu,
        "cc_vs_interp": check_cc_vs_interp,
        "cc_vs_mir": check_cc_vs_mir,
        "grid": check_grid,
        "sparsity_covers_jacobian": check_sparsity,
        "backend_as_asked": check_backend,
    }
    for name, fn in one.items():
        if name in checks:
            out[name] = spent() or fn(base)
    if "locale" in checks:
        out["locale"] = spent() or check_locale(base, locale_name)
    # After every check main runs (#1307), in what is left of the model's cap.
    for name, fn in (("restart", check_restart), ("reuse", check_reuse), ("batch", check_batch)):
        if name in checks:
            out[name] = spent() or fn(base)
    return {c: out[c] for c in checks if c in out}
