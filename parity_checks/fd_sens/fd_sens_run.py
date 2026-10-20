#!/usr/bin/env python3
"""bngsim's analytic forward sensitivities against central differences of its
own trajectories -> a ``_core`` report (issue #804).

The nightly checks forward sensitivities only against AMICI, and only on SBML.
A ``.net`` model has no sensitivity oracle at all (BNG2.pl computes none), and
when bngsim and AMICI agree nothing adjudicates. A finite difference of bngsim's
own state runs is an oracle independent of both, for any model bngsim can
integrate.

One job per (model, sensitivity method), both CVODES correctors (``staggered``
and ``simultaneous``), as ``amici_sens`` runs them:

    analytic and FD tensors within tolerance ...... PASS
    over tolerance, or a non-finite cell .......... DIFF
    bngsim DECLARED it cannot differentiate ........ UNSUPPORTED (by type)
    bngsim raised .................................. EXCEPTION
    no parameter to differentiate, or the FD
      reference did not converge with its step .... BAD_TEST (nothing compared)
    wall-clock cap exceeded ........................ TIMEOUT

The reference
-------------
For each parameter ``p`` (value ``v``), central differences at two steps
``h`` and ``h/2`` (``h = FD_REL_STEP·|v|``, or ``FD_REL_STEP`` at ``v = 0``),
each from two plain runs at ``FD_RTOL`` / ``FD_ATOL``, far tighter than the
analytic run's, combined by Richardson extrapolation: ``(4·D(h/2) - D(h))/3``,
error O(h⁴). The two traps ``test_time_switch_sens_fd_reference.py`` (#368)
documents are what that guards against:

* **An unresolvable step.** A parameter that barely moves a species puts the
  difference under the solver's own floor, and the quotient is noise over 2h.
  The step is relative (``FD_REL_STEP``, coarse on purpose) and the runs are at
  ``FD_RTOL`` = 1e-12. A column is compared only where its analytic value
  clears both the verdict's ``atol/|p|`` floor and the difference's own noise
  (:func:`judgeable_columns`); the rest are left out and named on the row. In
  every compared column the two steps' quotients must agree to ``--fd-tol`` on
  the column's own scale: one whose steps disagree is not a reference, and the
  row is BAD_TEST with the disagreement recorded rather than a DIFF blamed on
  the analytic side. A row with no column to compare is BAD_TEST too.
* **A switch node.** A central difference taken exactly at a time where a
  switch fires straddles it and reads half the one-sided derivative. The
  manifest keeps such models' sample grids off their switch times.

The two sides do not share the code under test. bngsim computes forward
sensitivities only through generated code (the interpreted path was retired,
GH #214), so every analytic run is compiled, which is the path #689 broke; the
plain runs of the difference are pinned to the ExprTk interpreter.

The verdict is ``_amici_sens.sens_verdict``: ``_core.differ`` over the
flattened tensor with the ``atol/|p|`` noise floor, the same oracle and the same
floor ``amici_sens`` uses, so a tolerance change lands in both. ``timing.bngsim``
carries the analytic run's work counters for the nightly efficiency check.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "amici_parity"))

import _amici_sens as asens  # noqa: E402
from _core import (  # noqa: E402
    JobResult,
    Outcome,
    capture_exception,
    tally,
    versions,
    write_report,
)
from _core.versions import git_rev  # noqa: E402

JOBS = HERE / "fd_sens_jobs.json"
SENS_METHODS = asens.SENS_METHODS

# The analytic run's tolerances: the suite's (amici_sens's) pair.
DEFAULT_RTOL = asens.DEFAULT_RTOL
DEFAULT_ATOL = asens.DEFAULT_ATOL
# The finite-difference runs: tight enough that a relative step of 1e-3 leaves
# the difference six or more orders above their own noise.
FD_RTOL = 1e-12
FD_ATOL = 1e-16
FD_REL_STEP = 1e-3
# Most parameters differenced per model: each costs four plain runs.
DEFAULT_PARAM_CAP = 6
DEFAULT_TIMEOUT = 300.0
REFERENCE_ENGINE = "finite_difference"


def load_model(path: Path):
    import bngsim

    if path.suffix == ".net":
        return bngsim.Model.from_net(str(path))
    return bngsim.Model.from_sbml(str(path))


def candidate_params(model) -> list[str]:
    """The parameters a column can be taken for: the model's knobs, less those a
    sensitivity is refused for by construction (compartment sizes, frozen
    parameters)."""
    excluded = set(getattr(model, "compartment_size_params", None) or ())
    excluded |= set(getattr(model, "frozen_params", None) or ())
    return [p for p in model.primary_param_names if p not in excluded]


def fd_step(value: float, rel: float = FD_REL_STEP) -> float:
    """The larger of the two central-difference steps for a parameter at ``value``."""
    return rel * abs(value) if value != 0.0 else rel


def richardson(d_h: np.ndarray, d_h2: np.ndarray) -> np.ndarray:
    """Central differences at ``h`` and ``h/2`` combined to O(h⁴)."""
    return (4.0 * d_h2 - d_h) / 3.0


def step_disagreement(d_h: np.ndarray, d_h2: np.ndarray) -> float:
    """How far one column's two quotients are apart, on the column's own scale:
    ``max |D(h) - D(h/2)| / max |D(h/2)|``. A converged reference reads O(h²); a
    step lost in the solver's noise reads O(1). Peak-relative, as ``_core.differ``
    judges a column: per cell, a cell near zero in an otherwise large column read
    1e-2 on references that converge (nested_derived_rate_const, expr_sens_chain)."""
    peak = float(np.max(np.abs(d_h2))) if d_h2.size else 0.0
    if peak == 0.0:
        return 0.0 if not np.any(d_h) else float("inf")
    return float(np.max(np.abs(d_h - d_h2))) / peak


def _plain_species(path: Path, param: str, value: float, job: dict) -> np.ndarray:
    import bngsim

    model = load_model(path)
    model.set_param(param, value)
    model.reset()
    run = bngsim.Simulator(model, method="ode", codegen=False).run(
        t_span=(job["t_start"], job["t_end"]),
        n_points=job["n_points"],
        rtol=FD_RTOL,
        atol=FD_ATOL,
        max_steps=job.get("max_steps", 100_000),
    )
    return np.asarray(run.species, dtype=float)


def fd_tensor(path: Path, params: list[str], values: list[float], job: dict):
    """The Richardson-extrapolated FD tensor ``(n_t, n_sp, n_p)`` and each
    column's step disagreement."""
    cols, disagreements = [], []
    for p, v in zip(params, values, strict=False):
        h = fd_step(v)
        quotients = []
        for step in (h, h / 2.0):
            up = _plain_species(path, p, v + step, job)
            down = _plain_species(path, p, v - step, job)
            quotients.append((up - down) / (2.0 * step))
        d_h, d_h2 = quotients
        disagreements.append(step_disagreement(d_h, d_h2))
        cols.append(richardson(d_h, d_h2))
    return np.stack(cols, axis=-1), disagreements


# How many multiples of a finite difference's own noise a column's analytic
# value must exceed before the difference is a reference for it.
FD_NOISE_FACTOR = 10.0


def judgeable_columns(analytic: np.ndarray, x_max: float, values, atol: float) -> list[int]:
    """The parameter columns a finite difference can judge.

    Two floors, and a column's analytic peak has to clear both:

    * the verdict's own, ``factor·atol/|p|`` (``_amici_sens.sens_resolution_floors``):
      below it the analytic run itself does not resolve the column, and the
      verdict forgives its cells anyway;
    * the difference's: the plain runs resolve a state to
      ``FD_RTOL·max|x| + FD_ATOL``, so a quotient over ``2h`` resolves
      ``FD_NOISE_FACTOR`` times that over the smaller step ``h/2``.

    A column under either is one neither side can speak for, and is left out
    of the comparison (counted on the row). Measured: erk_activation's q3 (value
    3e-4, step 1.5e-7) and mm_tqssa_stiff's Km (1e-8) have analytic peaks of
    5e-8 and 8e-4 under difference noise of order 1e-1 and 1e3, and their
    differences' two steps were 50% and 80% apart. A column wrongly LARGE
    analytically still clears both floors and is still judged.
    """
    floors = asens.sens_resolution_floors(values, atol)
    peaks = np.nanmax(np.abs(analytic), axis=(0, 1)) if analytic.size else np.zeros(0)
    out = []
    for j, v in enumerate(values):
        h_small = fd_step(v) / 2.0
        fd_floor = FD_NOISE_FACTOR * (FD_RTOL * x_max + FD_ATOL) / h_small
        if peaks[j] > max(floors[j], fd_floor):
            out.append(j)
    return out


def _worker(spec: dict, q) -> None:
    """One (model, method) job in a child process; puts one result dict on ``q``."""
    t0 = time.perf_counter()
    res: dict = {"key": spec["key"]}
    try:
        import bngsim

        path = REPO / spec["path"]
        model = load_model(path)
        cands = candidate_params(model)
        params = asens.select_params(cands, spec["param_cap"])
        res["n_param_candidates"] = len(cands)
        if not params:
            res.update(status="bad_test", why="no parameter to differentiate")
            q.put(res)
            return
        values = [float(model.get_param(p)) for p in params]
        try:
            sim = bngsim.Simulator(
                model,
                method="ode",
                sensitivity_params=list(params),
                sensitivity_method=spec["sens_method"],
            )
            t1 = time.perf_counter()
            run = sim.run(
                t_span=(spec["t_start"], spec["t_end"]),
                n_points=spec["n_points"],
                rtol=spec["rtol"],
                atol=spec["atol"],
                max_steps=spec.get("max_steps", 100_000),
            )
            integrate_sec = time.perf_counter() - t1
        except bngsim.SensitivityUnsupportedError as e:
            res.update(status="unsupported", why=str(e)[:400], exception_class=type(e).__name__)
            q.put(res)
            return
        analytic = np.asarray(run.sensitivities, dtype=float)
        stats = run.solver_stats if hasattr(run, "solver_stats") else {}
        from _core.work import work_counters

        res["timing"] = {
            "bngsim": {
                "integrate_sec": round(integrate_sec, 6),
                "config": {
                    "codegen": sim.codegen_backend,
                    "jacobian": sim.jacobian_strategy,
                    "sens_method": spec["sens_method"],
                },
                "work": work_counters(stats),
            }
        }
        reference, per_column = fd_tensor(path, params, values, spec)
        x_max = float(np.nanmax(np.abs(np.asarray(run.species, dtype=float))))
        judged = judgeable_columns(analytic, x_max, values, spec["atol"])
        disagreement = max((per_column[j] for j in judged), default=0.0)
        n_live = len(judged)
        res["params"] = params
        res["fd_step_disagreement"] = disagreement
        res["n_columns_judged"] = n_live
        res["columns_not_judged"] = [p for j, p in enumerate(params) if j not in judged]
        res["max_abs_sx"] = float(np.nanmax(np.abs(analytic))) if analytic.size else 0.0
        res["resolvable_columns"] = asens.resolvable_param_columns(analytic, values, spec["atol"])
        if analytic.shape != reference.shape:
            res.update(
                status="diff",
                value=float("inf"),
                why=f"tensor shapes differ: analytic {analytic.shape}, FD {reference.shape}",
            )
        elif n_live == 0:
            res.update(
                status="bad_test",
                why="no column carries a sensitivity above the solver's resolution: "
                "the comparison would establish nothing",
            )
        elif disagreement > spec["fd_tol"]:
            res.update(
                status="bad_test",
                why=(
                    f"the FD reference did not converge with its step: the two steps' "
                    f"quotients differ by {disagreement:.2e} relative (> {spec['fd_tol']:g})"
                ),
            )
        else:
            v = asens.sens_verdict(
                analytic[..., judged],
                reference[..., judged],
                param_values=[values[j] for j in judged],
                atol=spec["atol"],
            )
            res.update(
                status="pass" if v["passed"] else "diff",
                value=float(v["max_rel"]),
                verdict={
                    k: v[k]
                    for k in ("n_below_noise_floor", "n_noise_rescued", "noise_decisive")
                    if k in v
                },
            )
    except Exception as e:  # noqa: BLE001 - one model's failure is that row's EXCEPTION
        captured = capture_exception("bngsim", e)
        res.update(status="exception", exception=captured.text, exception_class=captured.cls)
    res["wall_sec"] = round(time.perf_counter() - t0, 3)
    q.put(res)


_OUTCOME = {
    "pass": Outcome.PASS,
    "diff": Outcome.DIFF,
    "exception": Outcome.EXCEPTION,
    "unsupported": Outcome.UNSUPPORTED,
    "bad_test": Outcome.BAD_TEST,
    "timeout": Outcome.TIMEOUT,
    "dead": Outcome.EXCEPTION,
}


def make_specs(jobs: list[dict], args) -> list[dict]:
    specs = []
    for job in jobs:
        for method in SENS_METHODS:
            specs.append(
                {
                    "key": f"{job['model_id']}|sens/{method}",
                    "model_id": job["model_id"],
                    "path": job["path"],
                    "t_start": float(job.get("t_start", 0.0)),
                    "t_end": float(job["t_end"]),
                    "n_points": int(job.get("n_points", 51)),
                    "max_steps": int(job.get("max_steps", 100_000)),
                    "rtol": args.rtol,
                    "atol": args.atol,
                    "fd_tol": args.fd_tol,
                    "param_cap": int(job.get("param_cap", args.param_cap)),
                    "sens_method": method,
                    "timeout": float(job.get("timeout", args.timeout)),
                }
            )
    return specs


def to_result(spec: dict, r: dict, stamp: dict) -> JobResult:
    outcome = _OUTCOME.get(r.get("status"), Outcome.EXCEPTION)
    extra = {
        "sens_method": spec["sens_method"],
        "params": r.get("params"),
        "n_param_candidates": r.get("n_param_candidates"),
        "fd_step_disagreement": r.get("fd_step_disagreement"),
        "n_columns_judged": r.get("n_columns_judged"),
        "columns_not_judged": r.get("columns_not_judged"),
        "max_abs_sx": r.get("max_abs_sx"),
        "resolvable_columns": r.get("resolvable_columns"),
        **(r.get("verdict") or {}),
    }
    return JobResult(
        model_id=spec["model_id"],
        method=f"sens/{spec['sens_method']}",
        reference_engine=REFERENCE_ENGINE,
        outcome=outcome.value,
        metric="max_rel_err",
        value=r.get("value"),
        tol=None,
        exception=r.get("exception") or r.get("why") or "",
        exception_class=r.get("exception_class"),
        wall_sec=r.get("wall_sec"),
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
        versions=stamp,
        comment="",
        timing=r.get("timing"),
        extra={k: v for k, v in extra.items() if v is not None},
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs", default=str(JOBS), help="the committed manifest")
    ap.add_argument("--out", default=str(HERE / "runs" / "report_fd_sens.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per job (s)")
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL, help="the analytic run's")
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL, help="the analytic run's")
    ap.add_argument(
        "--fd-tol",
        type=float,
        default=1e-3,
        help="how far apart the two steps' quotients may be before the reference "
        "is declared unconverged (BAD_TEST)",
    )
    ap.add_argument("--param-cap", type=int, default=DEFAULT_PARAM_CAP)
    ap.add_argument("--models", default="", help="comma-separated model_id substrings")
    args = ap.parse_args(argv)

    jobs = json.loads(Path(args.jobs).read_text())["jobs"]
    if args.models:
        subs = [s.strip() for s in args.models.split(",") if s.strip()]
        jobs = [j for j in jobs if any(s in j["model_id"] for s in subs)]
    specs = make_specs(jobs, args)
    print(f"fd_sens: {len(jobs)} models x {len(SENS_METHODS)} methods = {len(specs)} jobs")
    t0 = time.perf_counter()

    def on_done(finished, total, res):
        print(f"  [{finished}/{total}] {res.get('status', '?'):11} {res['key']}", flush=True)

    raw = asens.schedule(
        specs, _worker, workers=args.workers, timeout_of=lambda s: s["timeout"], on_done=on_done
    )
    stamp = versions.stamp(REFERENCE_ENGINE)
    results = [to_result(spec, r, stamp) for spec, r in zip(specs, raw, strict=False)]
    meta = {
        "suite": "fd_sens",
        "reference_engine": REFERENCE_ENGINE,
        "regime": "forward_sensitivity",
        "git_rev": git_rev(str(HERE)),
        "versions": stamp,
        "tally": tally(r.outcome for r in results),
        "n_jobs": len(specs),
        "n_models": len(jobs),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "config": {
            "rtol": args.rtol,
            "atol": args.atol,
            "fd_rtol": FD_RTOL,
            "fd_atol": FD_ATOL,
            "fd_rel_step": FD_REL_STEP,
            "fd_tol": args.fd_tol,
            "param_cap": args.param_cap,
        },
        "hardware": asens.hardware_info(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(out, results, meta=meta)
    print(f"report: {out}  {meta['tally']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
