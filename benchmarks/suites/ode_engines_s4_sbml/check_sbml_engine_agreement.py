#!/usr/bin/env python3
"""Cross-engine trajectory agreement for the Table S4 SBML models.

A timing comparison is only meaningful when the engines compute the *same*
trajectory. This companion to ``run_s4_timing.py`` records, per model, whether
bngsim ``from_sbml``, libRoadRunner (rr_parity's reference engine) and COPASI, the
two non-BNGsim engines that complete the SBML models natively, compute the same
trajectory, and how accurate each is at the settings it was timed at.

Both are measured against a converged reference: the model integrated once at
``REF_RTOL``/``REF_ATOL`` (libRoadRunner; bngsim where libRoadRunner fails), on the
figure's own output grid (issue #1250). The two questions need it:

- **Same computation.** Each engine is run at the reference tolerances too, and
  compared with the reference species by species, matched by SBML id: over every
  output row where the engine returns its trajectory (bngsim, libRoadRunner), and
  at the final time for COPASI, which this reads its final state from. ``agree`` is
  True when every engine that completes is within ``AGREE_TOL`` of the reference;
  an engine that completes its timed run but not the run at the reference
  tolerances leaves agreement not established, since its timing row is then of a
  computation nothing here checked. This verdict used to compare bngsim with the others at the timing tolerance,
  which measures that tolerance's slack, not the engines: on Koo2013 at rtol 1e-6
  the final state swings by up to 7% with the number of output rows, in every
  engine, while at rtol 1e-12 the three agree to 2e-8. A 5% bar there was passed or
  failed by the grid.
- **Accuracy at the timed settings.** Each engine's final-state error from the
  reference at the figure's tolerances, ``*_error_at_timing_tolerances``. It is
  reported, not judged: it is a fact about the settings, which a reader of the
  timing table needs, and on Koo2013 it is several percent in every engine. A
  value there that is not finite is judged, as a disagreement: that is no
  tolerance's slack, and the timing row would be of a broken run.

A comparison is a relative error, ``|a - r| / max(|a|, |r|)``, over the values above
a floor tied to what the run resolves: ``RESOLVED`` times the reference's absolute
tolerance for the verdict, and the timed run's own absolute tolerance for its
accuracy (below it a solver controls only the absolute error). It never passes for
want of a comparison: a value that is not finite is a disagreement, and species
that cannot be matched one to one (one the reference has and the engine does not,
or the other way round), or no value above the floor, leave agreement not
established (``agree`` None), with the reason and how many species were compared.

The earlier report's order-independent comparison, of the sorted final values, is
kept as ``bngsim_vs_*_sorted_mre`` beside earlier reports; it cannot tell one
species from another, so nothing is judged on it.

Cheap (bngsim + RoadRunner + COPASI; no AMICI compile); safe to run outside the
timing sweep. Writes ``report_ode_engines_s4_agreement.json``.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    python check_sbml_engine_agreement.py            # writes the default report
    python check_sbml_engine_agreement.py --out /tmp/probe.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# Default to the checkout this file lives in (benchmarks/suites/<suite>/), not
# ~/Code/bngsim: the old default made every import below resolve only on a
# machine whose clone happened to sit at that path, and the module-scope
# `import _bng_common` died with ModuleNotFoundError anywhere else -- CI
# included (GH #489). BNGSIM_ROOT still overrides, which is the documented case:
# a different venv running against the canonical checkout.
BNGSIM = Path(os.environ.get("BNGSIM_ROOT") or Path(__file__).resolve().parents[3])
PARITY = BNGSIM / "parity_checks"
for _p in (str(PARITY), str(PARITY / "bng_parity")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rr_parity import _rr_common as rc  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_ode_engines_s4_agreement.json"
RR_PARITY = PARITY / "rr_parity"
MODELS_ROOT = RR_PARITY / "models"
ODE_JOBS = RR_PARITY / "ode_jobs.json"

# Same six models/order as run_s4_timing.py.
MODELS = [
    ("BIOMD0000000834", "Verma2016", 1),
    ("BIOMD0000000468", "Koo2013", 2),
    ("BIOMD0000000559", "Ouzounoglou2014", 3),
    ("BIOMD0000000667", "Hornberg2005", 4),
    ("BIOMD0000000474", "Smith2013", 5),
    ("BIOMD0000001065", "vonDassow2000", 6),
]

# The converged reference, and each engine's run for the same-computation verdict
# (issue #1250).
REF_RTOL, REF_ATOL = 1e-12, 1e-15
# An engine at the reference tolerances further than this from the reference
# computes a different trajectory: the timing row compares different solves. The
# rr_parity oracle's tolerance (ode_jobs.json, `oracle.tol`). Converged, the S4
# engines agree to 4e-6 or better over their trajectories.
AGREE_TOL = 1e-4
# For the verdict, a value is compared where the reference resolves it: above
# this many times REF_ATOL (1e-12). Below, its relative error measures the
# tolerance, not the engine. The accuracy at the timing tolerances is read over
# the values above the timed run's own atol, not this many times it.
RESOLVED = 1e3


def _sedml_initial_time(sedml_path, output_start, t_end):
    """SED-ML ``initialTime`` (integration start), distinct from ``outputStartTime``.
    Mirror of the harness reader so both integrate the same span [initialTime, t_end]
    (pre-window events fire). Falls back to the first uTC, then 0.0."""
    p = Path(sedml_path)
    if not p.exists():
        return 0.0

    def _attr(tag, name):
        mm = re.search(rf'{name}\s*=\s*"([^"]*)"', tag)
        try:
            return float(mm.group(1)) if mm else None
        except ValueError:
            return None

    first = None
    for m in re.finditer(r"<uniformTimeCourse\b[^>]*>", p.read_text(errors="replace")):
        tag = m.group(0)
        it = _attr(tag, "initialTime")
        if it is None:
            continue
        os_, oe = _attr(tag, "outputStartTime"), _attr(tag, "outputEndTime")
        if (
            os_ is not None
            and oe is not None
            and abs(os_ - output_start) < 1e-6
            and abs(oe - t_end) < 1e-6
        ):
            return it
        if first is None:
            first = it
    return first if first is not None else 0.0


def _canonical(names):
    """Species ids as bngsim spells them: an SBML id that is an ExprTk reserved
    word or builtin is renamed by the loader's own rule (``NULL`` is ``_ant_NULL``),
    and every other id is itself."""
    from bngsim._sbml_loader import _safe_name

    return [_safe_name(str(n)) for n in names]


def _compare(names, values, ref_names, ref_values, floor):
    """*values* against *ref_values*, species matched by id, as a dict:
    ``mre`` (the largest relative error over the values above *floor*, or None),
    ``compared`` (species with a value above it), ``species`` (the reference's),
    ``non_finite`` and ``why`` (None, or what kept it from a comparison).

    Rows are output rows; an engine with one row (its final state) is compared at
    the reference's last. The engine and the reference must report the same
    species, once each: one either of them lacks is not compared over what is
    left, since that would pass a species nothing checked."""
    keys, ref_keys = _canonical(names), _canonical(ref_names)
    out = {"mre": None, "compared": 0, "species": len(ref_keys), "non_finite": 0, "why": None}
    a = np.atleast_2d(np.asarray(values, float))
    r = np.atleast_2d(np.asarray(ref_values, float))
    if len(set(keys)) != len(keys) or len(set(ref_keys)) != len(ref_keys):
        out["why"] = "species ids are not unique"
        return out
    if a.shape[1] != len(keys) or r.shape[1] != len(ref_keys):
        out["why"] = "species ids and values do not line up"
        return out
    missing = [k for k in ref_keys if k not in keys]
    if missing:
        out["why"] = f"{len(missing)} of the reference's species are not reported: {missing[:3]}"
        return out
    extra = [k for k in keys if k not in ref_keys]
    out["extra_species"] = len(extra)
    if extra:
        out["why"] = f"{len(extra)} species are not in the reference: {extra[:3]}"
        return out
    a = a[:, [keys.index(k) for k in ref_keys]]
    if a.shape[0] == 1:
        r = r[-1:]
    if a.shape != r.shape:
        out["why"] = f"{a.shape[0]} output rows against the reference's {r.shape[0]}"
        return out
    bad = ~(np.isfinite(a) & np.isfinite(r))
    if bad.any():
        out["non_finite"] = int(bad.any(axis=0).sum())
        out["why"] = f"{out['non_finite']} species hold a value that is not finite"
        return out
    scale = np.maximum(np.abs(a), np.abs(r))
    above = scale > floor
    out["compared"] = int(above.any(axis=0).sum())
    if not above.any():
        out["why"] = f"no value above {floor:g}"
        return out
    out["mre"] = float(np.max(np.abs(a - r)[above] / scale[above]))
    return out


def _status(cmp) -> str:
    """``agree``, ``disagree`` or ``not established`` for one engine's converged
    comparison. Only a comparison that was made and is within AGREE_TOL agrees."""
    if cmp["non_finite"] or (cmp["mre"] is not None and cmp["mre"] > AGREE_TOL):
        return "disagree"
    return "agree" if cmp["mre"] is not None else "not established"


def _legacy_sorted_mre(a_final, x_final):
    """The earlier reports' order-independent number: sorted final values against
    each other, over values above 1e-6. None where there is nothing to sort
    against (different counts) or a value is not finite, which it used to drop."""
    a, x = np.asarray(a_final, float).ravel(), np.asarray(x_final, float).ravel()
    if a.shape != x.shape or not (np.isfinite(a).all() and np.isfinite(x).all()):
        return None
    a, x = np.sort(a), np.sort(x)
    scale = np.maximum(np.abs(a), np.abs(x))
    mask = scale > 1e-6
    return float(np.max(np.abs(a - x)[mask] / scale[mask])) if mask.any() else None


def _copasi_final_species(xml, t0, t1, npnt, rtol, atol):
    """Final SPECIES concentrations from a COPASI deterministic time course, read
    from the model's metabolite objects (exactly the N species — the raw time series
    also carries compartments and global quantities, which would corrupt an
    order-independent comparison). Returns (SBML ids, 1-D array of length N)."""
    import COPASI

    dm = COPASI.CRootContainer.addDatamodel()
    try:
        if not dm.importSBML(str(xml)):
            raise RuntimeError("COPASI importSBML returned False")
        task = dm.getTask("Time-Course")
        task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
        task.setScheduled(True)
        prob = task.getProblem()
        prob.setDuration(t1 - t0)
        prob.setStepNumber(npnt - 1)
        prob.setOutputStartTime(t0)
        prob.setTimeSeriesRequested(True)
        mth = task.getMethod()
        for pn, pv in (("Absolute Tolerance", atol), ("Relative Tolerance", rtol)):
            pp = mth.getParameter(pn)
            if pp is not None:
                pp.setValue(pv)
        if not task.initialize(COPASI.CCopasiTask.OUTPUT_UI):
            raise RuntimeError("COPASI task init failed")
        if not task.process(True):
            raise RuntimeError("COPASI task process failed")
        # After integration the model is left at the final state; read species
        # concentrations straight off the metabolite objects (species only).
        mod = dm.getModel()
        mod.updateInitialValues(COPASI.CCore.Framework_Concentration)
        metabs = mod.getMetabolites()
        ids = [metabs.get(i).getSBMLId() for i in range(metabs.size())]
        values = [metabs.get(i).getConcentration() for i in range(metabs.size())]
        return ids, np.array(values, float)
    finally:
        with contextlib.suppress(Exception):
            COPASI.CRootContainer.removeDatamodel(dm)


def _bngsim_final(xml, span, rtol, atol):
    """bngsim's (species names, trajectory) over *span* = (t0, t1, n_points)."""
    import bngsim

    t0, t1, n = span
    r = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), method="ode").run(
        t_span=(t0, t1), n_points=n, rtol=rtol, atol=atol
    )
    return list(r.species_names), np.asarray(r.species)


def _roadrunner_final(xml, span, rtol, atol):
    """libRoadRunner's (species names, trajectory) over *span*."""
    t0, t1, n = span
    _, values, names, _ = rc.rr_ode(str(xml), t0, t1, n, rtol, atol)
    return list(names), np.asarray(values)


def _copasi_final(xml, span, rtol, atol):
    """COPASI's (SBML ids, final species concentrations as one row) over *span*."""
    t0, t1, n = span
    ids, values = _copasi_final_species(xml, t0, t1, n, rtol, atol)
    return ids, values[None, :]


def _attempt(fn, *args):
    """``(fn(*args), None)``, or ``(None, why)`` where it raises."""
    try:
        return fn(*args), None
    except Exception as e:  # noqa: BLE001 - an engine's failure is recorded, not raised
        return None, f"{type(e).__name__}: {e}"


# The engines, in the order a reference is chosen from. Looked up when called, so
# a test can stand in for one.
def _engines():
    return (
        ("roadrunner", _roadrunner_final),
        ("bngsim", _bngsim_final),
        ("copasi", _copasi_final),
    )


def _version(module, read):
    """*module*'s version as *read* reads it, or why there is none."""
    try:
        return read(__import__(module))
    except ImportError:
        return "not installed"
    except AttributeError:
        return "?"


def _show(rec, key):
    v = rec.get(key)
    return f"{v:.1e}" if v is not None else "  -    "


def _check_model(rec, xml, span, rtol, atol):
    """Fill *rec* with one model's comparisons (see the module docstring)."""
    # --- At the timing tolerances ------------------------------------------------
    timed = {}
    for engine, fn in _engines():
        got, err = _attempt(fn, xml, span, rtol, atol)
        if got is None:
            rec[f"{engine}_error"] = err
        else:
            timed[engine] = got
    if "bngsim" not in timed:
        rec["error"] = f"bngsim from_sbml: {rec.pop('bngsim_error')}"
        rec["agree"] = None
        return
    bn_names, bn = timed["bngsim"]
    rec.update({"n_species": int(bn.shape[1]), "agree_tol": AGREE_TOL})
    # The earlier reports' numbers: each engine against bngsim, sorted final values.
    for engine in ("roadrunner", "copasi"):
        if engine in timed:
            rec[f"bngsim_vs_{engine}_sorted_mre"] = _legacy_sorted_mre(
                bn[-1], timed[engine][1][-1]
            )

    # --- The converged reference, and each engine at its tolerances -------------
    tight = {}
    for engine, fn in _engines():
        got, err = _attempt(fn, xml, span, REF_RTOL, REF_ATOL)
        if got is None:
            rec[f"{engine}_reference_tolerance_error"] = err
        else:
            tight[engine] = got
    ref_engine = next((e for e in ("roadrunner", "bngsim") if e in tight), None)
    rec["reference"] = (
        {"engine": ref_engine, "rtol": REF_RTOL, "atol": REF_ATOL} if ref_engine else None
    )
    if ref_engine is None:
        rec["agree"] = None
        rec["why"] = "no engine completed at the reference tolerances"
        return
    ref_names, ref = tight[ref_engine]
    # Same computation: every engine that completes, converged, on the reference.
    statuses = {}
    for engine, (names, values) in tight.items():
        if engine == ref_engine:
            continue
        cmp = _compare(names, values, ref_names, ref, RESOLVED * REF_ATOL)
        rec[f"{engine}_vs_reference_converged"] = cmp
        statuses[engine] = _status(cmp)
    # An engine timed at the figure's tolerances that did not complete at the
    # reference's: its timing row is of a computation nothing here checked.
    for engine in timed:
        if engine not in tight:
            rec[f"{engine}_vs_reference_converged"] = {
                "mre": None,
                "compared": 0,
                "species": len(ref_names),
                "non_finite": 0,
                "why": "no run at the reference tolerances: "
                + rec[f"{engine}_reference_tolerance_error"],
            }
            statuses[engine] = "not established"
    # Accuracy at the timed settings, at the final time: reported, not judged,
    # except a value that is not finite anywhere in the timed run, which is no
    # tolerance's slack.
    broken = [e for e, (_, v) in timed.items() if not np.isfinite(np.asarray(v, float)).all()]
    for engine, (names, values) in timed.items():
        cmp = _compare(names, np.asarray(values)[-1:], ref_names, ref[-1:], atol)
        rec[f"{engine}_error_at_timing_tolerances"] = cmp["mre"]
        rec[f"{engine}_error_at_timing_tolerances_detail"] = cmp
    rec["converged_status"] = statuses
    if broken:
        rec["agree"] = False
        rec["why"] = f"a value that is not finite at the timing tolerances: {', '.join(broken)}"
    elif not statuses:
        rec["agree"] = None
        rec["why"] = "only the reference engine completed at the reference tolerances"
    elif "disagree" in statuses.values():
        rec["agree"] = False
    elif "not established" in statuses.values():
        rec["agree"] = None
        rec["why"] = "; ".join(
            f"{e}: {rec[f'{e}_vs_reference_converged']['why']}"
            for e, st in statuses.items()
            if st == "not established"
        )
    else:
        rec["agree"] = True
    mres = [rec[f"{e}_vs_reference_converged"]["mre"] for e in statuses]
    rec["worst_converged_mre"] = max((m for m in mres if m is not None), default=None)


def _summary(rec) -> str:
    head = f"#{rec['row']} {rec['label']:16}"
    if "error" in rec:
        return f"{head} FAIL: {rec['error']}"
    if rec.get("reference") is None:
        return f"{head} {rec.get('why')}"
    tag = {True: "AGREE", False: "DISAGREE", None: "NOT ESTABLISHED"}[rec["agree"]]
    if rec.get("why"):
        tag += f" ({rec['why']})"
    conv = "  ".join(
        f"{e}={_show(rec[f'{e}_vs_reference_converged'], 'mre')}"
        f"({rec[f'{e}_vs_reference_converged']['compared']}/"
        f"{rec[f'{e}_vs_reference_converged']['species']})"
        for e in rec["converged_status"]
    )
    timed = "  ".join(
        f"{e} {_show(rec, f'{e}_error_at_timing_tolerances')}" for e, _ in _engines()
    )
    return (
        f"{head} converged vs {rec['reference']['engine']}: {conv}  {tag};  "
        f"at the timing tolerances: {timed}"
    )


def build_parser():
    """The CLI. ``--help`` must not start the agreement pass (GH #489).

    This script reads no argv at all, so ``--help`` ran all three engines over
    the Table S4 models and overwrote the *committed*
    ``report_ode_engines_s4_agreement.json`` at the end. ``--out`` is what makes
    a diagnostic run possible without that; the pass itself takes no options,
    since the model list and tolerance are fixed here by design.
    """
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help=f"Write the report here (default: {OUT.name}).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import bngsim

    jobs = {j["model_id"]: j for j in json.loads(ODE_JOBS.read_text())["jobs"]}
    results = []
    for mid, label, row in MODELS:
        job = jobs.get(mid)
        p = job["params"]
        xml = MODELS_ROOT / mid / Path(job["model"]).name
        sedml = MODELS_ROOT / mid / p["sedml"] if p.get("sedml") else None
        output_start, t1 = float(p["t_start"]), float(p["t_end"])
        # Integrate from SED-ML initialTime (not outputStartTime) so pre-window
        # events fire — same span as the timing harness.
        t0 = _sedml_initial_time(sedml, output_start, t1) if sedml else output_start
        npnt, rtol, atol = int(p["n_points"]), float(p["rtol"]), float(p["atol"])
        rec = {
            "model_id": mid,
            "label": label,
            "row": row,
            "horizon": {
                "t_start": t0,
                "output_start": output_start,
                "t_end": t1,
                "n_points": npnt,
                "rtol": rtol,
                "atol": atol,
            },
        }

        try:
            _check_model(rec, xml, (t0, t1, npnt), rtol, atol)
        except Exception as e:  # noqa: BLE001 - one model's failure is recorded, not fatal
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["agree"] = None
        results.append(rec)
        print(_summary(rec))

    doc = {
        "_meta": {
            "suite": "ode_engines_s4_sbml",
            "description": (
                "Cross-engine trajectory agreement for the Table S4 models, against "
                "a converged reference (issue #1250): each engine at the reference "
                "tolerances is compared with it species by species, matched by SBML "
                "id, over its trajectory (COPASI: final state), and agree is True "
                "only when every engine that completes is within agree_tol; False "
                "when one is not or holds a non-finite value, at either tolerance; "
                "None when agreement could not be established (see why), including "
                "for an engine timed but not run at the reference tolerances. Each "
                "engine's final-state error at the figure's tolerances is reported, "
                "not judged. "
                "bngsim_vs_*_sorted_mre are the earlier reports' order-independent "
                "numbers, kept for comparison."
            ),
            "reference": {
                "engines": ["libRoadRunner", "bngsim where libRoadRunner fails"],
                "rtol": REF_RTOL,
                "atol": REF_ATOL,
            },
            "engines": ["bngsim Model.from_sbml", "libRoadRunner", "COPASI"],
            "metric": (
                "max |a - r| / max(|a|, |r|) over values above a floor: RESOLVED x "
                "REF_ATOL for the verdict, the timed run's atol for its accuracy"
            ),
            "resolved_factor": RESOLVED,
            "agree_tol": AGREE_TOL,
            "versions": {
                "bngsim": bngsim.__version__,
                "roadrunner": _version("roadrunner", lambda m: m.__version__),
                # Where COPASI is not installed its column is an error, and
                # this was an ImportError after every engine had run.
                "copasi": _version("COPASI", lambda m: m.CVersion.VERSION.getVersion()),
            },
            "hardware": rc.hardware_info(),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"\nwrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
