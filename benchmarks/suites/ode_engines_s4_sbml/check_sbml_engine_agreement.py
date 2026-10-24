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
  ``agree`` is True when every one that completes is within ``AGREE_TOL`` of the
  reference: the engines converge to one trajectory. This verdict used to compare
  bngsim with the others at the timing tolerance, which measures that tolerance's
  slack, not the engines: on Koo2013 at rtol 1e-6 the final state swings by up to 7%
  with the number of output rows, in every engine, while at rtol 1e-12 the three
  agree to 2e-8. A 5% bar there was passed or failed by the grid.
- **Accuracy at the timed settings.** Each engine's final-state error from the
  reference at the figure's tolerances, ``*_error_at_timing_tolerances``. It is
  reported, not judged: it is a fact about the settings, which a reader of the
  timing table needs, and on Koo2013 it is several percent in every engine.

Agreement uses an ORDER-INDEPENDENT comparison of the sorted final-state value
multisets — robust for spatial models whose species share display names across grid
cells (name alignment misaligns those and manufactures a spurious disagreement).
COPASI's final state is read from its 148 metabolite objects, not its raw time series
(which also carries compartments/global quantities); RoadRunner is additionally
name-aligned where names are unique, as a cross-check. The engines' differences from
bngsim at the timing tolerances are kept as ``bngsim_vs_*`` for comparison with
earlier reports.

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
# engines agree to 2e-8 or better.
AGREE_TOL = 1e-4


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


def _mre(a, x):
    a = np.asarray(a, float)
    x = np.asarray(x, float)
    n = min(len(a), len(x))
    a, x = a[:n], x[:n]
    scale = np.maximum(np.abs(a), np.abs(x))
    mask = scale > 1e-6
    return float(np.max(np.abs(a - x)[mask] / scale[mask])) if mask.any() else 0.0


def _sorted_mre(a_final, x_final):
    """Order-independent: compare the sorted final-state value multisets. Robust to
    species-name/order differences across engines (a spatial model's repeated display
    names defeat name alignment)."""
    return _mre(np.sort(np.asarray(a_final, float)), np.sort(np.asarray(x_final, float)))


def _mre_by_name(an, a, xn, x):
    ad = {n.lower(): float(a[-1, i]) for i, n in enumerate(an)}
    xd = {n.lower(): float(x[-1, i]) for i, n in enumerate(xn)}
    common = [k for k in ad if k in xd]
    if not common:
        return None, 0
    return _mre([ad[k] for k in common], [xd[k] for k in common]), len(common)


def _copasi_final_species(xml, t0, t1, npnt, rtol, atol):
    """Final SPECIES concentrations from a COPASI deterministic time course, read
    from the model's metabolite objects (exactly the N species — the raw time series
    also carries compartments and global quantities, which would corrupt an
    order-independent comparison). Returns a 1-D array of length N."""
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
        return np.array([metabs.get(i).getConcentration() for i in range(metabs.size())], float)
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
    """COPASI's final species concentrations over *span*."""
    t0, t1, n = span
    return _copasi_final_species(xml, t0, t1, n, rtol, atol)


def _attempt(fn, *args):
    """``(fn(*args), None)``, or ``(None, why)`` where it raises."""
    try:
        return fn(*args), None
    except Exception as e:  # noqa: BLE001 - an engine's failure is recorded, not raised
        return None, f"{type(e).__name__}: {e}"


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

        span = (t0, t1, npnt)

        # --- At the timing tolerances --------------------------------------------
        timed, err = _attempt(_bngsim_final, xml, span, rtol, atol)
        if timed is None:
            rec["error"] = f"bngsim from_sbml: {err}"
            results.append(rec)
            print(f"#{row} {label:16} bngsim FAIL: {err}")
            continue
        bn_names, bn = timed
        rec.update({"n_species": int(bn.shape[1]), "agree_tol": AGREE_TOL})
        rr_timed, err = _attempt(_roadrunner_final, xml, span, rtol, atol)
        if rr_timed is None:
            rec["roadrunner_error"] = err
        cop_timed, err = _attempt(_copasi_final, xml, span, rtol, atol)
        if cop_timed is None:
            rec["copasi_error"] = err
        # Kept from earlier reports: each engine against bngsim at these tolerances.
        if rr_timed is not None:
            rr_name, ncommon = _mre_by_name(bn_names, bn, *rr_timed)
            rec.update(
                {
                    "bngsim_vs_roadrunner_sorted_mre": _sorted_mre(bn[-1], rr_timed[1][-1]),
                    "bngsim_vs_roadrunner_name_mre": rr_name,
                    "n_common_species": ncommon,
                }
            )
        if cop_timed is not None:
            rec["bngsim_vs_copasi_sorted_mre"] = _sorted_mre(bn[-1], cop_timed)

        # --- The converged reference, and each engine at its tolerances ----------
        tight = {}
        for engine, fn in (
            ("roadrunner", _roadrunner_final),
            ("bngsim", _bngsim_final),
            ("copasi", _copasi_final),
        ):
            got, err = _attempt(fn, xml, span, REF_RTOL, REF_ATOL)
            if got is None:
                rec[f"{engine}_reference_tolerance_error"] = err
            else:
                tight[engine] = got if engine == "copasi" else got[1][-1]
        ref_engine = next((e for e in ("roadrunner", "bngsim") if e in tight), None)
        rec["reference"] = (
            {"engine": ref_engine, "rtol": REF_RTOL, "atol": REF_ATOL} if ref_engine else None
        )
        if ref_engine is None:
            rec["agree"] = None
            results.append(rec)
            print(f"#{row} {label:16} no converged reference")
            continue
        ref = tight[ref_engine]
        # Same computation: every engine that completes, converged, on the reference.
        converged = {e: _sorted_mre(v, ref) for e, v in tight.items() if e != ref_engine}
        for e, v in converged.items():
            rec[f"{e}_vs_reference_converged_mre"] = v
        # Accuracy at the timed settings: reported, not judged.
        for e, final in (
            ("bngsim", bn[-1]),
            ("roadrunner", rr_timed[1][-1] if rr_timed is not None else None),
            ("copasi", cop_timed),
        ):
            if final is not None:
                rec[f"{e}_error_at_timing_tolerances"] = _sorted_mre(final, ref)
        rec["agree"] = all(v <= AGREE_TOL for v in converged.values()) if converged else None
        rec["worst_converged_mre"] = max(converged.values()) if converged else None
        tag = ("AGREE" if rec["agree"] else "DISAGREE") if rec["agree"] is not None else "n/a"

        print(
            f"#{row} {label:16} converged vs {ref_engine}: "
            + "  ".join(f"{e}={_show(rec, f'{e}_vs_reference_converged_mre')}" for e in converged)
            + f"  {tag};  at the timing tolerances: "
            f"bngsim {_show(rec, 'bngsim_error_at_timing_tolerances')}  "
            f"RR {_show(rec, 'roadrunner_error_at_timing_tolerances')}  "
            f"COPASI {_show(rec, 'copasi_error_at_timing_tolerances')}"
        )
        results.append(rec)

    doc = {
        "_meta": {
            "suite": "ode_engines_s4_sbml",
            "description": (
                "Cross-engine trajectory agreement for the Table S4 models, against "
                "a converged reference (issue #1250): each engine at the reference "
                "tolerances must be within agree_tol of it (agree), and each "
                "engine's error at the figure's tolerances is reported "
                "(*_error_at_timing_tolerances). Final state, order-independent "
                "(sorted species multiset). bngsim_vs_* are the engines' "
                "differences from bngsim at the timing tolerances, as earlier "
                "reports gave them."
            ),
            "reference": {
                "engines": ["libRoadRunner", "bngsim where libRoadRunner fails"],
                "rtol": REF_RTOL,
                "atol": REF_ATOL,
            },
            "engines": ["bngsim Model.from_sbml", "libRoadRunner", "COPASI"],
            "metric": "sorted final-state species multiset, max relative error",
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
