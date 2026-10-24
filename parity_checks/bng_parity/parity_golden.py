#!/usr/bin/env python3
"""Emit ``_core`` GOLDEN references for the bng_parity suite (bngsim-only).

Golden references are the per-job bngsim reference data downstream consumers
(PyBNF, PyBioNetGen) regenerate through their own bridges to verify they
reproduce bngsim, *without* needing a legacy-BNG / RoadRunner / AMICI install
for comparison. This is pure bngsim output — no differ, no subprocess reference
side. (The timing+parity matrix — ``bng_ode_run.py`` / ``bng_stoch_run.py`` — is
what runs the reference engine and compares; this file only fingerprints bngsim.)

Contract (the four locked decisions)
-------------------------------------
* **D1 — artifact(s):** ``.gdat`` (named observables) is canonical and always
  golden when present; ``.scan`` (parameter_scan output) is golden alongside it
  when present; ``.cdat`` (species) is a FALLBACK used only when a job emits no
  ``.gdat`` and no ``.scan`` (so there is always something to fingerprint). A
  job emitting several files (multi-suffix / multi-phase models, e.g. an
  equilibration + main run on different time grids) is covered file-by-file:
  the job checksum spans all its files, and the fingerprint carries one
  per-file entry under ``artifacts``.
* **D2 — stochastic basis:** a single fixed ``seed=1`` trajectory (the
  manifest ``DEFAULT_SEED``), NOT an ensemble. The checksum is then a true
  byte-identity key within a pinned (version, platform, seed) cell — the
  strongest possible check for a consumer verifying its bridge on a pinned
  bngsim wheel. (bngsim's C++ re-seeds deterministically, so same model state
  + ``seed=1`` reproduces a trajectory exactly; see GH #10 seed semantics.)
* **D3 — full-trajectory subset:** only the hand-picked ``TRAJECTORY_ALLOWLIST``
  below (each entry with a one-line reason) gets its full ``(time, values,
  names)`` written under ``golden/trajectories/``; every other job gets
  checksum + fingerprint only. Allow-listed jobs that were not part of the
  consumed sweep are simply emitted without a trajectory (no error).
* **D4 — mechanic:** this lean separate script consumes an existing bngsim
  sweep ``--sweep-out`` (or runs its own bngsim-only sweep via
  ``parity_sweep.py --simulator bngsim --n-seeds 1``) and writes
  ``golden/golden.json`` + the trajectory subset.

BNG2.pl is required
-------------------
bngsim has no BNGL parser: for a BNGL/.net job it shells out to BNG2.pl to
generate the reaction network, then simulates that network in-process. So even
though golden generation never *compares* against legacy BNG, it still needs
``$BNGPATH`` (or ``$BNG2_PL``) set for network generation. No path is hardcoded.

Usage
-----
    export BNGPATH=/path/to/BioNetGen-2.9.3
    # run a fresh bngsim-only seed=1 sweep of the fast tier, then golden it:
    python parity_golden.py --include fast/ --workers 3
    # or reuse an existing parity_sweep bngsim output tree:
    python parity_golden.py --sweep-out runs/core/bngsim --no-run-sweep

Verify (issue #582)
-------------------
``--verify`` re-runs the selected jobs the same way and compares them with the
committed ``golden.json`` instead of writing it: a job whose checksum matches is
identical; one whose checksum differs passes if every artifact's fingerprint is
within ``--verify-rtol``/``--verify-atol`` of its golden on ``t_last`` and every
per-variable stat (``|x - y| <= rtol·|y| + atol``,
``_core.fingerprint.fingerprint_distance``, the cross-platform comparison), and
is drift otherwise. A job that failed to
run, produced nothing to fingerprint, or has no golden entry fails too, and so
does a run that compared fewer than ``--min-compared`` jobs. Exit 1 on any of
those. This is the gate CI runs (``golden-verify.yml``) over the deterministic
allow-listed models, each of which must be compared::

    python parity_golden.py --verify --gate

Only the jobs the sweep ran (its ``_summary.json``) are judged. A sweep root
keeps every earlier run's output, and a stale job directory is not this run.

A PR that changes these numbers on purpose regenerates their entries with
``--merge --gate``, which replaces only the entries of the jobs the sweep ran
and records the regeneration in ``_meta.regenerations``. Regenerate only from
output that agrees with BNG2.pl (the nightly ``bng_ode`` baseline), so the gate
defends a cross-validated answer and not whatever bngsim last printed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves

from _core import (  # noqa: E402
    Golden,
    read_golden,
    read_manifest,
    require_bng,
    versions,
    write_golden,
)
from _core.fingerprint import (  # noqa: E402
    CHECKSUM_SIGFIGS,
    _round_sig,
    fingerprint_distance,
    golden_pair,
)
from _core.versions import git_rev  # noqa: E402

SWEEP = HERE / "parity_sweep.py"
DEFAULT_SEED = 1  # manifest _meta.defaults.seed; the pinned stochastic trajectory
MAX_WORKERS_SAFE = 4  # this 6-core / 16 GB box froze before; never exceed.


def _scrub_home(p: str) -> str:
    """Replace the absolute home-dir prefix with ``~`` so provenance paths written
    into the committed golden ``_meta`` stay machine- and username-independent for
    public release. Applied to the reference paths only, never to fingerprints."""
    home = str(Path.home())
    return "~" + p[len(home) :] if p.startswith(home) else p


def _repo_relpath(p: str) -> str:
    """Repo-relative path for the committed golden ``_meta`` (machine- and
    layout-independent for public release); home-scrubbed absolute fallback for
    paths outside the repo tree (e.g. a scratch ``sweep_out``)."""
    try:
        return str(Path(p).resolve().relative_to(HERE.parents[1]))
    except ValueError:
        return _scrub_home(p)


# Hand-picked representative subset that gets a FULL committed trajectory (D3).
# Each entry: model_id -> reason. Spans single/multi-artifact, oscillatory, and
# (when those tiers are swept) ssa/nf. Entries not present in the consumed sweep
# are emitted without a trajectory rather than erroring, so the list can be
# aspirational ahead of the slow/glacial stochastic golden run.
TRAJECTORY_ALLOWLIST: dict[str, str] = {
    # --- deterministic (ode), realized by the fast tier --------------------
    "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl": "small single-action ODE; the canonical smoke trajectory",
    "fast/rulehub/Published/Hat2016/Hat_2016.bngl": "3-suffix multi-phase ODE (equil/irrad/relax) on DIFFERENT time grids — exercises the multi-artifact path",
    "fast/rulehub/Published/Pekalski2013/Pekalski_2013.bngl": "two named suffixes (ode1/ode2) -> two .gdat files of different lengths",
    "fast/rulehub/Published/PyBioNetGen/core/polynomialground/polynomial_ground.bngl": "wt/mut paired runs -> two equal-grid .gdat files",
    "fast/rulehub/Examples/biology/brusselatoroscillator/brusselator-oscillator.bngl": "stiff limit-cycle oscillator — sharp features sensitive to integrator divergence",
    "fast/rulehub/Examples/biology/e2frbcellcycleswitch/e2f-rb-cell-cycle-switch.bngl": "bistable cell-cycle switch — a long flat-then-jump trajectory",
    "fast/rulehub/Tutorials/nfkb/nfkb.bngl": "multiple None-suffix actions overwriting one .gdat (last-writer-wins) — medium network",
    "fast/rulehub/Published/Lang2024/Lang_2024.bngl": "recent published multi-action ODE model",
    "fast/rulehub/Examples/biology/sonichedgehoggradient/sonic-hedgehog-gradient.bngl": "4-action gradient model — several observables",
    # --- stochastic (ssa / nf), realized only when those tiers are swept ----
    "original/bngl_models/my_models/ssa/oscillatory_system.bngl": "small SSA oscillator at seed=1 — discrete-event trajectory",
    "original/bngl_models/my_models/ssa/gk_simple.bngl": "minimal Goldbeter-Koshland SSA at seed=1",
    "slow/rulehub/Tutorials/NativeTutorials/ABCssa/ABC_ssa.bngl": "canonical A+B->C SSA tutorial at seed=1",
    "slow/rulehub/Tutorials/NativeTutorials/CircadianOscillator/CircadianOscillator.bngl": "larger SSA circadian oscillator at seed=1",
    "glacial/rulehub/Published/Goldstein1980/blbr_heterogeneity_goldstein1980.bngl": "NFsim parameter_scan -> .scan artifact at seed=1 — exercises scan + nf paths",
}

# What `--verify --gate` re-runs on every pull request (issue #582): the
# deterministic allow-listed models, whose fingerprints compare across platforms
# within a tolerance. A seed=1 stochastic trajectory reproduces only within one
# platform, and the slow/glacial tiers cost minutes each.
GATED_MODELS: list[str] = [m for m in TRAJECTORY_ALLOWLIST if m.startswith("fast/")]

# A gated model whose run differs between platforms by more than the default
# tolerance, with the measurement: model_id -> (rtol, atol, reason). `--verify`
# judges it by these instead of --verify-rtol / --verify-atol.
VERIFY_TOLERANCE_OVERRIDES: dict[str, tuple[float, float, str]] = {
    "fast/rulehub/Published/Lang2024/Lang_2024.bngl": (
        1e-4,
        1e-6,
        "integrates at rtol 1e-8 over t = 0..136472 (1200 rows of a cell cycle): "
        "Linux x86_64 against a macOS arm64 golden read 1.11 times rtol 1e-5 "
        "(about 1.1e-5 relative) on identical code, rounding carried over the "
        "long horizon. At 1e-4 that is 0.11 of it, and 0.15 in a later run on "
        "main two days on (the reading moves with the step sequence); the other "
        "gated models sit at 0.07 or less of 1e-5.",
    ),
}


def _load_parity_diff():
    """Import the sibling parity_diff module (for its tuned ``load_normalized``)."""
    spec = importlib.util.spec_from_file_location("parity_diff", HERE / "parity_diff.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _resolve_bngpath() -> str:
    """Resolve legacy-BNG (for BNGL network generation) via the shared resolver.

    bngsim has no BNGL parser, so a BNGL/.net job needs BNG2.pl to generate its
    network before bngsim simulates it in-process. ``_core.require_bng`` takes
    ``$BNG2_PL`` / ``$BNGPATH`` first and falls back to the BNG2.pl PyBioNetGen
    bundles (`uv sync --group parity`), so an env var is an override rather than
    the only way in; it exports ``$BNGPATH`` for the child sweep. Nothing is
    hardcoded, and a failure names every place it looked.
    """
    return str(
        require_bng(
            "golden generation never compares against legacy BNG, but bngsim still "
            "needs BNG2.pl to generate the reaction network from BNGL before it "
            "simulates in-process."
        ).root
    )


def _run_sweep(sweep_out: Path, jobs_path: Path, args) -> None:
    """Run a bngsim-only seed=1 sweep into ``sweep_out`` (D2: single fixed seed)."""
    cmd = [
        sys.executable,
        str(SWEEP),
        "--jobs",
        str(jobs_path),
        "--simulator",
        "bngsim",
        "--out",
        str(sweep_out),
        "--n-seeds",
        "1",  # D2: a single fixed seed=1 trajectory, no ensemble
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--regime",
        args.regime,
    ]
    if args.include:
        cmd += ["--include", args.include]
    if args.exclude:
        cmd += ["--exclude", args.exclude]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.models:
        cmd += ["--models", args.models]
    print(f"$ BNGPATH={os.environ['BNGPATH']} \\\n  {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"bngsim sweep failed (exit {proc.returncode})")


def _job_out_dir(sweep_out: Path, model_id: str, method: str) -> Path:
    """Locate a job's bngsim output dir.

    parity_sweep writes ``<sweep_out>/<relparent>/<stem>/{det|seedK}/``. A
    deterministic (ode) job lands in ``det``; a stochastic job's pinned seed=1
    run lands in ``seed1``. (When the consumed sweep was a multi-seed report run,
    ``seed1`` is still present — we use exactly that one.)

    The role is chosen from the manifest ``method``, but parity_sweep classifies
    the regime from the *model text* (``is_stochastic``), and the two can disagree
    for protocol/scan models — e.g. a ``parameter_scan`` the manifest tags ``ode``
    that the sweep nonetheless seeds and writes under ``seed1/``. So we prefer the
    method-implied role but fall back to whichever role the sweep actually
    produced, rather than silently dropping such a job from the golden set.
    """
    rel = Path(model_id)
    base = sweep_out / rel.parent / rel.stem
    primary = "det" if method != "stochastic" else f"seed{DEFAULT_SEED}"
    alt = f"seed{DEFAULT_SEED}" if primary == "det" else "det"
    primary_dir = base / primary
    if primary_dir.is_dir():
        return primary_dir
    alt_dir = base / alt
    if alt_dir.is_dir():
        return alt_dir
    return primary_dir  # neither exists; caller records out_dir_missing


def _run_status(out_dir: Path) -> str | None:
    """The sweep's recorded run status for a job ('ok'/'crash'/…), or None.

    parity_sweep writes a ``_run.log`` per unit with a ``# status=<s>`` header.
    A crashed/timed-out run can leave partial output (e.g. a multi-action model
    killed mid-write on its trailing action), and that partial trajectory is
    NOT byte-reproducible — a consumer whose timeout lands a step earlier gets a
    different file. So golden never fingerprints non-ok runs (see the crash
    guard in the main loop). None means the log is absent/unreadable (older
    sweeps): treat as ok and fall back to artifact-level checks.
    """
    log = out_dir / "_run.log"
    if not log.is_file():
        return None
    try:
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("# status="):
                return line.split("=", 1)[1].strip()
            if not line.startswith("#"):
                break  # header ends at the first non-comment line
    except OSError:
        return None
    return None


def _read_protocol_prefix(out_dir: Path) -> dict | None:
    """The per-model multi-phase replay verdict the sweep recorded, or None (GH #179).

    parity_sweep writes ``_provenance.json`` (``{engine, method, track,
    protocol_prefix}``) next to a bngsim run's artifacts. The ``protocol_prefix`` block
    carries ``dirty_carryover`` / ``replayed`` / ``segments`` / ``reinit_ics`` /
    ``replay_error`` — whether this job ran a faithful full-protocol replay (option-2),
    a single-segment direct run, or a graceful option-1 fallback. None when absent
    (a non-bngsim or older sweep, or a single-phase job that recorded no prefix).
    """
    prov = out_dir / "_provenance.json"
    if not prov.is_file():
        return None
    try:
        return json.loads(prov.read_text()).get("protocol_prefix")
    except (OSError, ValueError):
        return None


def _select_artifacts(out_dir: Path) -> tuple[list[Path], str]:
    """Per D1: ``.gdat`` + ``.scan`` if present, else ``.cdat`` fallback.

    Returns ``(files, reason)``; ``files`` empty with a reason when nothing is
    fingerprintable (job dir missing, or only non-data artifacts like ``.net``).
    """
    if not out_dir.is_dir():
        return [], "out_dir_missing"
    gdats = sorted(out_dir.glob("*.gdat"))
    scans = sorted(out_dir.glob("*.scan"))
    primary = gdats + scans
    if primary:
        return primary, ""
    cdats = sorted(out_dir.glob("*.cdat"))  # fallback only with no gdat/scan
    if cdats:
        return cdats, ""
    return [], "no_gdat_cdat_scan"


def _split(data: np.ndarray, names) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A loaded ``(n_time, 1+n_var)`` array -> (time, values, var-names)."""
    time = data[:, 0]
    values = data[:, 1:]
    if names is not None and len(names) == data.shape[1]:
        vnames = list(names[1:])
    else:  # positional fallback (header unavailable / width mismatch)
        vnames = [f"col{i}" for i in range(1, data.shape[1])]
    return time, values, vnames


def _build_from_files(files: list[Path], load_normalized):
    """Checksum/fingerprint/trajectories for a fixed file set, or (None, reason)."""
    per_file_checksum: dict[str, str] = {}
    artifacts: dict[str, dict] = {}
    trajectories: dict[str, dict] = {}
    for f in files:
        data, names, err = load_normalized(str(f))
        if err or data is None:
            return None, f"load_error:{f.name}:{err}"
        if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] == 0:
            return None, f"degenerate:{f.name}:shape={getattr(data, 'shape', None)}"
        time, values, vnames = _split(data, names)
        cs, fp = golden_pair(time, values, vnames)
        per_file_checksum[f.name] = cs
        artifacts[f.name] = fp
        # Store the trajectory rounded to the checksum's sig figs so the
        # committed reference matches exactly what the checksum hashes.
        trajectories[f.name] = {
            "time": _round_sig(np.asarray(time, float), CHECKSUM_SIGFIGS).tolist(),
            "names": vnames,
            "values": _round_sig(np.asarray(values, float), CHECKSUM_SIGFIGS).tolist(),
        }
    job_checksum = hashlib.sha256(
        "\n".join(f"{k}={per_file_checksum[k]}" for k in sorted(per_file_checksum)).encode()
    ).hexdigest()
    job_fingerprint = {
        "n_artifacts": len(artifacts),
        "kinds": sorted({Path(n).suffix.lstrip(".") for n in artifacts}),
        "artifacts": artifacts,
    }
    return (job_checksum, job_fingerprint, trajectories), None


def build_golden_for_job(out_dir: Path, load_normalized):
    """Build (checksum, fingerprint, trajectories) for one job, or (None, reason).

    ``checksum`` spans every selected file (sha256 over the sorted per-file
    checksums) so a byte divergence in *any* artifact is caught. ``fingerprint``
    nests one flat per-file fingerprint under ``artifacts`` — each directly
    comparable with ``_core.fingerprint.fingerprint_distance`` for the
    cross-platform fallback. ``trajectories`` maps filename -> rounded
    ``(time, names, values)`` for the allow-listed subset (caller decides whether
    to write it).

    Per D1, ``.cdat`` is the fallback when no ``.gdat``/``.scan`` is present. We
    also fall back to ``.cdat`` when the primary ``.gdat``/``.scan`` selection is
    *degenerate* (e.g. a time-only ``.gdat`` from a model with no observables):
    a network ODE/SSA run still writes a full species ``.cdat`` worth
    fingerprinting. (A network-free NFsim run writes no ``.cdat``, so such a job
    stays correctly skipped — there is genuinely nothing to fingerprint.)
    """
    files, reason = _select_artifacts(out_dir)
    if not files:
        return None, reason
    result, reason = _build_from_files(files, load_normalized)
    if result is None and reason.startswith("degenerate"):
        primary_names = {f.name for f in files}
        cdats = [c for c in sorted(out_dir.glob("*.cdat")) if c.name not in primary_names]
        if cdats:
            cdat_result, cdat_reason = _build_from_files(cdats, load_normalized)
            if cdat_result is not None:
                return cdat_result, None
    return result, reason


def _traj_relpath(model_id: str) -> str:
    """Suite-relative (under golden/) trajectory path for a model_id."""
    safe = model_id.replace("/", "__")
    if safe.endswith(".bngl"):
        safe = safe[: -len(".bngl")]
    return f"trajectories/{safe}.json"


def _downsample_traj(artifacts: dict, stride: int) -> dict:
    """Keep every ``stride``-th row of each artifact (the last row always kept)."""
    out = {}
    for fn, a in artifacts.items():
        n = len(a["time"])
        idx = list(range(0, n, stride))
        if idx and idx[-1] != n - 1:
            idx.append(n - 1)  # always preserve the final state
        out[fn] = {
            "time": [a["time"][i] for i in idx],
            "names": a["names"],
            "values": [a["values"][i] for i in idx],
        }
    return out


def _traj_payload(model_id, method, reason, artifacts, stride):
    payload = {
        "model_id": model_id,
        "method": method,
        "seed": DEFAULT_SEED if method == "stochastic" else None,
        "reason": reason,
        "sigfigs": CHECKSUM_SIGFIGS,
        "artifacts": _downsample_traj(artifacts, stride) if stride > 1 else artifacts,
    }
    if stride > 1:
        # The checksum/fingerprint in golden.json stay full-resolution; this
        # committed trajectory is a size-bounded debug reference only.
        payload["downsampled_stride"] = stride
    return json.dumps(payload, separators=(",", ":")) + "\n"


def job_fingerprint_distance(a: dict, b: dict, rtol: float, atol: float) -> float:
    """How far two job fingerprints are apart, in units of the tolerance: the
    worst ``_core.fingerprint.fingerprint_distance`` over the job's artifacts
    (``t_last`` and every per-variable stat). At most 1 is within it. inf when
    the two name different artifacts, or for any structural difference or
    one-sided blow-up inside one (issue #572)."""
    aa, ba = a.get("artifacts", {}), b.get("artifacts", {})
    if set(aa) != set(ba):
        return float("inf")
    return max((fingerprint_distance(aa[n], ba[n], rtol, atol) for n in aa), default=0.0)


def _nonfinite_cells_differ(trajectories: dict, traj_path: Path) -> str | None:
    """Where a fresh run's non-finite cells differ from its committed
    trajectory's, the first such cell; None when they agree.

    The fingerprint's min/max/mean skip non-finite cells, so a NaN or inf
    anywhere but the last row reaches no stat: a run that blew up once
    mid-trajectory matched its golden's fingerprint (Lang_2024 with one NaN at
    row 290 read 0.03 of its tolerance). The checksum sees it, but across
    platforms the checksum already differs. An allow-listed job's committed
    trajectory keeps every cell, or every stride-th row and the fresh run is cut
    the same way, and a blow-up is no rounding difference: its cells must match
    exactly, NaN for NaN and an inf of the same sign for an inf. Columns are
    matched by name, the time column included, and a row is reported by its
    index in the run."""
    payload = json.loads(traj_path.read_text())
    stride = payload.get("downsampled_stride") or 1
    fresh = _downsample_traj(trajectories, stride) if stride > 1 else trajectories
    for fn, ref in payload["artifacts"].items():
        if fn not in fresh:
            return f"{fn}: not in this run"
        run = fresh[fn]
        if sorted(run["names"]) != sorted(ref["names"]):
            return f"{fn}: other variables than the golden's"
        order = [run["names"].index(n) for n in ref["names"]]
        a = np.column_stack([run["time"], np.asarray(run["values"], float)[:, order]])
        b = np.column_stack([ref["time"], np.asarray(ref["values"], float)])
        if a.shape != b.shape:
            return f"{fn}: {a.shape} cells against {b.shape}"
        rows = list(range(len(trajectories[fn]["time"])))
        if stride > 1:  # the rows _downsample_traj kept
            rows = rows[::stride] + ([rows[-1]] if (len(rows) - 1) % stride else [])
        names = ["time", *ref["names"]]
        for kind, test in (("NaN", np.isnan), ("+inf", np.isposinf), ("-inf", np.isneginf)):
            one_sided = test(a) != test(b)
            if one_sided.any():
                row, col = (int(i) for i in np.argwhere(one_sided)[0])
                side = "this run" if test(a)[row, col] else "the golden"
                return f"{fn}: {names[col]} row {rows[row]} is {kind} in {side} only"
    return None


def _swept_model_ids(sweep_summary: dict) -> set[str] | None:
    """The model_ids the consumed sweep ran, from its ``_summary.json``, or None
    for a sweep that recorded no per-unit results.

    A sweep root accumulates: a run leaves every earlier run's output in place,
    so the default ``runs/golden/bngsim`` can hold a full 895-job sweep under a
    nine-model re-run. A full generation reads every job present on purpose
    (golden.json accumulates across staged runs). ``--verify`` and ``--merge``
    must judge, or replace, only what this sweep ran."""
    root, results = sweep_summary.get("root"), sweep_summary.get("results")
    if not root or not isinstance(results, list):
        return None
    swept = set()
    for r in results:
        try:
            swept.add(Path(r["bngl"]).relative_to(root).as_posix())
        except (KeyError, TypeError, ValueError):
            continue
    return swept


def verify_against_golden(
    golden_path: Path,
    sweep_out: Path,
    jobs,
    load_normalized,
    rtol: float,
    atol: float,
    min_compared: int,
    *,
    swept: set[str] | None = None,
    expected=(),
) -> int:
    """Compare a fresh sweep with the committed golden (issue #582). Returns the
    exit code: 0 when every job the sweep ran is identical or within
    ``rtol``/``atol`` and at least ``min_compared`` were compared, else 1.

    ``swept`` names the model_ids the sweep ran (``_swept_model_ids``); output
    under ``sweep_out`` for any other job is an earlier run's, so it is neither
    judged nor counted toward ``min_compared``. None judges every job present.
    Each model_id in ``expected`` (the gated set) must be compared, and one the
    sweep left no output for fails as NOT_SWEPT, by name. A job with a committed
    trajectory (every gated one) must also put its non-finite cells where the
    trajectory has them (``_nonfinite_cells_differ``)."""
    committed = {g["model_id"]: g for g in json.loads(golden_path.read_text())["golden"]}
    rows: list[tuple[str, str, str]] = []
    failed = 0
    compared = 0
    for job in jobs:
        if swept is not None and job.model_id not in swept:
            continue  # not this sweep's: any output there is stale
        out_dir = _job_out_dir(sweep_out, job.model_id, job.method)
        if not out_dir.is_dir():
            continue  # not selected for this sweep (an expected job is caught below)
        status = _run_status(out_dir)
        if status is not None and status != "ok":
            rows.append((job.model_id, "RUN_FAILED", status))
            failed += 1
            continue
        built, reason = build_golden_for_job(out_dir, load_normalized)
        if built is None:
            rows.append((job.model_id, "NOTHING_TO_COMPARE", reason))
            failed += 1
            continue
        golden = committed.get(job.model_id)
        if golden is None:
            rows.append((job.model_id, "NO_GOLDEN", "no entry in golden.json"))
            failed += 1
            continue
        compared += 1
        job_rtol, job_atol = VERIFY_TOLERANCE_OVERRIDES.get(job.model_id, (rtol, atol))[:2]
        tol = f"rtol {job_rtol:g}, atol {job_atol:g}"
        checksum, fingerprint, trajectories = built
        if checksum == golden["checksum"]:
            rows.append((job.model_id, "IDENTICAL", ""))
            continue
        worst = job_fingerprint_distance(fingerprint, golden["fingerprint"], job_rtol, job_atol)
        blowup = None
        traj_path = golden_path.parent / golden["trajectory"] if golden.get("trajectory") else None
        if traj_path is not None and not traj_path.is_file():
            rows.append((job.model_id, "NO_GOLDEN", f"{golden['trajectory']} is missing"))
            failed += 1
            continue
        if worst <= 1.0 and traj_path is not None:
            blowup = _nonfinite_cells_differ(trajectories, traj_path)
        if blowup:
            rows.append((job.model_id, "DRIFT", f"a blow-up no stat sees: {blowup}"))
            failed += 1
        elif worst <= 1.0:
            rows.append((job.model_id, "WITHIN_TOL", f"{worst:.2g} of the tolerance ({tol})"))
        else:
            rows.append((job.model_id, "DRIFT", f"{worst:.3g} times the tolerance ({tol})"))
            failed += 1
    judged = {r[0] for r in rows}
    for model_id in expected:
        if model_id not in judged:
            rows.append((model_id, "NOT_SWEPT", "the sweep left no output for it"))
            failed += 1
    width = max((len(r[0]) for r in rows), default=10)
    for model_id, verdict, note in rows:
        print(f"{model_id:<{width}}  {verdict:<18} {note}")
    print(
        f"\ngolden verify: {compared} compared, {failed} failing "
        f"(rtol {rtol:g}, atol {atol:g}; "
        f"{len(VERIFY_TOLERANCE_OVERRIDES)} model(s) by override)"
    )
    if compared < min_compared:
        print(
            f"FAIL: compared {compared} < --min-compared {min_compared}: the gate checked "
            "too little to mean anything"
        )
        return 1
    return 1 if failed else 0


def _protocol_replay_summary(protocol_replay: dict[str, dict]) -> dict:
    """Roll the per-model multi-phase replay verdicts up (GH #179)."""
    return {
        "models": len(protocol_replay),
        "dirty_carryover": sum(1 for v in protocol_replay.values() if v.get("dirty_carryover")),
        "replayed": sum(1 for v in protocol_replay.values() if v.get("replayed")),
        "fallback": sum(
            1
            for v in protocol_replay.values()
            if v.get("dirty_carryover") and not v.get("replayed")
        ),
        "reinit_ics": sum(int(v.get("reinit_ics") or 0) for v in protocol_replay.values()),
    }


def merge_regeneration(
    old_meta: dict,
    old_golden: list[Golden],
    fresh: list[Golden],
    meta: dict,
    protocol_replay: dict[str, dict],
) -> tuple[dict, list[Golden]]:
    """A partial regeneration (``--merge``): the ``fresh`` entries replace the
    ones they name, every other entry stays as the full generation in
    ``old_meta`` wrote it, and what was redone, by what, is recorded in
    ``_meta.regenerations`` beside that generation rather than over it.

    ``meta`` is this run's would-be ``_meta``, the provenance of the record. The
    regenerated models' multi-phase replay verdicts (GH #179) are this run's in
    ``protocol_replay``: each replaces the old one, or drops it where this run
    recorded none, and the roll-up is recounted."""
    by_id = {g.model_id: g for g in fresh}
    golden = sorted(
        [g for g in old_golden if g.model_id not in by_id] + list(by_id.values()),
        key=lambda g: g.model_id,
    )
    replay = {k: v for k, v in (old_meta.get("protocol_replay") or {}).items() if k not in by_id}
    replay.update({k: v for k, v in protocol_replay.items() if k in by_id})
    regenerations = list(old_meta.get("regenerations") or [])
    regenerations.append(
        {
            "date": _dt.date.today().isoformat(),
            "git_rev": meta["git_rev"],
            "versions": meta["versions"],
            "bionetgen_commit": meta["bionetgen_commit"],
            "bngsim_build_commit": meta["bngsim_build_commit"],
            "models": sorted(by_id),
        }
    )
    merged = {
        **old_meta,
        "protocol_replay_summary": _protocol_replay_summary(replay),
        "protocol_replay": dict(sorted(replay.items())),
        "n_trajectories": sum(1 for g in golden if g.trajectory),
        "regenerations": regenerations,
    }
    return merged, golden


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs", default=str(HERE / "jobs.json"), help="_core manifest (jobs.json)")
    ap.add_argument(
        "--sweep-out",
        default="",
        help="bngsim sweep root to consume (the dir with the per-model trees + "
        "_summary.json). Defaults to <golden-dir>/../runs/golden/bngsim. Run a "
        "fresh seed=1 bngsim sweep into it unless --no-run-sweep.",
    )
    ap.add_argument(
        "--golden-dir",
        default=str(HERE / "golden"),
        help="Output dir for golden.json + trajectories/ (committed).",
    )
    ap.add_argument(
        "--no-run-sweep",
        action="store_true",
        help="Consume an existing --sweep-out instead of running a fresh sweep.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help=f"Concurrent sims for the sweep (clamped to {MAX_WORKERS_SAFE}; RAM-governed).",
    )
    ap.add_argument("--timeout", type=int, default=180, help="Per-model sweep timeout (s)")
    ap.add_argument(
        "--regime",
        choices=("all", "ode", "stochastic"),
        default="all",
        help="Restrict the sweep to deterministic (ode) or stochastic units — "
        "stage the fast ODE set apart from the slow, supervised stochastic set. "
        "(The fingerprint pass still covers every job already present in the "
        "shared --sweep-out, so golden.json accumulates across staged runs.)",
    )
    ap.add_argument("--include", default="", help="Substring path filter (subset / smoke)")
    ap.add_argument("--exclude", default="", help="Substring path filter — drop matches")
    ap.add_argument("--limit", type=int, default=0, help="Max jobs in the sweep (0=all)")
    ap.add_argument("--models", default="", help="Comma-separated basenames to restrict the sweep")
    ap.add_argument(
        "--max-traj-kb",
        type=int,
        default=950,
        help="Per-trajectory-file size cap (KB); rows are downsampled to fit "
        "(stays under the 1 MB check-added-large-files hook). golden.json keeps "
        "full-resolution checksum+fingerprint regardless.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Compare the jobs this sweep ran with the committed golden.json instead "
        "of writing it (issue #582); exit 1 on drift.",
    )
    ap.add_argument(
        "--verify-rtol",
        type=float,
        default=1e-5,
        help="--verify: relative tolerance on each fingerprint stat of a job whose "
        "checksum differs (|x - y| <= rtol·|y| + atol).",
    )
    ap.add_argument(
        "--verify-atol",
        type=float,
        default=1e-6,
        help="--verify: absolute tolerance on each fingerprint stat.",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="The CI gate's selection: the deterministic allow-listed models "
        "(GATED_MODELS), --regime ode, and --min-compared all of them.",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="Write mode: replace only the entries of the jobs this sweep ran (its "
        "_summary.json, not every job dir under --sweep-out) in the existing "
        "golden.json, keep every other entry, and record the regeneration in "
        "_meta.regenerations (a partial regeneration of a subset, e.g. the gated one).",
    )
    ap.add_argument(
        "--min-compared",
        type=int,
        default=1,
        help="--verify: fail when fewer jobs than this were compared.",
    )
    args = ap.parse_args()

    if args.verify and args.merge:
        ap.error("--verify compares and --merge writes: pass one")
    if args.gate and not (args.verify or args.merge):
        # A plain write keeps only the jobs present in the sweep, so from a fresh
        # --sweep-out it would write the nine gated entries and drop the rest.
        ap.error("--gate selects a subset: pass --verify, or --merge to regenerate just it")

    if args.gate:
        args.regime = "ode"
        args.models = ",".join(Path(m).stem for m in GATED_MODELS)
        args.min_compared = max(args.min_compared, len(GATED_MODELS))

    if args.workers > MAX_WORKERS_SAFE:
        print(f"WARN: --workers {args.workers} clamped to {MAX_WORKERS_SAFE} (memory safety).")
        args.workers = MAX_WORKERS_SAFE

    bngpath = _resolve_bngpath()
    jobs_path = Path(args.jobs).resolve()
    golden_dir = Path(args.golden_dir).resolve()
    sweep_out = (
        Path(args.sweep_out).resolve()
        if args.sweep_out
        else (HERE / "runs" / "golden" / "bngsim").resolve()
    )
    golden_path = golden_dir / "golden.json"
    if (args.merge or args.verify) and not golden_path.is_file():
        sys.exit(f"ABORT: --{'merge' if args.merge else 'verify'} needs {golden_path}.")

    if not args.no_run_sweep:
        _run_sweep(sweep_out, jobs_path, args)
    elif not (sweep_out / "_summary.json").exists():
        sys.exit(
            f"ABORT: --no-run-sweep but no sweep found at {sweep_out} (_summary.json missing)."
        )

    # Fold the sweep's BNGsim-backend provenance + per-job engine audit into the
    # golden so it RECORDS which PyBioNetGen/bngsim produced it and which engine each
    # job ran on. Neither layer can silently substitute the legacy stack anymore: the
    # sweep drives bngsim directly (run_bngsim_job — GH #175), and since the pin moved
    # to 43b09a5 the bridge itself raises instead of quietly routing an un-inspectable
    # model to BNG2.pl under simulator='bngsim' (RuleWorld/PyBioNetGen#109, GH #4).
    # See bngsim_backend.py. Best-effort: older sweeps lack these keys.
    sweep_summary = {}
    try:
        summary_file = sweep_out / "_summary.json"
        if summary_file.exists():
            sweep_summary = json.loads(summary_file.read_text())
    except Exception:
        sweep_summary = {}

    _meta, jobs = read_manifest(jobs_path)
    pd = _load_parity_diff()
    swept = _swept_model_ids(sweep_summary)
    if swept is None:
        if args.merge or args.gate:
            what = "--merge replaces" if args.merge else "--gate judges"
            sys.exit(f"ABORT: {what} the jobs the sweep ran, and {summary_file} lists none.")
        if args.verify:
            print(
                f"WARN: {summary_file} lists no swept jobs; judging the output under "
                f"{sweep_out} whichever run wrote it."
            )
    if args.gate:
        # The gated set, whatever else a consumed (--no-run-sweep) sweep ran.
        swept = set(GATED_MODELS) if swept is None else swept & set(GATED_MODELS)

    if args.verify:
        return verify_against_golden(
            golden_path,
            sweep_out,
            jobs,
            pd.load_normalized,
            args.verify_rtol,
            args.verify_atol,
            args.min_compared,
            swept=swept,
            expected=GATED_MODELS if args.gate else (),
        )

    golden: list[Golden] = []
    skipped: dict[str, str] = {}
    protocol_replay: dict[str, dict] = {}  # GH #179: per-model multi-phase replay verdict
    n_traj = 0
    traj_writes: list[tuple[Path, str]] = []  # written with golden.json, not before
    # --merge regenerates only what this sweep ran. The sweep root also holds
    # every earlier run's output, which a full generation reads on purpose.
    for job in [j for j in jobs if j.model_id in swept] if args.merge else jobs:
        out_dir = _job_out_dir(sweep_out, job.model_id, job.method)
        status = _run_status(out_dir)
        if status is not None and status != "ok":
            # Crashed/timed-out run: any output is partial and not byte-
            # reproducible. Skip rather than fingerprint a fragile reference.
            skipped[job.model_id] = f"run_{status}"
            continue
        built, reason = build_golden_for_job(out_dir, pd.load_normalized)
        if built is None:
            # out_dir_missing just means this job wasn't in the consumed sweep
            # (filtered subset) — not a failure; record it quietly.
            skipped[job.model_id] = reason
            continue
        checksum, fingerprint, trajectories = built
        traj_rel = None
        if job.model_id in TRAJECTORY_ALLOWLIST:
            traj_rel = _traj_relpath(job.model_id)
            traj_path = golden_dir / traj_rel
            reason = TRAJECTORY_ALLOWLIST[job.model_id]
            # Bound each committed trajectory below the large-file hook cap by
            # downsampling rows if needed (golden.json stays full-resolution).
            stride = 1
            text = _traj_payload(job.model_id, job.method, reason, trajectories, stride)
            cap_bytes = args.max_traj_kb * 1024
            while len(text.encode()) > cap_bytes and stride < 64:
                stride *= 2
                text = _traj_payload(job.model_id, job.method, reason, trajectories, stride)
            if len(text.encode()) > cap_bytes:
                print(
                    f"WARN: {job.model_id} trajectory still > {args.max_traj_kb}KB at stride {stride}"
                )
            elif stride > 1:
                print(
                    f"  downsampled {job.model_id} trajectory (stride {stride}) to fit {args.max_traj_kb}KB cap"
                )
            traj_writes.append((traj_path, text))
            n_traj += 1
        golden.append(
            Golden(
                model_id=job.model_id,
                method=job.method,
                checksum=checksum,
                fingerprint=fingerprint,
                trajectory=traj_rel,
            )
        )
        # GH #179: carry this job's multi-phase replay verdict into the golden _meta.
        pp = _read_protocol_prefix(out_dir)
        if pp is not None:
            protocol_replay[job.model_id] = pp

    golden.sort(key=lambda g: g.model_id)
    missing_only = sum(1 for r in skipped.values() if r == "out_dir_missing")
    real_skips = {k: v for k, v in skipped.items() if v != "out_dir_missing"}
    # Scrub the home-dir prefix off the backend's ``bionetgen_path`` (the one
    # machine-specific path carried up from the gitignored sweep _summary into the
    # committed golden _meta) — keep the rest of the backend provenance verbatim.
    backend = sweep_summary.get("bngsim_backend")
    if backend and backend.get("bionetgen_path"):
        backend = {**backend, "bionetgen_path": _scrub_home(backend["bionetgen_path"])}
    meta = {
        "suite": "bng_parity",
        "kind": "golden",
        "jobs_manifest": _repo_relpath(str(jobs_path)),
        "sweep_out": _repo_relpath(str(sweep_out)),
        "bngpath": _scrub_home(bngpath),
        "git_rev": git_rev(str(HERE)),
        "versions": versions.stamp(),  # bngsim + python + platform (no ref engine)
        # BNGsim-backend provenance + per-job engine audit, carried up from the
        # sweep's _summary.json. Each engine is pinned by an ARTIFACT identifier,
        # hoisted here beside the rest of the reproduction record:
        # ``bionetgen_commit`` for the bridge (reproduce with
        # requirements-pybionetgen.txt) and ``bngsim_build_commit`` for the engine
        # itself — the commit its compiled extension was built from. The bare
        # ``versions.bngsim`` below cannot do that job now that bngsim publishes to
        # PyPI: it bumps only at release, so every commit between two releases
        # reports the same string (GH #163; ``bngsim_backend.bngsim_install`` in the
        # block above says which artifact — index / wheel / editable — supplied it).
        # The sweep drives GENUINE bngsim directly (run_bngsim_job), so
        # ``engine_audit.by_track`` records the bngsim engine each job used and
        # ``engine_audit.unsupported`` lists any job bngsim could not run (skipped,
        # NOT silently run on the legacy stack) — see GH #175.
        "bngsim_backend": backend,
        "bionetgen_commit": (backend or {}).get("bionetgen_commit"),
        "bngsim_build_commit": (backend or {}).get("bngsim_build_commit"),
        "engine_audit": sweep_summary.get("engine_audit"),
        # GH #179: per-model multi-phase replay verdict — for each dirty_carryover job,
        # whether bngsim ran a faithful full-protocol replay (``replayed=True`` + the
        # segment count, ``reinit_ics`` for a #181 seed-species re-init) or fell back to
        # the option-1 single-segment best-effort (``replayed=False`` + ``replay_error``).
        # The ``protocol_replay_summary`` rolls these up; single-phase jobs are absent.
        "protocol_replay_summary": _protocol_replay_summary(protocol_replay),
        "protocol_replay": dict(sorted(protocol_replay.items())),
        "seed": DEFAULT_SEED,
        "checksum_sigfigs": CHECKSUM_SIGFIGS,
        "n_golden": len(golden),
        "n_manifest_jobs": len(jobs),
        "n_trajectories": n_traj,
        "n_not_in_sweep": missing_only,
        "contract": (
            "D1: .gdat canonical + .scan when present; .cdat fallback only with no "
            "gdat/scan. Multi-file jobs: checksum spans all files, fingerprint nests "
            "one flat per-file entry under 'artifacts' (each fingerprint_distance-"
            "comparable). D2: stochastic = single fixed seed=1 trajectory (byte-"
            "identity within a pinned version/platform/seed cell). D3: full "
            "trajectories only for the TRAJECTORY_ALLOWLIST subset, rounded to "
            f"{CHECKSUM_SIGFIGS} sig figs. Consumers regenerate through their bridge "
            "on a pinned bngsim and match the checksum byte-for-byte; the per-file "
            "fingerprint is the cross-platform tolerance fallback."
        ),
    }
    if real_skips:
        meta["skipped"] = dict(sorted(real_skips.items())[:50])
        print(f"WARN: {len(real_skips)} swept job(s) failed or produced nothing to fingerprint.")

    if args.merge:
        # Every job this sweep ran must come back with a fresh entry. One that
        # failed would keep its old entry, beside a record saying it was redone.
        fresh = {g.model_id for g in golden}
        unfresh = dict(skipped)
        if args.gate:
            unfresh |= {m: "not swept" for m in GATED_MODELS if m not in fresh | set(unfresh)}
        if unfresh:
            for model_id, why in sorted(unfresh.items()):
                print(f"  {model_id}: {why}")
            sys.exit(
                f"ABORT: --merge: {len(unfresh)} job(s) to regenerate produced no entry; "
                f"{golden_dir} is left as it was."
            )

    golden_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        old_meta, old_golden = read_golden(golden_path)
        meta, golden = merge_regeneration(old_meta, old_golden, golden, meta, protocol_replay)
    for traj_path, text in traj_writes:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(text)
    write_golden(golden_path, golden, meta=meta)
    print(f"\ngolden: {golden_path}")
    print(f"  golden records:   {len(golden)} / {len(jobs)} manifest jobs")
    print(f"  trajectories:     {n_traj} (allow-listed)")
    print(f"  not in sweep:     {missing_only}")
    print(f"  no-artifact skips: {len(real_skips)}")
    print(f"  generated:        {_dt.datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
