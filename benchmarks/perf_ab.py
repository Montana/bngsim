#!/usr/bin/env python3
"""Local wall-clock A/B of two bngsim builds on one machine (GH #702).

The nightly parity workflow catches efficiency regressions that change how much
work the solver does (its ``solver_stats`` counters are deterministic). It cannot
see a constant-factor slowdown -- an extra copy in the RHS, a lost optimization
flag, slower model loading -- because wall-clock time on a shared CI runner moves
by more than that from run to run. This script measures exactly that, on a machine
you control, by timing two builds against each other on the same models in
interleaved order.

    python benchmarks/perf_ab.py                              # origin/main vs HEAD
    python benchmarks/perf_ab.py --baseline v0.16.0 --candidate HEAD
    python benchmarks/perf_ab.py --candidate python:.venv/bin/python   # uncommitted work
    python benchmarks/perf_ab.py --quick                      # models up to 150 species

A side is a git ref (a wheel is built from that commit in a throwaway worktree,
once, and installed into its own venv under the cache directory) or
``python:<interpreter>`` (an environment that already has bngsim, e.g. the dev
venv with an editable build of uncommitted changes).

Every measurement is a fresh subprocess: load the ``.net``, build the Simulator,
one cold run, then warm runs (``model.reset()`` between them). The first repeat of
every (model, workload, side) is a discarded warmup that also fills that side's
codegen cache. Sides alternate order every repeat, so drift in machine speed hits
both equally.

Per model and metric the result is the ratio of medians (candidate / baseline)
with a bootstrap 95 % interval over repeats; a model is flagged SLOWER when the
ratio exceeds 1 + ``--threshold`` and the whole interval is above 1. Exit 1 if
anything is flagged. Solver work counters are compared too: a time change with
identical counters is a constant-factor change; one with different counters is
algorithmic (and the nightly would see it as well).

Leave the machine idle while it runs. Results go to ``--out`` (default: a
timestamped directory under the cache) as ``results.json`` and ``summary.md``.

Models: ``benchmarks/_dev/suite_ode.json`` -- 28 committed BNG networks, 2 to 3,744
species, with their horizons.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SUITE = HERE / "_dev" / "suite_ode.json"
CACHE = Path(os.environ.get("BNGSIM_PERF_AB_CACHE", Path.home() / ".cache" / "bngsim" / "perf-ab"))

WORKLOADS = ("ode", "codegen", "sens")
METRICS = ("load_sec", "setup_sec", "cold_sec", "warm_sec")
WORK_KEYS = ("n_steps", "n_rhs_evals", "n_jac_evals")

# Runs inside the side's interpreter. Only long-stable public API, so an older
# baseline build can run it too.
WORKER = r"""
import json, statistics, sys, time
import bngsim

spec = json.loads(sys.argv[1])
t0 = time.perf_counter()
model = bngsim.Model.from_net(spec["net"])
load_sec = time.perf_counter() - t0

kw = {"method": "ode"}
if spec["workload"] in ("codegen", "sens"):
    kw["codegen"] = True
if spec["workload"] == "sens":
    kw["sensitivity_params"] = list(model.param_names)[: spec["n_sens"]]
t0 = time.perf_counter()
sim = bngsim.Simulator(model, **kw)
setup_sec = time.perf_counter() - t0

run_kw = {"t_span": (0.0, spec["t_end"]), "n_points": spec["n_steps"] + 1}
t0 = time.perf_counter()
r = sim.run(**run_kw)
cold_sec = time.perf_counter() - t0
warm = []
for _ in range(spec["n_warm"]):
    model.reset()
    t0 = time.perf_counter()
    sim.run(**run_kw)
    warm.append(time.perf_counter() - t0)
stats = dict(r.solver_stats or {})
print(json.dumps({
    "load_sec": load_sec,
    "setup_sec": setup_sec,
    "cold_sec": cold_sec,
    "warm_sec": statistics.median(warm) if warm else None,
    "backend": getattr(sim, "codegen_backend", None),
    "work": {k: int(stats[k]) for k in ("n_steps", "n_rhs_evals", "n_jac_evals") if k in stats},
}))
"""


# The side's version and compiled-core identity; guarded so an older build that
# predates _build_provenance still reports its version.
IDENTITY = r"""
import json, bngsim
d = {"version": bngsim.__version__}
try:
    import bngsim._build_provenance as bp
    d["identity"] = bp.identity_line()
except Exception:
    d["identity"] = None
print(json.dumps(d))
"""


# --------------------------------------------------------------------------- #
# builds
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required to build and install a side from a git ref")
    return uv


def build_wheel(sha: str, python: str) -> Path:
    """The wheel for ``sha`` and interpreter ``python``, built once in a throwaway
    worktree and cached. The interpreter is explicit: left to itself ``uv build``
    picks whatever Python it finds first, and a cp313 wheel does not install into
    the 3.12 venv it is meant for."""
    out = CACHE / "wheels" / sha[:12] / f"py{python}"
    have = sorted(out.glob("bngsim-*.whl"))
    if have:
        return have[-1]
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bngsim-src-") as tmp:
        src = Path(tmp) / "src"
        _git("worktree", "add", "--detach", str(src), sha)
        try:
            print(f"  building a wheel from {sha[:12]} (a C++ compile; minutes)", flush=True)
            r = subprocess.run(
                [_uv(), "build", "--wheel", "--python", python, "--out-dir", str(out), str(src)]
            )
            if r.returncode != 0:
                raise SystemExit(f"wheel build failed for {sha[:12]} (rc={r.returncode})")
        finally:
            _git("worktree", "remove", "--force", str(src))
    have = sorted(out.glob("bngsim-*.whl"))
    if not have:
        raise SystemExit(f"the build wrote no wheel into {out}")
    return have[-1]


def ensure_env(sha: str, python: str) -> Path:
    """A venv under the cache holding the wheel built from ``sha``."""
    venv = CACHE / "venvs" / f"{sha[:12]}-py{python}"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    done = venv / ".perf_ab_installed"  # only a completed install is reused
    if done.exists():
        return py
    wheel = build_wheel(sha, python)
    subprocess.run([_uv(), "venv", "--clear", "--python", python, str(venv)], check=True)
    subprocess.run([_uv(), "pip", "install", "--python", str(py), str(wheel)], check=True)
    done.write_text(wheel.name + "\n")
    return py


def resolve_side(spec: str, python: str) -> dict:
    """``REF`` or ``python:<interpreter>`` -> {label, python, identity}."""
    if spec.startswith("python:"):
        py = Path(spec[len("python:") :]).expanduser()
        if not py.is_absolute():
            py = (Path.cwd() / py).resolve()
        label = str(py)
    else:
        sha = _git("rev-parse", "--verify", f"{spec}^{{commit}}")
        py = ensure_env(sha, python)
        label = f"{spec} ({sha[:12]})"
    ident = subprocess.run([str(py), "-c", IDENTITY], capture_output=True, text=True)
    identity = (
        json.loads(ident.stdout) if ident.returncode == 0 else {"error": ident.stderr[-400:]}
    )
    return {"spec": spec, "label": label, "python": str(py), "identity": identity}


# --------------------------------------------------------------------------- #
# statistics (pure; unit-tested)
# --------------------------------------------------------------------------- #
def ratio_ci(
    a: list[float], b: list[float], *, n_boot: int = 2000, seed: int = 0, level: float = 0.99
) -> tuple[float, float, float]:
    """median(b) / median(a) and a bootstrap interval over repeats.

    99 % rather than 95 % by default: a run tests every (model, workload, metric),
    so at 95 % an A/A comparison of one build against itself flags a few of them.
    """
    ma, mb = statistics.median(a), statistics.median(b)
    point = mb / ma if ma > 0 else float("inf")
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ra = statistics.median(rng.choices(a, k=len(a)))
        rb = statistics.median(rng.choices(b, k=len(b)))
        if ra > 0:
            boots.append(rb / ra)
    if not boots:
        return point, point, point
    boots.sort()
    tail = (1 - level) / 2
    return point, boots[int(tail * (len(boots) - 1))], boots[int((1 - tail) * (len(boots) - 1))]


# Below this many repeats a bootstrap interval is too coarse to act on; the
# numbers are still reported.
MIN_REPEATS_TO_FLAG = 5


def verdict(
    point: float,
    lo: float,
    hi: float,
    threshold: float,
    *,
    a: float = 1.0,
    b: float = 1.0,
    min_delta: float = 0.0,
    n: int = MIN_REPEATS_TO_FLAG,
) -> str:
    """SLOWER / faster / "" for one metric.

    A flag needs all three: the ratio past the threshold, the whole interval on
    that side of 1, and an absolute change of at least ``min_delta`` seconds --
    a 10 % change in a 3 ms measurement is timer noise, not a regression.
    """
    if n < MIN_REPEATS_TO_FLAG or abs(b - a) < min_delta:
        return ""
    if point > 1 + threshold and lo > 1:
        return "SLOWER"
    if point < 1 / (1 + threshold) and hi < 1:
        return "faster"
    return ""


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0]
    return statistics.geometric_mean(xs) if xs else float("nan")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def measure(side: dict, spec: dict, timeout: float) -> dict:
    env = dict(os.environ)
    env["BNGSIM_CODEGEN_CACHE_DIR"] = str(CACHE / "codegen" / side["cache_key"])
    try:
        r = subprocess.run(
            [side["python"], "-c", WORKER, json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:g} s"}
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout)[-400:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def select_models(args) -> list[dict]:
    models = json.loads(SUITE.read_text())["models"]
    if args.models:
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m["name"] in want]
    if args.max_species:
        models = [m for m in models if m["species"] <= args.max_species]
    return models


def summarize(
    rows: list[dict], sides: list[dict], threshold: float, meta: dict
) -> tuple[str, int]:
    lines = [
        "# bngsim A/B",
        "",
        f"baseline: `{sides[0]['label']}` {sides[0]['identity']}",
        f"candidate: `{sides[1]['label']}` {sides[1]['identity']}",
        f"machine: {meta['machine']} · repeats {meta['repeats']} (+1 warmup) · warm runs {meta['n_warm']}",
        f"load average before / after: {meta['load_before']} / {meta['load_after']}",
        "",
        f"Ratio = candidate / baseline (median over repeats); SLOWER = above x{1 + threshold:.2f} "
        f"with the whole 99 % interval above 1 and at least {meta['min_delta_ms']:g} ms slower "
        f"(flags need >= {MIN_REPEATS_TO_FLAG} repeats).",
        "",
    ]
    n_slow = 0
    by_wl: dict[str, list[dict]] = {}
    for r in rows:
        by_wl.setdefault(r["workload"], []).append(r)
    for wl, rs in by_wl.items():
        lines += [
            f"## {wl}",
            "",
            "| model | species | metric | baseline s | candidate s | ratio | 99 % CI | | work |",
        ]
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            if r.get("error"):
                lines.append(
                    f"| {r['model']} | {r['species']} | error | | | | | | {r['error'][:80]} |"
                )
                continue
            for m in ("warm_sec", "cold_sec", "setup_sec", "load_sec"):
                s = r["stats"].get(m)
                if not s:
                    continue
                n_slow += s["verdict"] == "SLOWER"
                work = (
                    ""
                    if m != "warm_sec"
                    else ("same" if r["work_same"] else f"DIFFERS {r['work']}")
                )
                lines.append(
                    f"| {r['model']} | {r['species']} | {m} | {s['a']:.4g} | {s['b']:.4g} | "
                    f"{s['ratio']:.3f} | {s['lo']:.3f}-{s['hi']:.3f} | {s['verdict']} | {work} |"
                )
        for m in ("warm_sec", "cold_sec"):
            gm = geomean(
                [r["stats"][m]["ratio"] for r in rs if not r.get("error") and r["stats"].get(m)]
            )
            lines.append(f"\ngeometric-mean {m} ratio over {len(rs)} models: **{gm:.3f}**")
        lines.append("")
    lines.append(f"**{n_slow} metric(s) SLOWER.**" if n_slow else "No significant slowdown.")
    return "\n".join(lines) + "\n", n_slow


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--baseline", default="origin/main", help="git ref or python:<interpreter>")
    ap.add_argument("--candidate", default="HEAD", help="git ref or python:<interpreter>")
    ap.add_argument(
        "--workloads", default="ode,codegen", help=f"comma list of {','.join(WORKLOADS)}"
    )
    ap.add_argument(
        "--models", default="", help="comma list of suite_ode.json names (default all)"
    )
    ap.add_argument("--max-species", type=int, default=0, help="skip models larger than this")
    ap.add_argument(
        "--repeats", type=int, default=5, help="measured repeats per side (plus one warmup)"
    )
    ap.add_argument("--n-warm", type=int, default=3, help="warm runs inside each measurement")
    ap.add_argument(
        "--n-sens", type=int, default=5, help="sensitivity parameters for the sens workload"
    )
    ap.add_argument("--threshold", type=float, default=0.05, help="flag ratios above 1 + this")
    ap.add_argument(
        "--min-delta-ms",
        type=float,
        default=2.0,
        help="flag only changes of at least this many milliseconds (timer noise floor)",
    )
    ap.add_argument("--timeout", type=float, default=900.0, help="per-measurement wall cap (s)")
    ap.add_argument(
        "--python", default="3.12", help="interpreter version for venvs built from refs"
    )
    ap.add_argument("--quick", action="store_true", help="only models up to 150 species")
    ap.add_argument("--out", default="", help="output directory")
    args = ap.parse_args(argv)
    if args.quick:
        args.max_species = args.max_species or 150

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    bad = set(workloads) - set(WORKLOADS)
    if bad:
        ap.error(f"unknown workload(s): {sorted(bad)}")
    models = select_models(args)
    if not models:
        ap.error("no models selected")

    print("preparing the two sides", flush=True)
    sides = [resolve_side(args.baseline, args.python), resolve_side(args.candidate, args.python)]
    for i, s in enumerate(sides):
        s["cache_key"] = f"{i}-" + hashlib.sha1(s["label"].encode()).hexdigest()[:10]
        print(f"  {'AB'[i]}: {s['label']}  {s['identity']}", flush=True)
        if "error" in s["identity"]:
            raise SystemExit(f"side {'AB'[i]} cannot import bngsim: {s['identity']['error']}")

    out = (
        Path(args.out)
        if args.out
        else CACHE / "runs" / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)
    # Building a side's wheel just pegged every core; give the load a moment to fall
    # back before the first measurement rather than timing the tail of a compile.
    if hasattr(os, "getloadavg"):
        for _ in range(12):
            if os.getloadavg()[0] <= 0.5 * (os.cpu_count() or 1):
                break
            time.sleep(10)
    load_before = [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None
    if load_before and load_before[0] > 0.5 * (os.cpu_count() or 1):
        print(
            f"WARNING: load average {load_before} -- the machine is busy; timings will be noisy",
            flush=True,
        )

    cases = [(m, w) for m in models for w in workloads if not (w == "sens" and m["species"] > 400)]
    raw: dict[tuple, dict] = {}
    total = len(cases) * (args.repeats + 1) * 2
    done = 0
    for rep in range(args.repeats + 1):
        for m, w in cases:
            order = (0, 1) if rep % 2 == 0 else (1, 0)
            for i in order:
                spec = {
                    "net": str(HERE / m["net_file"]),
                    "t_end": float(m["t_end"]),
                    "n_steps": int(m["n_steps"]),
                    "workload": w,
                    "n_warm": args.n_warm,
                    "n_sens": args.n_sens,
                }
                res = measure(sides[i], spec, args.timeout)
                done += 1
                if rep == 0:
                    continue  # warmup: fills the codegen cache, pages in the libraries
                raw.setdefault((m["name"], w, i), {"runs": []})["runs"].append(res)
            print(f"\r  {done}/{total} measurements", end="", flush=True)
    print(flush=True)
    load_after = [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None

    rows = []
    for m, w in cases:
        runs = [raw.get((m["name"], w, i), {"runs": []})["runs"] for i in (0, 1)]
        errs = [r["error"] for side in runs for r in side if "error" in r]
        row = {"model": m["name"], "species": m["species"], "workload": w}
        if errs:
            row["error"] = errs[0]
            rows.append(row)
            continue
        row["stats"] = {}
        for metric in METRICS:
            a = [r[metric] for r in runs[0] if r.get(metric) is not None]
            b = [r[metric] for r in runs[1] if r.get(metric) is not None]
            if not a or not b:
                continue
            point, lo, hi = ratio_ci(a, b)
            ma, mb = statistics.median(a), statistics.median(b)
            row["stats"][metric] = {
                "a": ma,
                "b": mb,
                "ratio": point,
                "lo": lo,
                "hi": hi,
                "n": min(len(a), len(b)),
                "verdict": verdict(
                    point,
                    lo,
                    hi,
                    args.threshold,
                    a=ma,
                    b=mb,
                    min_delta=args.min_delta_ms / 1000.0,
                    n=min(len(a), len(b)),
                ),
            }
        wa, wb = runs[0][0].get("work"), runs[1][0].get("work")
        row["work_same"] = wa == wb
        row["work"] = {"baseline": wa, "candidate": wb}
        row["backend"] = [runs[0][0].get("backend"), runs[1][0].get("backend")]
        rows.append(row)

    meta = {
        "machine": f"{platform.node()} {platform.platform()} {platform.processor() or platform.machine()} "
        f"({os.cpu_count()} cpus)",
        "repeats": args.repeats,
        "n_warm": args.n_warm,
        "min_delta_ms": args.min_delta_ms,
        "load_before": load_before,
        "load_after": load_after,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    md, n_slow = summarize(rows, sides, args.threshold, meta)
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "sides": sides, "rows": rows}, indent=1)
    )
    (out / "summary.md").write_text(md)
    print(md)
    print(f"results: {out}")
    return 1 if n_slow else 0


if __name__ == "__main__":
    sys.exit(main())
