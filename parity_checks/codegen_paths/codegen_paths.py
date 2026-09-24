"""Codegen-path parity sweep over the committed BNGL corpus (issue #803, step 1).

Does the model-based codegen path (``prepare_model_codegen``, what SBML/Antimony/
builder models use) reproduce the interpreter on every BNGL model, with and
without forward sensitivities -- and how does it compare with the ``.net`` codegen
path that ``Model.from_net``/``from_bngl`` models take today? See README.md.

    python codegen_paths.py nets                  # BNG2.pl generate_network per model
    python codegen_paths.py plan                  # horizon + sensitivity parameters
    python codegen_paths.py run                   # every (model, arm) cell, first pass
    python codegen_paths.py report                # results.jsonl, summary.md, flagged.txt
    python codegen_paths.py run --tight           # re-run flagged models at tight tolerance
    python codegen_paths.py report                # again, now with the adjudication table

Every step resumes: a cell whose JSON exists is skipped, and a cell whose log
exists is taken to be in flight under another supervisor, so two ``run`` processes
can share one output directory. Each cell runs in its own process group and is
killed with everything under it (sympy, clang) at its arm's timeout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PC = HERE.parent
sys.path.insert(0, str(PC))
sys.path.insert(0, str(HERE))

from _metrics import (  # noqa: E402
    horizon,
    per_species_atol,
    pick_sens_params,
    richardson,
    sens_err,
    traj_err,
)

CORPUS = PC / "bng_parity" / "models"
JOBS = PC / "bng_parity" / "jobs.json"
ARMS = ("interp", "net", "model", "net_sens", "model_sens", "fd")
SENS_ARMS = ("net_sens", "model_sens", "fd")
TIMEOUT = {
    "interp": 900,
    "net": 900,
    "model": 900,
    "net_sens": 1800,
    "model_sens": 1800,
    "fd": 2400,
}
TRAJ_TOL = 1e-4  # plain / sensitivity-run trajectory vs the interpreter
SENS_TOL = 1e-3  # scaled sensitivity vs Richardson FD


def safe(model_id: str) -> str:
    return model_id.replace("/", "__")


def jl_read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def supervised(argv: list[str], log: Path, timeout: float, env: dict | None = None) -> int | None:
    """Run ``argv`` in its own process group; kill the whole group at ``timeout``.

    ``subprocess.run(timeout=)`` kills only the direct child and then blocks on a
    pipe an orphaned clang grandchild still holds, so it bounds nothing here.
    Returns the exit code, or None on a timeout.
    """
    with log.open("w") as fh:
        proc = subprocess.Popen(
            argv, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            return None


# ── nets ───────────────────────────────────────────────────────────────────

_NET_CHILD = r"""
import json, sys, time
from bngsim._bngl_loader import bngl_to_net
import bngsim
t0 = time.time()
net = bngl_to_net(sys.argv[1], timeout=int(sys.argv[2]))
gen = time.time() - t0
cd = bngsim.Model.from_net(str(net))._core.codegen_data()
print("ROW" + json.dumps({"net": str(net), "gen_sec": round(gen, 2), "n_species": len(cd["species"]),
    "n_reactions": len(cd["reactions"]), "n_params": len(cd["parameters"]),
    "n_functions": len(cd["functions"]), "n_observables": len(cd["observables"])}))
"""


def cmd_nets(a) -> None:
    """One ``bngl_to_net`` per model -- the network ``Model.from_bngl`` loads, from
    its content-addressed cache when present."""
    out = a.out / "nets.jsonl"
    done = {r["model_id"] for r in jl_read(out)}
    jobs = [j for j in json.loads(a.jobs.read_text())["jobs"] if j["model_id"] not in done]
    logs = a.out / "nets_logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} networks to generate ({len(done)} done)", flush=True)

    def one(job: dict) -> dict:
        mid = job["model_id"]
        row = {"model_id": mid, "methods": job["params"]["methods"]}
        log = logs / f"{safe(mid)}.log"
        t0 = time.time()
        rc = supervised(
            [sys.executable, "-c", _NET_CHILD, str(a.corpus / mid), str(a.gen_timeout)],
            log,
            a.gen_timeout + 120,
        )
        text = log.read_text(errors="replace")
        rows = [line[3:] for line in text.splitlines() if line.startswith("ROW")]
        if rows:
            row.update(json.loads(rows[-1]), status="OK")
        else:
            row.update(
                status="GEN_TIMEOUT" if rc is None or "timed out" in text else "GEN_FAIL",
                msg=text[-800:],
            )
        row["sec"] = round(time.time() - t0, 1)
        return row

    with out.open("a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row["status"] != "OK" or i % 50 == 0:
                print(
                    f"[{i}/{len(jobs)}] {row['status']:11s} {row['sec']:7.1f}s {row['model_id']}",
                    flush=True,
                )


# ── plan ───────────────────────────────────────────────────────────────────


def cmd_plan(a) -> None:
    import bngsim
    from bngsim._codegen import _should_chunk

    out = a.out / "plan.jsonl"
    have = {r["model_id"] for r in jl_read(out)}
    n = 0
    with out.open("a") as fh:
        for r in jl_read(a.out / "nets.jsonl"):
            if r["status"] != "OK" or r["model_id"] in have:
                continue
            t_end, n_points, src = horizon(a.corpus / r["model_id"])
            try:
                cd = bngsim.Model.from_net(r["net"])._core.codegen_data()
                P, fam = pick_sens_params(cd)
                rtypes = dict(Counter(x["type"] for x in cd["reactions"]))
            except Exception as e:  # noqa: BLE001 - recorded; the model runs without P
                P, fam, rtypes = [], {"plan_error": repr(e)}, {}
            fh.write(
                json.dumps(
                    {
                        **r,
                        "t_end": t_end,
                        "n_points": n_points,
                        "horizon_src": src,
                        "sens_params": P,
                        "families": fam,
                        "rxn_types": rtypes,
                        "chunked": _should_chunk(r["n_reactions"]),
                    }
                )
                + "\n"
            )
            n += 1
    print(f"planned {n} new models ({len(have) + n} total)")


# ── run ────────────────────────────────────────────────────────────────────


def write_meta(a) -> None:
    import bngsim
    from _core import versions
    from bngsim._bngl_loader import resolve_bng2

    try:
        bng2 = str(resolve_bng2())
    except Exception as e:  # noqa: BLE001
        bng2 = f"unresolved: {e}"
    meta = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "bngsim": getattr(bngsim, "__version__", None),
        "bngsim_build": bngsim.capabilities().get("build"),
        "git_rev": versions.git_rev(str(PC.parent)),
        "bng2_pl": bng2,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }
    (a.out / f"meta_{int(time.time())}.json").write_text(json.dumps(meta, indent=1, default=str))


def cmd_run(a) -> None:
    plans = jl_read(a.out / "plan.jsonl")
    cells_dir = a.out / ("tight" if a.tight else "cells")
    if a.only:
        plans = [p for p in plans if re.search(a.only, p["model_id"])]
    only_file = a.only_file or (a.out / "flagged.txt" if a.tight else None)
    if only_file:
        keep = set(Path(only_file).read_text().split())
        plans = [p for p in plans if p["model_id"] in keep]
    plans.sort(key=lambda p: (p["n_reactions"], p["n_species"]))
    arms = a.arms.split(",") if a.arms else [x for x in ARMS if not (a.tight and x == "net")]
    cells = [
        (p, arm) for p in plans for arm in arms if not (arm in SENS_ARMS and not p["sens_params"])
    ]
    write_meta(a)
    env = dict(os.environ, BNGSIM_CODEGEN_CACHE_DIR=str(a.cache or a.out / "cg"))
    env.setdefault("BNGSIM_CODEGEN_JOBS", "2")
    print(
        f"{len(cells)} cells over {len(plans)} models, {a.workers} workers -> {cells_dir}",
        flush=True,
    )

    def one(p: dict, arm: str) -> tuple[str, str, str, float]:
        d = cells_dir / safe(p["model_id"])
        d.mkdir(parents=True, exist_ok=True)
        try:  # claim: the .log doubles as the in-flight marker
            os.close(os.open(d / f"{arm}.log", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            return p["model_id"], arm, "SKIP", 0.0
        spec = {"t_end": p["t_end"], "n_points": p["n_points"], "sens_params": p["sens_params"]}
        if a.tight:
            import numpy as np

            first = a.out / "cells" / safe(p["model_id"]) / "interp.npz"
            atol = per_species_atol(np.load(first)["species"]) if first.exists() else [1e-12]
            spec.update(rtol=1e-10, atol=atol, fd_rtol=1e-12, fd_atol=[x * 1e-2 for x in atol])
        t0 = time.time()
        rc = supervised(
            [
                sys.executable,
                str(HERE / "_cell.py"),
                p["net"],
                arm,
                str(d / arm),
                json.dumps(spec),
            ],
            d / f"{arm}.log",
            TIMEOUT[arm] * a.timeout_scale,
            env,
        )
        dt = time.time() - t0
        if not (d / f"{arm}.json").exists():
            status = "TIMEOUT" if rc is None else f"CRASH rc={rc}"
            (d / f"{arm}.json").write_text(json.dumps({"arm": arm, "status": status, "wall": dt}))
            return p["model_id"], arm, status, dt
        return p["model_id"], arm, "done", dt

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, p, arm) for p, arm in cells]
        for i, fut in enumerate(as_completed(futs), 1):
            mid, arm, status, dt = fut.result()
            if status not in ("SKIP", "done") or dt > 60 or i % 100 == 0:
                print(
                    f"[{i}/{len(cells)} {time.time() - t0:6.0f}s] {status:9s} {dt:7.1f}s {arm:10s} {mid}",
                    flush=True,
                )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


# ── report ─────────────────────────────────────────────────────────────────


def load_cell(d: Path, arm: str) -> tuple[dict | None, dict]:
    import numpy as np

    j = d / f"{arm}.json"
    if not j.exists():
        return None, {}
    meta = json.loads(j.read_text())
    if meta.get("status") != "OK":
        # A killed cell still says where its time went.
        log = d / f"{arm}.log"
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                m = re.match(r"PHASE \S+ compile start \((\d+) bytes\)", line)
                if m:
                    meta["c_bytes_partial"] = int(m.group(1))
                m = re.match(r"PHASE \S+ run start \(setup ([\d.]+)s\)", line)
                if m:
                    meta["setup_sec_partial"] = float(m.group(1))
    npz = d / f"{arm}.npz"
    return meta, (dict(np.load(npz)) if npz.exists() else {})


def ok(m: dict | None) -> bool:
    return bool(m and m.get("status") == "OK")


def trajs(a: dict, ref: dict, keys: tuple[str, ...], abs_floor: float) -> float | None:
    vals = [traj_err(a[k], ref[k], abs_floor=abs_floor) for k in keys if k in a and k in ref]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


# Species and observables (linear in species) must agree to solver tolerance.
# Functions are reported apart: a time or state switch sampled exactly at its
# switching instant is legitimately left- or right-continuous, so a function-only
# difference is a different finding from a trajectory difference.
STATE = ("species", "obs")
FUNCS = ("expr",)


def model_row(p: dict, d: Path) -> dict:
    M, A = {}, {}
    for arm in ARMS:
        M[arm], A[arm] = load_cell(d, arm)
    r: dict = {
        k: p[k]
        for k in (
            "model_id",
            "n_species",
            "n_reactions",
            "n_functions",
            "chunked",
            "sens_params",
            "t_end",
        )
    }
    for arm in ARMS:
        m = M[arm]
        r[f"{arm}_status"] = None if m is None else m.get("status")
        if m and m.get("status") == "ERROR":
            r[f"{arm}_exc"] = f"{m.get('exc_type')}: {(m.get('exc') or '')[:300]}"
        if arm in ("net", "model", "net_sens", "model_sens") and m:
            if ok(m):
                if m.get("net_codegen_path", True):
                    took_its_path = bool(m.get("sim_net_path")) == arm.startswith("net")
                else:
                    # Since #803 step 3 both arms compile the built model: the
                    # control is that they built the same artifact.
                    twin = M[arm.replace("net", "model", 1) if arm.startswith("net") else arm]
                    took_its_path = bool(twin) and m.get("so") == twin.get("so")
                r[f"{arm}_path_ok"] = m.get("backend") == "cc" and took_its_path
                r[f"{arm}_codegen_sec"] = m.get("codegen_sec")
                r[f"{arm}_gen_sec"] = sum(g["sec"] for g in m.get("gen", []))
                r[f"{arm}_c_bytes"] = sum(c["bytes"] for c in m.get("compiles", []))
            else:
                r[f"{arm}_codegen_sec_partial"] = m.get("setup_sec_partial")
                r[f"{arm}_c_bytes_partial"] = m.get("c_bytes_partial")
        if arm in ("net_sens", "model_sens") and ok(m):
            r[f"{arm}_analytic"] = m.get("has_analytic_sens_rhs")
            r[f"{arm}_decline"] = m.get("sens_rhs_decline_reason")
    ref = A["interp"] if ok(M["interp"]) else None
    # The reference run's own absolute tolerance (first-pass cells predate the
    # recorded field and ran at the spec default, 1e-10).
    atol = float((M["interp"] or {}).get("atol_min", (M["interp"] or {}).get("atol_max", 1e-10)))
    for arm in ("net", "model", "net_sens", "model_sens"):
        if ref is not None and ok(M[arm]):
            r[f"{arm}_vs_interp"] = trajs(A[arm], ref, STATE, atol)
            r[f"{arm}_funcs_vs_interp"] = trajs(A[arm], ref, FUNCS, atol)
    if ok(M["net"]) and ok(M["model"]):
        r["model_vs_net"] = trajs(A["model"], A["net"], STATE + FUNCS, 0.0)
    if ok(M["net_sens"]) and ok(M["model_sens"]):
        r["sens_model_vs_net_raw"] = traj_err(A["model_sens"]["sens"], A["net_sens"]["sens"])
    if ok(M["fd"]) and ref is not None:
        p0 = [M["fd"]["p0"][q] for q in p["sens_params"]]
        fd = A["fd"]["fd"]
        rich = richardson(fd[0], fd[1])
        r["fd_noise"] = sens_err(fd[1], rich, p0, ref["species"])
        for arm in ("net_sens", "model_sens"):
            if ok(M[arm]):
                r[f"{arm}_vs_fd"] = sens_err(A[arm]["sens"], rich, p0, ref["species"])
        if ok(M["net_sens"]) and ok(M["model_sens"]):
            r["sens_model_vs_net"] = sens_err(
                A["model_sens"]["sens"], A["net_sens"]["sens"], p0, ref["species"]
            )
    return r


def classify(first: dict, t: dict) -> str:
    """One label for a flagged model after its tight-tolerance re-run.

    Leads with PATHS DIFFER when the model and .net paths disagree anywhere (the
    only case that bears on routing), then says what the tight re-run showed for
    the model path against the interpreter and the FD oracle."""
    notes = []
    if any(
        d.get(k) not in (None, 0.0)
        for d in (first, t)
        for k in ("model_vs_net", "sens_model_vs_net")
    ):
        notes.append("PATHS DIFFER")
    for k, what in (
        ("model_vs_interp", "trajectory"),
        ("model_sens_vs_interp", "sensitivity-run trajectory"),
        ("model_funcs_vs_interp", "functions"),
        ("model_sens_funcs_vs_interp", "sensitivity-run functions"),
    ):
        v, v0 = t.get(k), first.get(k)
        if v0 is not None and v0 > TRAJ_TOL:
            if v is None:
                notes.append(f"{what}: tight re-run did not finish")
            elif v <= TRAJ_TOL:
                notes.append(f"{what} within tolerance at tight tol")
            else:
                notes.append(f"{what} still differ(s)")
    v, noise = t.get("model_sens_vs_fd"), t.get("fd_noise")
    if v is not None and v > SENS_TOL:
        notes.append(
            "FD not decisive (FD's own error >= 1/3 of the gap)"
            if noise is not None and noise >= v / 3
            else "sensitivity differs from FD"
        )
    elif first.get("model_sens_vs_fd") is not None and first["model_sens_vs_fd"] > SENS_TOL:
        notes.append(
            "sensitivity agrees with FD at tight tol" if v is not None else "tight FD missing"
        )
    return "; ".join(dict.fromkeys(notes)) or "agrees at tight tol"


def f(x) -> str:
    return "-" if x is None else f"{x:.1e}"


def short(model_id: str) -> str:
    """Last directory plus file name: the corpus holds same-named models."""
    return "/".join(model_id.split("/")[-2:])


def cmd_report(a) -> None:
    plans = jl_read(a.out / "plan.jsonl")
    nets = jl_read(a.out / "nets.jsonl")
    rows, tight = [], {}
    for p in plans:
        d = a.out / "cells" / safe(p["model_id"])
        if d.exists():
            rows.append(model_row(p, d))
        dt = a.out / "tight" / safe(p["model_id"])
        if dt.exists():
            tight[p["model_id"]] = model_row(p, dt)
    with (a.out / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=float) + "\n")
    flagged = [
        r["model_id"]
        for r in rows
        if any(
            (r.get(k) or 0) > TRAJ_TOL
            for k in (
                "model_vs_interp",
                "model_sens_vs_interp",
                "model_funcs_vs_interp",
                "model_sens_funcs_vs_interp",
            )
        )
        or (r.get("model_sens_vs_fd") or 0) > SENS_TOL
    ]
    (a.out / "flagged.txt").write_text("\n".join(flagged) + "\n")

    L: list[str] = []
    P = L.append
    st = lambda r, arm: r.get(f"{arm}_status")  # noqa: E731
    gen = Counter(n["status"] for n in nets)
    P(f"# Codegen-path parity: {len(nets)} corpus models, {gen.get('OK', 0)} networks\n")
    P("## Arm outcomes\n")
    P("| | " + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for s in ("OK", "ERROR", "TIMEOUT", None):
        P(
            f"| {s or 'not run'} | "
            + " | ".join(str(sum(1 for r in rows if st(r, x) == s)) for x in ARMS)
            + " |"
        )
    bad_pc = [
        (r["model_id"], x)
        for r in rows
        for x in ("net", "model", "net_sens", "model_sens")
        if r.get(f"{x}_path_ok") is False
    ]
    P(
        f"\nPositive control (codegen arm ran compiled code on the path it claims): {'FAILED ' + str(bad_pc) if bad_pc else 'held on every cell'}."
    )

    P("\n## Failures by path\n")
    for x_n, x_m, lab in (
        ("net", "model", "plain codegen"),
        ("net_sens", "model_sens", "sensitivity run"),
    ):
        groups: dict[str, list] = {"model path only": [], ".net path only": [], "both paths": []}
        for r in rows:
            if st(r, x_n) is None or st(r, x_m) is None:
                continue
            fn, fm = st(r, x_n) != "OK", st(r, x_m) != "OK"
            key = (
                "both paths"
                if fn and fm
                else "model path only"
                if fm
                else ".net path only"
                if fn
                else None
            )
            if key:
                groups[key].append(r)
        P(f"**{lab}**: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()) + "\n")
        for k, v in groups.items():
            for r in v:
                P(
                    f"- {k}: `{r['model_id']}` -- {st(r, x_n)}/{st(r, x_m)} {(r.get(x_m + '_exc') or r.get(x_n + '_exc') or '')[:140]}"
                )
        P("")

    P("## Trajectories vs the interpreter\n")
    P("Max column-normalised error; species and observables, then functions (which may switch).\n")
    P("| | n | <= 1e-6 | (1e-6, 1e-4] | > 1e-4 |\n|---|---|---|---|---|")
    for key, lab in (
        ("model_vs_interp", "model path, plain"),
        ("net_vs_interp", ".net path, plain"),
        ("model_sens_vs_interp", "model path, sensitivity run"),
        ("net_sens_vs_interp", ".net path, sensitivity run"),
        ("model_funcs_vs_interp", "model path, plain: functions"),
        ("net_funcs_vs_interp", ".net path, plain: functions"),
        ("model_sens_funcs_vs_interp", "model path, sensitivity run: functions"),
        ("net_sens_funcs_vs_interp", ".net path, sensitivity run: functions"),
    ):
        v = [r[key] for r in rows if r.get(key) is not None]
        P(
            f"| {lab} | {len(v)} | {sum(x <= 1e-6 for x in v)} | {sum(1e-6 < x <= 1e-4 for x in v)} | {sum(x > 1e-4 for x in v)} |"
        )
    both = [r for r in rows if r.get("model_vs_net") is not None]
    diff = [r for r in both if r["model_vs_net"] != 0.0]
    P(
        f"\nModel path vs .net path, plain: byte-identical on {len(both) - len(diff)}/{len(both)}. Where they differ:\n"
    )
    for r in sorted(diff, key=lambda r: r["model_id"]):
        P(
            f"- `{r['model_id']}`: model vs interp {f(r.get('model_vs_interp'))}, .net vs interp {f(r.get('net_vs_interp'))}"
        )

    P("\n## Sensitivities\n")
    sv = [r for r in rows if r.get("sens_model_vs_net_raw") is not None]
    sd = [r for r in sv if r["sens_model_vs_net_raw"] != 0.0]
    P(
        f"Model path vs .net path: byte-identical on {len(sv) - len(sd)}/{len(sv)}. Where they differ "
        "(scaled error vs Richardson FD; FD's own error in brackets):\n"
    )
    for r in sorted(sd, key=lambda r: r["model_id"]):
        P(
            f"- `{r['model_id']}`: model {f(r.get('model_sens_vs_fd'))}, .net {f(r.get('net_sens_vs_fd'))} ({f(r.get('fd_noise'))})"
        )
    an = Counter(
        (r.get("model_sens_analytic"), r.get("net_sens_analytic"))
        for r in rows
        if st(r, "model_sens") == "OK" and st(r, "net_sens") == "OK"
    )
    P(f"\nAnalytic sensitivity RHS (model, .net): {dict(an)}")
    fv = [r["model_sens_vs_fd"] for r in rows if r.get("model_sens_vs_fd") is not None]
    P(
        f"\nModel path vs Richardson FD, first pass: {sum(x <= SENS_TOL for x in fv)}/{len(fv)} within {SENS_TOL:g}; the rest are adjudicated below."
    )

    if tight:
        P("\n## Adjudication: flagged models re-run at rtol 1e-10, per-species atol\n")
        P(
            "| model | trajectory 1st -> tight | sensitivity-run trajectory | functions (plain; sens run) | sensitivity vs FD (FD's own error) | verdict |\n|---|---|---|---|---|---|"
        )
        by = {r["model_id"]: r for r in rows}
        for mid in flagged:
            t, r0 = tight.get(mid), by[mid]
            if t is None:
                P(f"| `{mid}` | not re-run | | | | |")
                continue
            P(
                f"| `{short(mid)}` | {f(r0.get('model_vs_interp'))} -> {f(t.get('model_vs_interp'))} | "
                f"{f(r0.get('model_sens_vs_interp'))} -> {f(t.get('model_sens_vs_interp'))} | "
                f"{f(t.get('model_funcs_vs_interp'))}; {f(t.get('model_sens_funcs_vs_interp'))} | "
                f"{f(r0.get('model_sens_vs_fd'))} -> {f(t.get('model_sens_vs_fd'))} ({f(t.get('fd_noise'))}) | {classify(r0, t)} |"
            )

    P("\n## Codegen cost (cold cache, source generation + compile)\n")
    for xm, xn in (("model", "net"), ("model_sens", "net_sens")):
        pr = [(r.get(f"{xm}_codegen_sec"), r.get(f"{xn}_codegen_sec")) for r in rows]
        pr = [(m, n) for m, n in pr if isinstance(m, (int, float)) and isinstance(n, (int, float))]
        if pr:
            big = [(m, n) for m, n in pr if max(m, n) >= 5]
            P(
                f"- {xm} vs {xn}: n={len(pr)}, total {sum(m for m, _ in pr):.0f} s vs {sum(n for _, n in pr):.0f} s, "
                f"median ratio {statistics.median(m / n for m, n in pr if n > 0):.2f}; builds >= 5 s: {len(big)}, "
                f"total {sum(m for m, _ in big):.0f} s vs {sum(n for _, n in big):.0f} s"
            )

    def c(r, x):
        if st(r, x) == "OK":
            return f"{r[x + '_codegen_sec']:.0f} s"
        part = r.get(f"{x}_codegen_sec_partial")
        return f"{st(r, x)} (codegen {part:.0f} s)" if part else (st(r, x) or "-")

    P(
        "\n| reactions | species | model plain | .net plain | model sens | .net sens | sens C (MB) | model |\n|---|---|---|---|---|---|---|---|"
    )
    for r in sorted((r for r in rows if r["chunked"]), key=lambda r: -r["n_reactions"]):
        mb = r.get("model_sens_c_bytes") or r.get("model_sens_c_bytes_partial") or 0
        P(
            f"| {r['n_reactions']} | {r['n_species']} | {c(r, 'model')} | {c(r, 'net')} | {c(r, 'model_sens')} | "
            f"{c(r, 'net_sens')} | {mb / 1e6:.1f} | `{short(r['model_id'])}` |"
        )
    bud = sorted(
        {
            r["model_id"]
            for r in rows
            for x in ("net_sens", "model_sens")
            if "budget" in (r.get(x + "_decline") or "").lower()
        }
    )
    P(
        f"\nDerivation-budget declines: {len(bud)}"
        + (": " + ", ".join(f"`{m}`" for m in bud) if bud else "")
    )
    (a.out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n{len(rows)} models; {len(flagged)} flagged -> {a.out / 'flagged.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "runs" / "latest",
        help="output directory (default: runs/latest)",
    )
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nets", help="generate every corpus model's network with BNG2.pl")
    n.add_argument("--jobs", type=Path, default=JOBS)
    n.add_argument("--workers", type=int, default=6)
    n.add_argument("--gen-timeout", type=int, default=300)
    sub.add_parser("plan", help="choose the horizon and sensitivity parameters per model")
    r = sub.add_parser("run", help="run the (model, arm) cells")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--arms", help=f"comma-separated subset of {','.join(ARMS)}")
    r.add_argument("--only", help="regex over model ids")
    r.add_argument("--only-file", type=Path, help="file of model ids, one per line")
    r.add_argument("--tight", action="store_true", help="adjudication pass over flagged.txt")
    r.add_argument("--cache", type=Path, help="codegen cache dir (default: <out>/cg, cold)")
    r.add_argument("--timeout-scale", type=float, default=1.0)
    sub.add_parser("report", help="results.jsonl, summary.md and flagged.txt")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"nets": cmd_nets, "plan": cmd_plan, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
