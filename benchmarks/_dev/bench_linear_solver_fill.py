#!/usr/bin/env python3
"""Linear-solver choice vs LU fill-in on the ODE suite.

For every network in ``suite_ode.json`` (plus any ``--net`` given), report the
Jacobian density, two estimates of the LU fill of the Newton matrix I - gamma*J —
the engine's own (``jacobian_sparsity["lu_fill_estimate"]``, KLU's AMD analysis,
which the routing reads) and SuperLU's with a minimum-degree ordering on A + A^T
as an independent check — the linear solver bngsim picks, and the wall-clock of
one run under each backend:

    klu        force_sparse_linear_solver=True
    dense      force_dense_linear_solver=True, built-in dense LU
    lapack     force_dense_linear_solver=True, BNGSIM_LAPACK_DENSE=1, K=0 (BLAS from
               the first factorization)
    auto       whatever the build picks by itself

    uv run --no-sync python benchmarks/_dev/bench_linear_solver_fill.py
    uv run --no-sync python benchmarks/_dev/bench_linear_solver_fill.py --net my.net --t-end 43200 --param egf_nM=30

Each timing is the median of ``--repeats`` warm runs (``model.reset()`` between) in
this process; the first run of each configuration is a discarded warmup.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def lu_fill_estimate(core_model) -> float:
    """nnz(L+U)/n^2 of a SuperLU factorization of I + J with random values on J's pattern."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    s = core_model.jacobian_sparsity
    n = int(s["n"])
    if n == 0:
        return 0.0
    col_ptrs = np.asarray(s["col_ptrs"], dtype=np.int64)
    rows = np.asarray(s["row_indices"], dtype=np.int64)
    rng = np.random.default_rng(0)
    J = sp.csc_matrix((rng.uniform(-1, 1, rows.size), rows, col_ptrs), shape=(n, n))
    A = (sp.identity(n, format="csc") * (1.0 + abs(J).sum(axis=0).max()) - J).tocsc()
    lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A")
    return (lu.L.nnz + lu.U.nnz - n) / float(n * n)


def time_config(net, t_end, n_steps, params, config, repeats, rtol, atol):
    import bngsim

    env = {
        "BNGSIM_LAPACK_DENSE": "1",
        "BNGSIM_LAPACK_DENSE_K": "0",
        "BNGSIM_LAPACK_DENSE_MIN_N": "1",
    }
    saved = {k: os.environ.get(k) for k in env}
    if config == "lapack":
        os.environ.update(env)
    else:
        for k in env:
            os.environ.pop(k, None)
    try:
        model = bngsim.Model.from_net(str(net))
        for k, v in params.items():
            model.set_param(k, v)
        kw = {
            "klu": {"force_sparse_linear_solver": True},
            "dense": {"force_dense_linear_solver": True},
            "lapack": {"force_dense_linear_solver": True},
            "auto": {},
        }[config]
        sim = bngsim.Simulator(model, method="ode", **kw)
        times, stats = [], None
        for rep in range(repeats + 1):
            model.reset()
            for k, v in params.items():
                model.set_param(k, v)
            t0 = time.perf_counter()
            r = sim.run(t_span=(0.0, t_end), n_points=n_steps + 1, rtol=rtol, atol=atol)
            dt = time.perf_counter() - t0
            if rep:
                times.append(dt)
            stats = r.solver_stats
        return statistics.median(times), stats
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--net", action="append", default=[], help="extra .net file (repeatable)")
    ap.add_argument("--t-end", type=float, default=None, help="horizon for --net models")
    ap.add_argument("--n-steps", type=int, default=720)
    ap.add_argument("--param", action="append", default=[], help="NAME=VALUE for --net models")
    ap.add_argument("--no-suite", action="store_true", help="only the --net models")
    ap.add_argument(
        "--min-species", type=int, default=50, help="skip suite models smaller than this"
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--configs", default="klu,dense,lapack,auto")
    ap.add_argument("--json", type=Path, default=None, help="write results here")
    args = ap.parse_args()

    import bngsim

    jobs = []
    if not args.no_suite:
        suite = json.loads((HERE / "suite_ode.json").read_text())["models"]
        for m in suite:
            if m["species"] >= args.min_species:
                jobs.append(
                    (m["name"], REPO / "benchmarks" / m["net_file"], m["t_end"], m["n_steps"], {})
                )
    params = {k: float(v) for k, v in (p.split("=", 1) for p in args.param)}
    for net in args.net:
        jobs.append((Path(net).stem, Path(net), args.t_end, args.n_steps, params))

    configs = args.configs.split(",")
    rows = []
    hdr = f"{'model':34s} {'n':>5s} {'J dens':>7s} {'KLU est':>7s} {'SLU':>7s} " + " ".join(
        f"{c:>8s}" for c in configs
    )
    print(hdr + "   (seconds; auto solver in brackets)")
    for name, net, t_end, n_steps, prm in jobs:
        model = bngsim.Model.from_net(str(net))
        s = model._core.jacobian_sparsity
        fill = lu_fill_estimate(model._core)
        res = {}
        for c in configs:
            try:
                res[c] = time_config(
                    net, t_end, n_steps, prm, c, args.repeats, args.rtol, args.atol
                )
            except Exception as e:  # a config can legitimately fail (e.g. no KLU)
                res[c] = (float("nan"), {"error": str(e)[:80]})
        auto_ls = res.get("auto", (None, {}))[1].get("linear_solver", "?")
        rows.append(
            {
                "model": name,
                "n": s["n"],
                "density": s["density"],
                "lu_fill_estimate": s["lu_fill_estimate"],
                "lu_fill_superlu": fill,
                "seconds": {c: res[c][0] for c in configs},
                "linear_solver": {c: res[c][1].get("linear_solver") for c in configs},
                "n_steps": {c: res[c][1].get("n_steps") for c in configs},
            }
        )
        print(
            f"{name[:34]:34s} {s['n']:5d} {s['density']:7.3f} {s['lu_fill_estimate']:7.3f} {fill:7.3f} "
            + " ".join(f"{res[c][0]:8.3f}" for c in configs)
            + f"   [{auto_ls}]",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
