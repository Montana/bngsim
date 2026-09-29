#!/usr/bin/env python3
"""Coupling-loop and large-network ODE timings behind the mecha-net fixes.

Four measurements, each on a network passed on the command line (the networks
that exposed these issues are large rule-derived .net files; nothing here is
specific to them):

  full      one 12 h ODE run of a large network with the linear solver bngsim
            picks by itself (``--full-net``, with ``--param NAME=VALUE``).
  loop      a ReactionKernel coupling loop on ``--loop-net``: 720 steps of
            set_state (one coupling species overwritten) then advance(60 s),
            the shape of an operator-split hybrid.
  tfun      the jacobian strategy and one 12 h run when a function of
            ``--full-net`` is replaced by a time-indexed table function
            (``--tfun-func NAME``: its .net line ``NAME() <expr>`` becomes
            ``NAME() tfun(<file>, time, method=>"linear")`` over a step waveform).
  rollback  the same loop with a predictor-corrector: every step is advanced,
            rolled back (state and clock) and advanced again.

    uv run --no-sync python benchmarks/kernel/bench_coupling_loop.py \\
        --full-net egfr_erk.net --param egf_nM=30 --tfun-func egf_in \\
        --loop-net egfr_sos.net --loop-species 'ERK(S~PP)' --loop-value 1e6

Timings are wall clock in this process, median of ``--repeats`` after one warmup.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _median_time(fn, repeats):
    if repeats <= 0:  # a single, cold measurement (for slow configurations)
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_full(net, params, repeats):
    import bngsim

    model = bngsim.Model.from_net(str(net))
    sim = bngsim.Simulator(model, method="ode")
    out = {}

    def one():
        model.reset()
        for k, v in params.items():
            model.set_param(k, v)
        out["r"] = sim.run(t_span=(0.0, 43200.0), n_points=721, rtol=1e-6, atol=1e-2)

    t = _median_time(one, repeats)
    st = out["r"].solver_stats
    return {
        "seconds": t,
        "linear_solver": st["linear_solver"],
        "n_steps": st["n_steps"],
        "n_dense_blas_factorizations": st.get("n_dense_blas_factorizations"),
    }


def _kernel(net, force_dense=False):
    import bngsim

    model = bngsim.Model.from_net(str(net))
    return bngsim.ReactionKernel(model, method="ode", force_dense_linear_solver=force_dense)


def bench_loop(net, species, value, repeats, rollback=False, force_dense=False):
    kernel = _kernel(net, force_dense)
    i = kernel.state_names.index(species)
    has_time_api = "time" in kernel.set_state.__code__.co_varnames

    def set_state(state, t):
        if has_time_api:
            kernel.set_state(state, time=t)
        else:  # the private clock, before a public API existed
            kernel.set_state(state)
            kernel.simulator._current_time = t

    def one():
        kernel.reset()
        for step in range(720):
            s = kernel.get_state()
            s[i] = value if 200 <= step < 260 else 0.0
            t = kernel.time
            if rollback:
                set_state(s, t)
                kernel.advance(60.0)
                set_state(s, t)
            else:
                kernel.set_state(s)
            kernel.advance(60.0)

    t = _median_time(one, repeats)
    st = kernel.last_result.solver_stats
    return {
        "seconds": t,
        "linear_solver": st["linear_solver"],
        "blas_factorizations_in_last_step": st.get("n_dense_blas_factorizations"),
    }


def bench_tfun(net, func, params, repeats, hours=12.0):
    import bngsim

    text = Path(net).read_text()
    tmp = Path(tempfile.mkdtemp(prefix="bench_tfun_"))
    tfun = tmp / "wave.tfun"
    n_min = int(round(hours * 60))
    u = np.r_[
        np.zeros(min(30, n_min)), 30.0 * np.ones(n_min + 1 - min(30, n_min))
    ]  # step at 30 min
    np.savetxt(
        tfun, np.column_stack([np.arange(u.size) * 60.0, u]), header=f"time {func}", comments="# "
    )
    lines = text.splitlines()
    hit = [k for k, ln in enumerate(lines) if ln.split()[1:2] == [f"{func}()"]]
    if not hit:
        raise SystemExit(f"--tfun-func {func}: no '{func}()' line in the functions block")
    k = hit[0]
    lines[k] = " ".join(lines[k].split()[:2]) + f" tfun('{tfun}', time, method=>\"linear\")"
    wave = tmp / "wave.net"
    wave.write_text("\n".join(lines) + "\n")
    model = bngsim.Model.from_net(str(wave))
    sim = bngsim.Simulator(model, method="ode")
    out = {}

    def one():
        model.reset()
        for p, v in params.items():
            model.set_param(p, v)
        out["r"] = sim.run(
            t_span=(0.0, 60.0 * n_min), n_points=n_min + 1, rtol=1e-6, atol=1e-2, max_step=60.0
        )

    t = _median_time(one, repeats)
    st = out["r"].solver_stats
    return {
        "hours": hours,
        "seconds": t,
        "jacobian_strategy": sim.jacobian_strategy,
        "linear_solver": st["linear_solver"],
        "n_steps": st["n_steps"],
        "n_rhs_evals": st["n_rhs_evals"],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--full-net", type=Path)
    ap.add_argument("--param", action="append", default=[], help="NAME=VALUE for --full-net")
    ap.add_argument("--tfun-func", default=None)
    ap.add_argument("--loop-net", type=Path)
    ap.add_argument("--loop-species", default=None)
    ap.add_argument("--loop-value", type=float, default=1.0)
    ap.add_argument(
        "--loop-force-dense",
        action="store_true",
        help="force_dense_linear_solver for the loop (with BNGSIM_LAPACK_DENSE=1 this exercises "
        "the GH #132 count gate across coupling steps)",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tfun-hours", type=float, default=12.0)
    ap.add_argument("--only", default="full,loop,rollback,tfun")
    args = ap.parse_args()
    params = {k: float(v) for k, v in (p.split("=", 1) for p in args.param)}
    only = set(args.only.split(","))

    import bngsim

    print(
        f"bngsim {bngsim.__version__}  lapack_dense={bngsim.HAS_LAPACK_DENSE}  klu={bngsim.HAS_KLU}"
    )
    if args.full_net and "full" in only:
        print("full    ", bench_full(args.full_net, params, args.repeats), flush=True)
    if args.loop_net and args.loop_species and "loop" in only:
        print(
            "loop    ",
            bench_loop(
                args.loop_net,
                args.loop_species,
                args.loop_value,
                args.repeats,
                force_dense=args.loop_force_dense,
            ),
            flush=True,
        )
    if args.loop_net and args.loop_species and "rollback" in only:
        print(
            "rollback",
            bench_loop(
                args.loop_net,
                args.loop_species,
                args.loop_value,
                args.repeats,
                rollback=True,
                force_dense=args.loop_force_dense,
            ),
            flush=True,
        )
    if args.full_net and args.tfun_func and "tfun" in only:
        print(
            "tfun    ",
            bench_tfun(args.full_net, args.tfun_func, params, args.repeats, args.tfun_hours),
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
