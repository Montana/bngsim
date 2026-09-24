"""One (model, arm) cell of the codegen-path parity sweep, run as its own process.

    python _cell.py <net> <arm> <out_prefix> <spec_json>

Every arm loads a FRESH ``Model.from_net`` of the same ``.net`` (state carried
between runs on one Model is how a control arm fakes a defect):

  interp      codegen=False                   the ExprTk interpreter -- the reference
  net         codegen=True                    the .net codegen path (_net_path set, as
                                              from_net/from_bngl leave it)
  model       codegen=True, _net_path=""      the model-based codegen path
  net_sens    sensitivity_params=P            forward sensitivities, .net path
  model_sens  sensitivity_params=P, _net_path=""
  fd          interpreter, central FD of the trajectory in each p in P, at two steps

Clearing ``model._net_path`` is the one switch ``Simulator`` reads to choose the
.net path, so the two codegen arms differ only in the code under test.

Writes ``<out_prefix>.npz`` (arrays) and then ``<out_prefix>.json`` (metadata); the
JSON's presence means the cell finished, OK or with a recorded error. ``PHASE``
lines on stdout say where a cell the supervisor had to kill was stuck.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback

import bngsim
import bngsim._codegen as cg
import numpy as np

net, arm, out_prefix, spec_json = sys.argv[1:5]
spec = json.loads(spec_json)
T_END = float(spec["t_end"])
N_POINTS = int(spec["n_points"])
RTOL = float(spec.get("rtol", 1e-8))
_a = spec.get("atol", 1e-10)
ATOL = np.asarray(_a, float) if isinstance(_a, list) else float(_a)
P = list(spec.get("sens_params", []))

rec: dict = {
    "arm": arm,
    "net": net,
    "t_end": T_END,
    "n_points": N_POINTS,
    "rtol": RTOL,
    "atol_min": float(np.min(ATOL)),
    "atol_max": float(np.max(ATOL)),
}
arrays: dict[str, np.ndarray] = {}


def phase(msg: str) -> None:
    print(f"PHASE {time.time():.1f} {msg}", flush=True)


# Time source generation and the cc compile separately, without touching bngsim:
# both are module globals of bngsim._codegen, looked up at call time.
_orig_compile = cg.compile_rhs


def _compile(src, h):
    t0 = time.perf_counter()
    phase(f"compile start ({len(src)} bytes)")
    try:
        return _orig_compile(src, h)
    finally:
        rec.setdefault("compiles", []).append(
            {"bytes": len(src), "sec": round(time.perf_counter() - t0, 3), "hash": h}
        )


cg.compile_rhs = _compile


def _timed(name: str) -> None:
    orig = getattr(cg, name)

    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            rec.setdefault("gen", []).append(
                {"fn": name, "sec": round(time.perf_counter() - t0, 3)}
            )

    setattr(cg, name, w)


for _n in ("generate_combined_c", "generate_combined_from_model"):
    _timed(_n)


def load(model_path: bool):
    m = bngsim.Model.from_net(net)
    if model_path:
        m._net_path = ""
    return m


def run(sim):
    phase(f"run start (setup {rec.get('setup_sec', 0):.1f}s)")
    t0 = time.perf_counter()
    r = sim.run(t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL)
    return r, time.perf_counter() - t0


def record_sim(sim) -> None:
    rec["backend"] = sim.codegen_backend
    rec["so"] = str(getattr(sim, "_codegen_so_path", "") or "")
    rec["sim_net_path"] = str(getattr(sim, "_net_path", "") or "")
    rec["codegen_sec"] = sim.last_codegen_sec
    rec["cache_hit"] = sim.codegen_cache_hit
    rec["jacobian_strategy"] = sim.jacobian_strategy


def record_result(r) -> None:
    arrays["time"] = np.asarray(r.time)
    arrays["species"] = np.asarray(r.species)
    for key, attr in (("obs", "observables"), ("expr", "expressions")):
        try:
            arrays[key] = np.asarray(getattr(r, attr))
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            rec[f"{key}_err"] = repr(e)
    with contextlib.suppress(Exception):
        rec["n_steps"] = int(r.solver_stats.get("n_steps", -1))


try:
    t_load = time.perf_counter()
    if arm in ("interp", "net", "model"):
        m = load(arm == "model")
        rec["load_sec"] = time.perf_counter() - t_load
        phase("setup start")
        t0 = time.perf_counter()
        sim = bngsim.Simulator(m, method="ode", codegen=(arm != "interp"))
        rec["setup_sec"] = time.perf_counter() - t0
        record_sim(sim)
        r, rec["run_sec"] = run(sim)
        record_result(r)
    elif arm in ("net_sens", "model_sens"):
        m = load(arm == "model_sens")
        rec["load_sec"] = time.perf_counter() - t_load
        phase("setup start")
        t0 = time.perf_counter()
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=P)
        rec["setup_sec"] = time.perf_counter() - t0
        record_sim(sim)
        rec["has_analytic_sens_rhs"] = sim.has_analytic_sens_rhs
        rec["sens_rhs_decline_reason"] = sim.sens_rhs_decline_reason
        r, rec["run_sec"] = run(sim)
        record_result(r)
        arrays["sens"] = np.asarray(r.sensitivities)
        rec["sens_params"] = P
    elif arm == "fd":
        # Central FD of the INTERPRETER's trajectory (no codegen anywhere), at a
        # tighter tolerance than the arms it adjudicates, perturbing through
        # set_param so derived parameters and parameter-valued initial conditions
        # follow the primary -- the total derivative sensitivities report. Two
        # step sizes, so the report can Richardson-extrapolate and read FD's own
        # error off their difference.
        frtol = float(spec.get("fd_rtol", 1e-10))
        _fa = spec.get("fd_atol", 1e-12)
        fatol = np.asarray(_fa, float) if isinstance(_fa, list) else float(_fa)
        hs = [float(h) for h in spec.get("fd_h", [1e-3, 1e-4])]
        base = load(False)
        pvals = {p: base.get_param(p) for p in P}
        rec["p0"] = pvals
        t0 = time.perf_counter()
        out = np.full((len(hs), N_POINTS, base.n_species, len(P)), np.nan)
        for hi, h in enumerate(hs):
            for j, p in enumerate(P):
                v0 = pvals[p]
                dv = h * abs(v0) if v0 != 0 else h
                ys = []
                for sgn in (+1, -1):
                    mm = load(False)
                    mm.set_param(p, v0 + sgn * dv)
                    s = bngsim.Simulator(mm, method="ode", codegen=False)
                    rr = s.run(t_span=(0.0, T_END), n_points=N_POINTS, rtol=frtol, atol=fatol)
                    ys.append(np.asarray(rr.species))
                out[hi, :, :, j] = (ys[0] - ys[1]) / (2 * dv)
        rec["fd_sec"] = time.perf_counter() - t0
        rec["fd_h"] = hs
        arrays["fd"] = out
    else:
        raise SystemExit(f"unknown arm {arm!r}")
    rec["status"] = "OK"
except BaseException as e:  # noqa: BLE001 - every failure is a recorded outcome
    rec["status"] = "ERROR"
    rec["exc_type"] = type(e).__name__
    rec["exc"] = str(e)[-1500:]
    rec["tb"] = traceback.format_exc()[-3000:]
    with contextlib.suppress(Exception):
        err = cg.last_codegen_error()
        rec["last_codegen_error"] = None if err is None else f"{type(err).__name__}: {err}"[-800:]
with contextlib.suppress(Exception):
    rec["last_sens_rhs_decline"] = cg.last_sens_rhs_decline()
if arrays:
    np.savez_compressed(out_prefix + ".npz", **arrays)
with open(out_prefix + ".json", "w") as f:
    json.dump(rec, f, default=str)
