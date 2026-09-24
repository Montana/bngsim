"""Comparison metrics and per-model choices for the codegen-path parity sweep.

Pure functions (numpy only), so ``parity_checks/tests/test_codegen_paths.py`` can
pin them without a bngsim build or BNG2.pl.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

# ── error metrics ──────────────────────────────────────────────────────────


def traj_err(a, ref, floor_rel: float = 1e-6, abs_floor: float = 0.0) -> float | None:
    """Max over time of |a - ref|, normalised per column by that column's max |ref|.

    Two floors keep a column that is identically ~0 from being judged by division
    by ~0: ``floor_rel`` x the array's global max, and ``abs_floor`` -- pass the
    reference run's absolute tolerance, below which two solves cannot be told
    apart (an observable over species that sit at 1e-20 is roundoff on both
    sides, and without this reads as an O(1) disagreement). An all-zero reference
    is judged in absolute terms. NaN in exactly one of the two is ``inf``.
    """
    if a is None or ref is None:
        return None
    a = np.asarray(a, float)
    ref = np.asarray(ref, float)
    if a.shape != ref.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    if (np.isnan(a) ^ np.isnan(ref)).any():
        return float("inf")
    a = np.nan_to_num(a)
    ref = np.nan_to_num(ref)
    colmax = np.max(np.abs(ref), axis=0, keepdims=True)
    scale = colmax + floor_rel * float(np.max(np.abs(ref))) + abs_floor
    scale = np.where(scale > 0, scale, 1.0)
    return float(np.max(np.abs(a - ref) / scale))


def sens_err(a, ref, p0, yref) -> float | None:
    """Error in the SCALED sensitivity ``S[t, i, j] * p_j / max_t |y_i|``.

    That is the quantity a central difference with a relative step h resolves to
    about rtol/h, so it is the honest one to compare with an FD oracle. A raw
    per-column normalisation blows up on columns whose true sensitivity is ~0,
    where the FD noise *is* the column. ``a``/``ref`` are (time, species, param),
    ``p0`` the parameter values, ``yref`` the (time, species) reference trajectory.
    """
    if a is None or ref is None:
        return None
    a = np.asarray(a, float)
    ref = np.asarray(ref, float)
    if a.shape != ref.shape:
        return float("inf")
    if (np.isnan(a) ^ np.isnan(ref)).any():
        return float("inf")
    ymax = np.max(np.abs(np.asarray(yref, float)), axis=0)
    yscale = ymax + 1e-6 * float(np.max(ymax, initial=0.0))
    yscale = np.where(yscale > 0, yscale, 1.0)
    pscale = np.array([abs(v) if v != 0 else 1.0 for v in p0], float)
    d = (
        np.abs(np.nan_to_num(a) - np.nan_to_num(ref))
        * pscale[None, None, :]
        / yscale[None, :, None]
    )
    return float(np.max(d)) if d.size else 0.0


def richardson(fd_coarse, fd_fine, ratio: float = 10.0):
    """Richardson-extrapolate two central differences with steps ``ratio*h`` and ``h``.

    Central FD(h) = S + c h^2 + O(h^4), so this removes the h^2 term. Where the
    two steps differ by exactly ``ratio**2 - 1`` times the extrapolated error the
    fine step was truncation-limited, which is common on stiff or near-bifurcation
    models and is why a single-step FD is not an oracle.
    """
    fd_coarse = np.asarray(fd_coarse, float)
    fd_fine = np.asarray(fd_fine, float)
    return fd_fine + (fd_fine - fd_coarse) / (ratio**2 - 1.0)


def per_species_atol(y, rel: float = 1e-12, floor_rel: float = 1e-24) -> list[float]:
    """One absolute tolerance per species, ``rel`` x that species' own max |y|.

    A single atol scaled by the model's largest state lets one 1e12-count species
    set atol = 1 for its order-1 neighbours, and the reference itself goes bad
    (``m_vs_p_v1`` did exactly that: 17 steps, negative volumes). The floor,
    ``floor_rel`` x the largest state, is ``rel`` applied to roundoff scale: it
    only lifts a species that is itself roundoff next to the largest (or
    identically zero), so CVODE is not asked to resolve noise.
    """
    ymax = np.max(np.abs(np.asarray(y, float)), axis=0)
    floor = max(floor_rel * float(np.max(ymax, initial=0.0)), 1e-30)
    return [max(rel * float(v), floor) for v in ymax]


# ── per-model choices ──────────────────────────────────────────────────────

_IDENT = re.compile(r"[A-Za-z_]\w*")


def pick_sens_params(cd: dict, max_p: int = 4) -> tuple[list[str], dict]:
    """Up to ``max_p`` primary parameters, chosen to exercise each sensitivity family.

    From ``Model._core.codegen_data()``: first a primary behind a derived
    (``# ConstantExpression``) rate constant -- the chain rule the ``.net`` path was
    kept for --, then a primary inside a function that a Functional rate law calls,
    then a primary used directly as an Elementary rate constant; the rest by use
    count. A function-owned parameter slot (issue #227) is never chosen: it can be
    neither perturbed nor differentiated by, so its primaries stand in for it.
    """
    params = cd["parameters"]
    pinfo = {p["name"]: p for p in params}
    funcs = {f["name"]: f["expression"] for f in cd["functions"]}
    memo: dict[str, frozenset] = {}

    def prims_of_expr(expr: str, seen: frozenset = frozenset()) -> set[str]:
        out: set[str] = set()
        for tok in _IDENT.findall(expr or ""):
            if tok in seen:
                continue
            if tok in funcs:  # before pinfo: a function may own a parameter slot
                out |= prims_of_expr(funcs[tok], seen | {tok})
            elif tok in pinfo:
                if pinfo[tok]["is_const"]:
                    out.add(tok)
                else:
                    out |= prims_of_name(tok, seen | {tok})
        return out

    def prims_of_name(name: str, seen: frozenset = frozenset()) -> set[str]:
        if name not in memo:
            memo[name] = frozenset(prims_of_expr(pinfo[name]["expression"], seen | {name}))
        return set(memo[name])

    chain: Counter = Counter()
    func: Counter = Counter()
    direct: Counter = Counter()
    for r in cd["reactions"]:
        if r["type"] == "elementary":
            for pi in r["rate_param_indices"]:
                name = params[pi]["name"]
                if name in funcs:
                    for q in prims_of_expr(funcs[name]):
                        chain[q] += 1
                elif params[pi]["is_const"]:
                    direct[name] += 1
                else:
                    for q in prims_of_name(name):
                        chain[q] += 1
        else:
            for q in prims_of_expr(funcs.get(r.get("function_name") or "", "")):
                func[q] += 1
            for pi in r["rate_param_indices"]:
                name = params[pi]["name"]
                if name in funcs:
                    continue  # the function's own slot; its body was walked above
                for q in {name} if params[pi]["is_const"] else prims_of_name(name):
                    func[q] += 1

    def ranked(c: Counter) -> list[str]:
        return [k for k, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) if k not in funcs]

    chosen: list[str] = []
    for c in (chain, func, direct):
        for k in ranked(c):
            if k not in chosen:
                chosen.append(k)
                break
    for k in ranked(chain + func + direct):
        if len(chosen) >= max_p:
            break
        if k not in chosen:
            chosen.append(k)
    return chosen, {"n_chain": len(chain), "n_func": len(func), "n_direct": len(direct)}


def horizon(model_path: Path) -> tuple[float, int, str]:
    """``(t_end, n_points, source)`` for a model: its protocol's primary experiment,
    else the first ``simulate`` action's ``t_end``, else 100. ``n_points`` is
    clamped to [21, 201]; the sweep compares engines over one horizon, it does not
    replay the action script."""
    try:
        from bngsim.convert._protocol import parse_bngl_protocol

        e = parse_bngl_protocol(model_path, strict=False).primary_experiment()
        if e is not None:
            t0, t1 = e.t_span
            if t1 > t0 and np.isfinite(t1 - t0):
                return float(t1 - t0), max(min(int(e.n_points), 201), 21), "protocol"
    except Exception:  # noqa: BLE001 - fall through to the text scan
        pass
    text = Path(model_path).read_text(errors="replace")
    for m in re.finditer(r"simulate(?:_\w+)?\s*\(\s*\{([^}]*)\}", text):
        blob = m.group(1)
        te = re.search(r"t_end\s*=>\s*([0-9.eE+-]+)", blob)
        ts = re.search(r"t_start\s*=>\s*([0-9.eE+-]+)", blob)
        ns = re.search(r"n_(?:output_)?steps\s*=>\s*([0-9]+)", blob)
        if te:
            try:
                t = float(te.group(1)) - (float(ts.group(1)) if ts else 0.0)
                n = int(ns.group(1)) + 1 if ns else 51
            except ValueError:
                continue
            if t > 0:
                return t, max(min(n, 201), 21), "regex"
    return 100.0, 51, "default"
