"""Stage 4 of lanl/bngsim#616: the ensemble size a Girsanov gradient needs, measured.

For each stochbench SSA problem (github.com/wshlavacek/stochbench), run at its true
parameter values on its own sampling grid:

* ``M`` replicates with ``reaction_stats=True``;
* the likelihood-ratio score of every free parameter in log10
  (``bngsim.girsanov.scores``, chain-ruled through the ``__FREE`` aliases);
* the ensemble gradient of every data observable with its standard error
  (``bngsim.girsanov.gradient``, the sample covariance of observable and score);
* per (parameter, observable, time), the replicates a 10 % relative standard
  error would take, ``M_10 = M (se/|g|)^2 / 0.01``;
* the figure of merit the analysis says sets that cost, ``e = N_eff CV_f^2``,
  with ``N_eff = Var(W) / ln(10)^2`` the score's variance expressed as a firing
  count and ``CV_f`` the observable's coefficient of variation over replicates;
* the independence check behind it: the measured per-replicate variance of the
  centred product ``(f − f̄) W`` against ``Var(f) Var(W)``.

The mean-zero identity ``E[N_r − ∫ a_r ds] = 0`` is checked on every problem
before any gradient is believed, against the compensated process's exact
variance ``E[∫ a_r ds]``: pooled over every reaction (a z-score, worst over
times), per reaction where the ensemble expects at least ten firings (a z-score,
at ``t_end``), and by an exact Poisson tail for the rare reactions below that —
a rule-based network carries hundreds of reactions whose reactant complex exists
only transiently and which fire once or never in an ensemble; their counts are
Poisson, and the maximum of a normal z over hundreds of them is not a test of
anything. The number of rare reactions flagged at p < 1e-3 is reported beside
the number expected by chance.

A problem with a functional or Michaelis–Menten rate law is run and its
statistics recorded, but scored only where ``scores`` accepts it.

Usage::

    python benchmarks/girsanov/snr_stochbench.py \\
        --stochbench ~/Code/stochbench --bng2 /path/to/BNG2.pl \\
        --replicates 200 --seed 1 --out benchmarks/girsanov/results/snr_stochbench.json

``--problems`` restricts to named problem ids; ``--replicates 4`` is a timing pass.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import bngsim
import numpy as np
from bngsim import girsanov

LN10 = math.log(10.0)


def load_problems(root: Path):
    sys.path.insert(0, str(root / "src" / "python"))
    from stochbench.protocol import load_problem  # noqa: PLC0415

    dirs = sorted(
        p for p in (root / "Benchmark-Models").iterdir() if (p / "problem.json").exists()
    )
    return [load_problem(d) for d in dirs]


def substitute_free(text: str, truth: dict[str, float]) -> str:
    """Define every ``name__FREE`` placeholder at its true value.

    PyBNF replaces the placeholder textually; defining it as a parameter instead
    keeps the alias (``alpha_R alpha_R__FREE``) in the generated network, so the
    free parameter is scored through the chain rule exactly as a fit would use it.
    """
    lines = []
    for name, value in truth.items():
        if not re.search(rf"\b{re.escape(name)}\b", text):
            raise ValueError(f"free parameter {name} does not appear in the model")
        if re.search(rf"^\s*{re.escape(name)}\s", text, flags=re.M):
            raise ValueError(f"free parameter {name} is already defined in the model")
        lines.append(f"  {name} {float(value)!r}")
    block = "\n".join(lines)
    text, n = re.subn(r"^(\s*begin parameters\s*)$", rf"\1\n{block}", text, count=1, flags=re.M)
    if n != 1:
        raise ValueError("no parameters block to define the free parameters in")
    return text


def summarize(values: np.ndarray, w: np.ndarray, m: int) -> dict:
    """Per-parameter cost summaries from an ensemble's observables and scores."""
    grad, se = girsanov.gradient(values, w)  # (T, O, P)
    n_t, n_o, n_p = grad.shape
    var_f = values.var(axis=0, ddof=1)  # (T, O)
    mean_f = values.mean(axis=0)
    var_w = w.var(axis=0, ddof=1)  # (T, P)
    cv2 = np.full_like(var_f, np.nan)
    ok = mean_f != 0
    cv2[ok] = var_f[ok] / mean_f[ok] ** 2
    per_sample_var = (se * math.sqrt(m)) ** 2  # measured Var((f - f̄) W)
    pred_var = var_f[:, :, None] * var_w[:, None, :]  # independence prediction
    out = []
    for p in range(n_p):
        g = grad[1:, :, p]
        s = se[1:, :, p]
        resolved = np.abs(g) >= 2.0 * s  # the cells where M replicates see the parameter at 2σ
        rel = np.full(g.shape, np.inf)
        rel[g != 0] = s[g != 0] / np.abs(g[g != 0])
        m10 = m * rel**2 / 0.01
        ratio = per_sample_var[1:, :, p] / pred_var[1:, :, p]
        ratio = ratio[np.isfinite(ratio) & (pred_var[1:, :, p] > 0)]
        n_eff = var_w[1:, p] / LN10**2  # the score's variance as a firing count
        e = n_eff[:, None] * cv2[1:, :]  # (T-1, O): N_eff CV_f^2
        out.append(
            {
                "resolved_fraction": float(resolved.mean()),
                "m10_min": float(m10[resolved].min()) if resolved.any() else None,
                "m10_median": float(np.median(m10[resolved])) if resolved.any() else None,
                "m10_at_t_end_best_obs": float(m10[-1].min()),
                "n_eff_t_end": float(n_eff[-1]),
                "e_t_end_per_obs": [float(x) for x in e[-1]],
                "cov_over_indep_var_median": float(np.median(ratio)) if ratio.size else None,
                "cov_over_indep_var_max": float(ratio.max()) if ratio.size else None,
                "grad_t_end": [float(x) for x in grad[-1, :, p]],
                "se_t_end": [float(x) for x in se[-1, :, p]],
            }
        )
    return {"per_param": out, "cv2_t_end_per_obs": [float(x) for x in cv2[-1]]}


def run_problem(problem, bng2: Path, m: int, seed: int, workdir: Path) -> dict:
    text = substitute_free(problem.model_path.read_text(), problem.truth)
    bngl = workdir / f"{problem.id}.bngl"
    bngl.write_text(text)
    rec: dict = {"id": problem.id, "method": problem.method, "replicates": m}
    t0 = time.perf_counter()
    model = bngsim.Model.from_bngl(bngl, bng2_pl=bng2, net_out=workdir / f"{problem.id}.net")
    rec["network_s"] = time.perf_counter() - t0
    data = model._core.codegen_data()
    rec["n_species"] = len(data["species"]) if "species" in data else model.n_species
    rec["n_reactions"] = len(data["reactions"])
    rec["non_elementary"] = sum(r["type"] != "elementary" for r in data["reactions"])
    sim = bngsim.Simulator(model, method="ssa", reaction_stats=True)
    t0 = time.perf_counter()
    batch = sim.run_replicates(
        m,
        t_span=(problem.t_start, problem.t_end),
        n_points=problem.n_steps + 1,
        seed=seed,
        squeeze=True,
    )
    rec["simulate_s"] = time.perf_counter() - t0
    rec["propensity_backend"] = batch.ssa_diagnostics.get("propensity_backend")
    rec["negative_crossings"] = batch.ssa_diagnostics.get("n_negative_crossings")
    rec["steps_per_replicate"] = batch.solver_stats["n_steps"] / m
    counts = batch.reaction_firing_counts
    integrals = batch.reaction_propensity_integrals
    rec["mean_firings_t_end_per_reaction"] = [float(x) for x in counts[:, -1, :].mean(axis=0)]
    rec["reaction_labels"] = list(batch.reaction_labels)
    # Compensator identity: E[N_r - ∫a_r] = 0. The variance of the compensated
    # process is its quadratic variation E[∫a_r], exactly; the sample variance of
    # N - ∫a is not a substitute (for a channel that never fires it collapses to
    # the jitter of ∫a while the mean is -∫a). And a normal z is only a test where
    # the ensemble expects enough firings: a rule-based network has hundreds of
    # reactions whose reactant exists only transiently, with an expected ensemble
    # total of 1e-4..1e-3 firings, and one of them firing once is a 50σ "z" that
    # is nothing but a Poisson tail. So: pooled over reactions (well-behaved), per
    # reaction where the expected ensemble count is >= 10, and an exact Poisson
    # tail for the rest, with the number of flags expected by chance.
    d = counts[:, 1:, :] - integrals[:, 1:, :]
    pooled_num = d.sum(axis=2).mean(axis=0)  # (T-1,)
    pooled_lam = integrals[:, 1:, :].sum(axis=2).mean(axis=0)
    pooled_z = pooled_num / np.sqrt(np.maximum(pooled_lam, 1e-300) / m)
    rec["compensator_pooled_worst_z"] = float(np.abs(pooled_z).max())
    lam_end = m * integrals[:, -1, :].mean(axis=0)  # expected ensemble firings per reaction
    k_end = counts[:, -1, :].sum(axis=0)
    frequent = lam_end >= 10.0
    z_end = np.zeros_like(lam_end)
    z_end[frequent] = (k_end[frequent] - lam_end[frequent]) / np.sqrt(lam_end[frequent])
    rec["compensator_frequent_reactions"] = int(frequent.sum())
    rec["compensator_frequent_worst_z"] = float(np.abs(z_end).max()) if frequent.any() else None
    rec["compensator_z_t_end_per_reaction"] = [float(x) for x in z_end]
    rare = (~frequent) & (lam_end > 0)
    rec["compensator_rare_reactions"] = int(rare.sum())
    if rare.any():
        from scipy.stats import poisson  # noqa: PLC0415

        k, lam = k_end[rare], lam_end[rare]
        # two-sided exact tail: P(K >= k) for an excess, P(K <= k) for a deficit
        upper = poisson.sf(k - 1, lam)
        lower = poisson.cdf(k, lam)
        pval = np.minimum(1.0, 2.0 * np.minimum(upper, lower))
        rec["compensator_rare_min_p"] = float(pval.min())
        rec["compensator_rare_flagged"] = int((pval < 1e-3).sum())
        rec["compensator_rare_expected_flags"] = float(rare.sum() * 1e-3)
    else:
        rec["compensator_rare_min_p"] = None
        rec["compensator_rare_flagged"] = 0
        rec["compensator_rare_expected_flags"] = 0.0
    names = list(batch.observable_names)
    idx = [names.index(o) for o in problem.observables]
    values = np.asarray(batch.observables)[:, :, idx]
    rec["observables"] = list(problem.observables)
    rec["mean_t_end_per_obs"] = [float(x) for x in values[:, -1, :].mean(axis=0)]
    rec["free_parameters"] = list(problem.names)
    try:
        w = girsanov.scores(model, batch, problem.names, log10=True)
    except ValueError as exc:
        rec["scores_refused"] = str(exc)
        return rec
    rec.update(summarize(values, w, m))
    return rec


def markdown(records: list[dict]) -> str:
    lines = [
        "| problem | rxns (non-elem.) | steps/rep | M | sim s | compensator: pooled |z| · "
        "frequent |z|max (n) · rare flagged/expected (n) | "
        "resolved at 2σ | M₁₀ median | M₁₀ min | N_eff(t_end) | e = N_eff·CV² (t_end) |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        if "error" in r:
            lines.append(f"| {r['id']} | — | — | — | — | — | error: {r['error'][:60]} | | | | |")
            continue
        fz = r["compensator_frequent_worst_z"]
        comp = (
            f"{r['compensator_pooled_worst_z']:.2f} · "
            f"{'—' if fz is None else f'{fz:.2f}'} ({r['compensator_frequent_reactions']}) · "
            f"{r['compensator_rare_flagged']}/{r['compensator_rare_expected_flags']:.1f} "
            f"({r['compensator_rare_reactions']})"
        )
        base = (
            f"| {r['id']} | {r['n_reactions']} ({r['non_elementary']}) | "
            f"{r['steps_per_replicate']:.0f} | {r['replicates']} | {r['simulate_s']:.0f} | "
            f"{comp} | "
        )
        if "scores_refused" in r:
            lines.append(base + f"scores refused: {r['scores_refused'][:70]} | | | | |")
            continue
        for name, pp in zip(r["free_parameters"], r["per_param"], strict=True):
            med = "—" if pp["m10_median"] is None else f"{pp['m10_median']:.0f}"
            mn = "—" if pp["m10_min"] is None else f"{pp['m10_min']:.0f}"
            e = ", ".join(f"{x:.2g}" for x in pp["e_t_end_per_obs"])
            lines.append(
                base + f"{name}: {100 * pp['resolved_fraction']:.0f}% | {med} | {mn} | "
                f"{pp['n_eff_t_end']:.0f} | {e} |"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stochbench", type=Path, required=True)
    ap.add_argument("--bng2", type=Path, required=True, help="path to BNG2.pl")
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--problems", nargs="*", default=None, help="problem ids (default: every ssa problem)"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    problems = [p for p in load_problems(args.stochbench) if p.method == "ssa"]
    if args.problems:
        problems = [p for p in problems if p.id in set(args.problems)]
    records = []
    with tempfile.TemporaryDirectory(prefix="girsanov-snr-") as tmp:
        for problem in problems:
            print(f"== {problem.id}", flush=True)
            try:
                rec = run_problem(problem, args.bng2, args.replicates, args.seed, Path(tmp))
            except Exception as exc:  # noqa: BLE001 — a per-problem failure is a row, not an abort
                rec = {"id": problem.id, "error": f"{type(exc).__name__}: {exc}"}
            records.append(rec)
            print(
                json.dumps({k: v for k, v in rec.items() if k not in ("per_param",)}, indent=None)[
                    :600
                ],
                flush=True,
            )
    print()
    print(markdown(records))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"seed": args.seed, "records": records}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
