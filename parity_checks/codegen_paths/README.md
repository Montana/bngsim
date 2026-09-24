# `codegen_paths` — the two codegen paths against the interpreter

bngsim compiles an ODE right-hand side two ways. A model loaded with
`Model.from_net` or `Model.from_bngl` goes through the **`.net` path**: codegen
re-reads the `.net` file with its own parser (`_codegen._parse_net_file`) and
emits C from that text. SBML, Antimony and builder models go through the
**model path**: C is emitted from the model bngsim actually built
(`codegen_data()`, `prepare_model_codegen`). Every `codegen=True` run and every
forward-sensitivity run of a BNGL model takes the `.net` path, so where the two
readings of the file disagree, the trajectory itself changes (#784).

Issue #803 proposes routing `.net`/BNGL models through the model path and
deleting the `.net` one. This suite is its step 1: measure, on the whole
committed BNGL corpus, whether the model path is at parity before anything is
routed to it.

## What it runs

`codegen_paths.py nets` generates each corpus model's network with the same
`bngl_to_net` that `Model.from_bngl` uses. Every model with a network then gets
six **cells**, each in its own process, each on a fresh `Model.from_net`, against
a cold codegen cache:

| arm | what runs | role |
|---|---|---|
| `interp` | `codegen=False` | the ExprTk interpreter: the reference trajectory |
| `net` | `codegen=True` | the `.net` path, as `from_net`/`from_bngl` leave the model |
| `model` | `codegen=True`, `model._net_path = ""` | the model path. Clearing `_net_path` is the one switch `Simulator` reads, so the two codegen arms differ only in the code under test |
| `net_sens` | `sensitivity_params=P` | forward sensitivities, `.net` path |
| `model_sens` | `sensitivity_params=P`, `_net_path = ""` | forward sensitivities, model path |
| `fd` | interpreter only | central differences of the trajectory in each `p` in `P`, at two relative steps (1e-3, 1e-4), perturbing through `set_param` so derived parameters and parameter-valued initial conditions follow |

`P` is up to four primary parameters per model (`_metrics.pick_sens_params`): one
behind a derived `# ConstantExpression` rate constant — the chain rule that was
the stated reason for keeping `.net` models on their own path — one inside a
Functional rate law, one used directly as an Elementary rate constant. The
horizon is the model's primary protocol experiment (else its first `simulate`
action). The sweep compares engines over one horizon; it does not replay the
action script.

A **positive control** is recorded on every codegen cell: the compiled backend
ran, and `Simulator._net_path` was set on the `net` arms and empty on the
`model` arms.

## How a disagreement is judged

- **Trajectories** (`_metrics.traj_err`): the maximum over time of |a − ref|,
  normalised per column by that column's max |ref|, with floors so a column that
  is ~0 is not divided by ~0 — including the reference run's `atol`, below
  which two solves cannot be told apart. Species and observables are judged
  together. Functions are reported apart, because a time or state switch sampled
  exactly at its switching instant is legitimately left- or right-continuous.
- **Sensitivities** (`_metrics.sens_err`): the error in `S·p / max|y|`, the
  quantity a central difference with a relative step resolves. The oracle is the
  two FD steps **Richardson-extrapolated**. A single step is not an oracle: on
  stiff and near-bifurcation models the fine step is truncation-limited, and the
  two steps differ by exactly 99× the extrapolated error. The difference between
  the fine step and the extrapolation is reported as FD's own error; where it is
  at least a third of the gap, FD is not decisive.
- **Adjudication.** Every model over 1e-4 (trajectories) or 1e-3 (sensitivities)
  is re-run with `run --tight`: rtol 1e-10, an `atol` per species of 1e-12 × that
  species' own scale, and FD at rtol 1e-12. A real defect does not move with the
  tolerance. Solver noise and chaos do.

## Running it

Needs BNG2.pl (an installed PyBioNetGen, or `$BNGPATH`/`$BNG2_PL`), a C compiler,
and the corpus under `../bng_parity/models`. About two hours on 16 cores with 4–8
workers. The largest networks (`e7`, 172,032 reactions) dominate.

```
python codegen_paths.py nets                  # BNG2.pl generate_network per model
python codegen_paths.py plan                  # horizon + sensitivity parameters
python codegen_paths.py run --workers 6       # every (model, arm) cell
python codegen_paths.py report                # results.jsonl, summary.md, flagged.txt
python codegen_paths.py run --tight           # re-run flagged models at tight tolerance
python codegen_paths.py report                # again, with the adjudication table
python defects_784.py                         # the #784 reproductions on both paths
```

Output goes to `runs/latest/` (`--out` to change; `runs/` is gitignored). Every
step resumes. A cell whose `.log` exists is in flight, so two `run` processes can
share a directory. A cell is killed together with everything under it — sympy,
clang — at its arm's timeout (900 s plain, 1800 s sensitivity, 2400 s FD). A
killed cell still reports how long codegen took, from `PHASE` lines in its log.

`defects_784.py` needs no BNG2.pl. It runs each #784 reproduction, plus #608's
unindexed-block file, on the interpreter and both codegen paths, against closed
forms or scipy.

## Results: 2026-09-24, `main` at 4c0dac1

Run on macOS arm64 (16 cores), BNG2.pl 2.9.3 from PyBioNetGen, cold codegen cache,
4–8 workers. `runs/` is not committed. `codegen_paths.py report` regenerates
`summary.md`, and these tables are taken from it.

**Verdict: on this corpus the model path is never less correct than the `.net`
path, and differs from it only where the `.net` path is wrong.** It has two gaps
of its own, listed under "Gaps in the model path" below. The stated reason for
keeping `.net` models on their own path, "issue #15's derived-parameter chain
rules", does not hold: the two paths' sensitivities are byte-identical on every
model the `.net` path reads correctly.

### Coverage

895 corpus models; 818 produced a network. Of the 77 that did not:

- 63 are NF-only rule sets whose network is unbounded (no network within 300 s).
- 7 are BNG2.pl aborts (`FunctionProduct type RateLaw is not yet supported`).
- 7 are large ODE rule sets with no network after 30 minutes: `ChylekFceRI_2014`,
  `ChylekTCR_2014`, `Chylek_library`, `Chattaraj_2021`, `Massole_2023`,
  `Suderman_2013`, `cleavage_mechanism_v1`.

| | interp | net | model | net_sens | model_sens | fd |
|---|---|---|---|---|---|---|
| OK | 814 | 811 | 812 | 800 | 802 | 807 |
| error | 0 | 1 | 2 | 10 | 10 | 2 |
| timeout | 4 | 6 | 4 | 5 | 3 | 6 |
| not run (no sensitivity parameter) | 0 | 0 | 0 | 3 | 3 | 3 |

The positive control held on every codegen cell.

### Where the two paths differ

Plain trajectories are byte-identical on 802 of the 810 models where both paths
ran. Sensitivities are byte-identical on 791 of 799. Every difference is a model
the `.net` path reads wrongly (#689: a scientific-notation stat factor, or a
legacy `Sat`):

| model | model path vs interpreter | `.net` path vs interpreter | sensitivities vs FD, model / `.net` (FD's own error) |
|---|---|---|---|
| `LR_comp` | 2.6e-08 | 1.0 | 2.1e-07 / 0.49 (1.8e-09) |
| `LRR_comp` | 5.0e-08 | 1.0 | 3.8e-06 / 0.47 (4.1e-08) |
| `energy_example1` | 2.5e-15 | 1.0 | 1.5e-06 / 18 (2.3e-05) |
| `test_sat` | 4.7e-15 | 1.0 | 4.1e-07 / 0.77 (2.0e-07) |
| `RasRaf_cBNGL_eBNGL_v3` | 6.1e-15 | 1.0 | 1.9e-06 / 1.1 (2.2e-08) |
| `catalysis` | 3.1e-12 | 1.0 | 1.5e-06 / 4.4 (9.6e-07) |
| `wofsy_goldstein` | 5.0e-13 | 1.0 | 1.6e-06 / 1.9 (2.9e-07) |
| `CaOscillate_Sat` | 5.3e-07 | 2.3e+23 | 1.2e-02 / 1.3e+25 (9.8e-02; an oscillator, FD not decisive) |
| `mwc` (2 copies) | 9.7e-13 | timeout (CVODE stalls on the zeroed rates) | 1.0e-07 / timeout |
| `db_3rd_order_EnergyBNGL_v1` | compile error (gap 1 below) | 1.0, silently | – / 0.78 |

Where both paths fail, they fail the same way. Five models fail on both plain
runs: `e6`, `e6_1`, `e7` and `gm_game_of_life` time out (so does the
interpreter), and `l-type-calcium-channel-dynamics` hits a CVODE error (see
"Found along the way"). Twelve sensitivity runs fail on both paths: three
timeouts, eight CVODE stalls, errors or refused state switches, and `e7`'s
compile timeout.

The analytic sensitivity RHS is used on the same 783 models on both paths, and
declined on the same 16. Those use `max()`, `min()` or non-smooth `if()`, and the
two paths give the same reason.

### Model path against the interpreter and FD

| | ≤ 1e-6 | (1e-6, 1e-4] | > 1e-4 |
|---|---|---|---|
| plain trajectory | 797 | 11 | 4 |
| sensitivity-run trajectory | 758 | 33 | 9 |
| functions, sensitivity run | 771 | 18 | 11 |

Against Richardson FD, the model path's sensitivities are within 1e-3 on 760 of
797 models on the first pass. The 44 models flagged by either measure were
re-run with `run --tight`. None is a difference between the paths except
`CaOscillate_Sat` (#689 above). They fall into five groups, plus two that did not
settle:

- **Solver tolerance.** The difference disappears at tight tolerance: `PBPK_v1`
  (states around 1e-8, where `atol` 1e-10 is loose), `transport_v2`,
  `transport_v3`, `ph_schrodinger`, `Lang_2024`, `heise`, `fceri_ji_red_3`,
  `temp`, `ml_kmeans`, `ml_svm`, and the sensitivity-run trajectory of
  `l-type-calcium-channel-dynamics`.
- **Chaotic or ill-conditioned IVPs.** The trajectory keeps moving with the
  tolerance, identically on both codegen paths: `ph_lorenz_attractor`,
  `eco_food_web_chaos_3sp`, and `eco_coevolution_host_parasite`, which the
  bng_parity overrides already run at 1e-12. Also the sensitivity-run trajectory
  of `predator-prey-dynamics`.
- **FD not decisive.** On oscillating and switching models FD's own error is at
  least a third of the gap: `Lisman`, `vilar_2002b`/`vilar_2002c`,
  `CaOscillate_Func`/`CaOscillate_Sat_2`, `fceri_ji` (and the three corpus models
  that share its network),
  `edg_zero_rate_rule`, `edg_synth_bonded_complex`, `test2`, `oscillator_1`,
  `overlap_rules2`, `geneModel`, `model_for_phase2_v12`, `Rab_v5_4`, `fceri_fyn`.
- **Not differentiable in the chosen parameter.** Initial conditions of the form
  `rint(466689*f)` make the analytic sensitivity exactly 0, while FD steps across
  the integer jumps: `Rag_Ragulator_assembly`, `V600E_BRAF_v1`,
  `Model_V600E_BRAF_v4`, `Model_V600E_BRAF_v5`. FD scales as 1/h or flips sign
  between the two steps: `ExampleModel4_v6`, `proliferation`. State switches:
  `l-type-calcium-channel-dynamics`, and `m_vs_p_v1`, whose switches read an
  observable named `t` that is a counter species, not the clock.
- **A function sampled at its switch time.** `ItalyModel_v7`'s `Data_DailyCases`
  is a step function of time. At each integer sample a sensitivity run reports the
  next interval's value, while plain runs report the current one. This happens on
  both paths (see "Found along the way").
- **Not settled at tight tolerance:** `cs_hash_function` and `test_func`, whose
  sensitivity runs hit the 1800 s limit at rtol 1e-10 on both paths. On the first
  pass, model and `.net` are byte-identical on both. `test_func`'s gap to FD
  (1.4e-2) is below FD's own error (7.7e-2), and `cs_hash_function`'s difference
  is in a switching function (`max()`/`if()`) of the sensitivity run, with
  trajectories agreeing to 7e-8.

### #784 reproductions (`defects_784.py`)

| issue | model path | `.net` path | interpreter |
|---|---|---|---|
| #608 unindexed blocks | 2/2 | 0/2 | 1/1 |
| #689 sci-notation / leading-dot stat factor, `Sat`, `Hill` | 9/9 | 0/9 | 6/6 |
| #694 derived-parameter override, incl. cross-process cache order | 5/5 | 2/5 | – |
| #699 forward function reference, flat and chunked | 4/4 | 0/4 | 2/2 |
| #721 tfun time index `T` | 4/4 | 0/4 | 2/2 |
| #730 `.net` rewritten after load | 2/2 | 0/2 | – |
| #731 same-mtime rewrite | 1/1 | 0/1 | – |
| #734 `=`, `--`, `a==b<1` in function text | **0/6** | 0/6 | 3/3 |

### Codegen cost

Cold cache; source generation plus `cc`. Plain builds: 922 s on the model path
vs 902 s on the `.net` path in total (median ratio 1.02). Sensitivity builds:
2155 s vs 2170 s (median 1.00). On the builds of 5 s or more, the worst
model/`.net` ratio is 1.22 (plain) and 1.07 (sensitivity). No model declined an
analytic sensitivity RHS on a derivation budget, on either path. Duplicate
networks in the corpus (`e5_1`, `fceri_ji_4`, `test_network_gen`) hit the cache
their twin filled, which is why they show 0 s.

Every build of 2,000 reactions or more (chunked): codegen seconds per arm; `TIMEOUT (codegen N s)` is a cell killed during integration after N s of codegen; `ERROR` on a sensitivity build here is the 600 s compile timeout.

| reactions | species | model plain | .net plain | model sens | .net sens | sens C (MB) | model |
|---|---|---|---|---|---|---|---|
| 172032 | 16386 | TIMEOUT (codegen 258 s) | TIMEOUT (codegen 247 s) | ERROR | ERROR | 138.8 | `corpus/e7.bngl` |
| 36864 | 4098 | TIMEOUT | TIMEOUT | 229 s | 245 s | 29.3 | `corpus/e6.bngl` |
| 36864 | 4098 | TIMEOUT (codegen 2 s) | TIMEOUT | 2 s | 0 s | 0.0 | `nf/e6_1.bngl` |
| 24388 | 1122 | 63 s | 62 s | 599 s | ERROR | 32.7 | `BaruaBCR2012/BaruaBCR_2012.bngl` |
| 15328 | 1281 | 22 s | 24 s | 48 s | 49 s | 12.1 | `fcerifyn/fceri_fyn.bngl` |
| 7680 | 1026 | 14 s | 14 s | 32 s | 32 s | 6.0 | `corpus/e5.bngl` |
| 7680 | 1026 | 0 s | 0 s | 0 s | 0 s | 0.0 | `nf/e5_1.bngl` |
| 6914 | 185 | 19 s | 19 s | 50 s | 49 s | 6.5 | `ode/atg_model_v2.bngl` |
| 6400 | 593 | 12 s | 11 s | 26 s | 27 s | 3.8 | `ode/before_bunching.bngl` |
| 5022 | 624 | 12 s | 12 s | 72 s | 72 s | 5.7 | `ode/Models_n.bngl` |
| 4198 | 589 | 9 s | 9 s | 47 s | 47 s | 4.3 | `ode/IGF1R_model_v1.bngl` |
| 4022 | 355 | 10 s | 10 s | 40 s | 41 s | 4.1 | `ode/basal_receptor_signaling.bngl` |
| 4022 | 355 | 9 s | 10 s | 40 s | 40 s | 4.1 | `ode/basal_EGFR_signaling_HeLa_S3.bngl` |
| 3852 | 543 | 12 s | 11 s | 63 s | 64 s | 4.4 | `ode/Models_c.bngl` |
| 3749 | 356 | 7 s | 6 s | 13 s | 13 s | 3.0 | `corpus/egfr_net.bngl` |
| 3749 | 356 | 10 s | 9 s | 21 s | 21 s | 3.0 | `ode/egfr_net_6.bngl` |
| 3749 | 356 | 11 s | 11 s | 21 s | 21 s | 3.0 | `Blinov2006/Blinov_2006.bngl` |
| 3749 | 356 | 11 s | 11 s | 23 s | 24 s | 3.0 | `02-egfr/egfr_ground.bngl` |
| 3749 | 356 | 6 s | 6 s | 14 s | 13 s | 3.0 | `egfrnet/egfr_net.bngl` |
| 3680 | 354 | 6 s | 6 s | 13 s | 13 s | 2.9 | `corpus/fceri_ji.bngl` |
| 3680 | 354 | 9 s | 10 s | 20 s | 21 s | 2.9 | `ode/fceri_ji.bngl` |
| 3680 | 354 | 0 s | 0 s | 0 s | 0 s | 0.0 | `ode/fceri_ji_4.bngl` |
| 3680 | 354 | 0 s | 0 s | 0 s | 0 s | 0.0 | `ode/test_network_gen.bngl` |
| 3680 | 354 | 7 s | 6 s | 12 s | 13 s | 2.8 | `FceRIji/FceRI_ji.bngl` |
| 3328 | 1025 | 6 s | 6 s | 12 s | 11 s | 2.4 | `nfsim_basicmodels/v21.bngl` |
| 2948 | 577 | 6 s | 6 s | 23 s | 24 s | 3.6 | `ode/Compressed_model_Hela.bngl` |
| 2948 | 577 | 10 s | 9 s | 32 s | 32 s | 3.6 | `ode/Reduced_IGF1R_hela_cell_specific_model.bngl` |
| 2809 | 104 | 4 s | 4 s | 10 s | 11 s | 2.0 | `Lin2019/prion_model.bngl` |
| 2737 | 409 | 10 s | 10 s | 17 s | 16 s | 2.0 | `Barua2013/Barua_2013__PATCHED.bngl` |
| 2519 | 330 | 7 s | 7 s | 30 s | 30 s | 3.1 | `ode/MTORC1_assembly_v3.bngl` |
| 2519 | 330 | 7 s | 7 s | 31 s | 31 s | 3.1 | `ode/MTORC1_assembly_v3b.bngl` |
| 2484 | 257 | 5 s | 6 s | 11 s | 10 s | 2.1 | `nf/fcr.bngl` |
| 2484 | 257 | 6 s | 6 s | 10 s | 12 s | 2.1 | `FceRIviz/FceRI_viz.bngl` |

Source generation is where the model path is measurably slower, and only at
extreme size: 11.0 s against 4.7 s at 172,032 reactions. That is small beside
those models' compile times.

### Gaps in the model path

1. **Integer-valued stat factors above 2^64 (about 1.8e19).** `str(int(sf))` prints
   `1e+24` as a 25-digit integer literal, which C rejects
   (`_codegen.py` `generate_rhs_from_model`, `_emit_sens_rhs_body`, `_c_scalar`).
   The failure is loud; one corpus model hits it (`db_3rd_order_EnergyBNGL_v1`).
2. **#734 ExprTk spellings.** `_translate_expr_to_c` copies a lone `=`, `--` and
   an unparenthesised relational chain into C unchanged. The result is silently
   wrong. BNG2.pl parenthesises relational chains, so no corpus model hits it.

### Found along the way (same on both paths, so outside #803)

- **No GH #176 retry with a compiled Jacobian.** When the analytical Jacobian
  fails, `_run_ode_with_jacobian_fallback` retries with the FD one, but only
  without codegen. On `l-type-calcium-channel-dynamics` the interpreter recovers,
  while `codegen=True` raises `CV_ERR_FAILURE`. With `jacobian="fd"` the same
  `.so` agrees with the interpreter to the last digit (1071 steps, identical
  state), so the stated reason for the exclusion, "not re-selectable at run
  time", looks stale.
- **Sensitivity runs report a time switch's right-hand limit at the switch
  instant; plain runs report the left** (`ItalyModel_v7`). A fit that compares a
  step-function observable at integer times sees it shifted by one interval under
  sensitivities.
- **`BaruaBCR_2012` (24,388 reactions).** Its sensitivity build is 32.7 MB of C
  and sits at the 600 s compile timeout on both paths. `e7` (138.8 MB) is past it.
- **A failed compile is paid twice on the model path, for a sensitivity run of a
  model with 256 or more species.** `Simulator`'s large-model auto-codegen
  compiles first; if that fails, the sensitivity build compiles the same source
  again. `e7`'s model-path sensitivity cell took 1231 s to fail, against 601 s on
  the `.net` path, which skips auto-codegen. A successful build is not repeated
  (`BaruaBCR`: one compile). Step 3 routes `.net` models into that block.
- **`rint()` in initial conditions.** An initial condition such as `rint(N*f)`
  makes the analytic sensitivity in `f` exactly 0. Nothing warns that the
  response is not differentiable there.

### Caveats

- **Wall times are not absolute.** The machine was loaded, with up to four
  supervisors and, for part of the run, four orphaned BNG2.pl processes. The two
  paths' cells ran interleaved under the same load, so their ratio is
  meaningful; the absolute seconds are not.
- **The loader itself is not under test.** The interpreter reference and the
  model path both consume the same C++-loaded model. This measures codegen's
  fidelity to that model. The loader against BNG2.pl's own `run_network` is
  `../bng_parity`'s job.
- **MIR JIT backend.** It is not built here. It compiles the same generated
  source, so the source-level parity carries over, but it was not run.
- **Once step 3 routes `.net` models to the model path,** the `net` arms will
  measure nothing and should be retired.
