- **`run_batch` dropped `reaction_stats`, and three SSA entry points lost the
  compiled propensities.** With `reaction_stats=True`, every `run_batch` row
  came back with no firing counts or propensity integrals. With codegen left on,
  `run_replicates(num_processors > 1)`, `run_batch` and any run after
  `intervene` or `restore` never got the propensity library the Simulator
  built, so on a build with a C compiler they ran the slower interpreted loop
  (MIR-JIT'd on a MIR build) while `run()` reported `"cc"`. Every SSA engine is
  now built with the Simulator's settings, so all four report the same backend
  and the parallel replicates match the sequential ones bit for bit. Also,
  `BNGSIM_SSA_NO_CODEGEN`, `BNGSIM_SSA_PROP_CC` and `BNGSIM_SSA_PROP_JIT` set to
  `0` are now off in Python too, as the C++ engine already read them; before,
  `0` kept Python from building the library.
