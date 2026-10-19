- **A loaded result reports the initial-condition seed and the SSA and PSA
  diagnostics its run did.** `Result.save` wrote none of them. A run with
  `sensitivity_params=["A0", "k"]` whose `ic_sensitivity_seed` was
  `{'A': {'A0': 1.0}}` loaded back as `None`, "not recorded", and a recorded
  `{}`, "nothing seeds", did too. An SSA run of `species A = 3.5` that rounded
  1 population on the `cc` backend loaded back as 0 on `unknown`, and a PSA run
  with a measured speedup of 32.7 as inactive. A file written before this loads
  them as before. `Result.squeeze` also gave every row of a stack the first
  row's compartment sizes, so a batch over a size of 2 and 5 held 2 for both,
  and dropped the per-sample volume factors. `as_roadrunner` refuses a stack,
  so no output read them, but the stack and its saved file now hold each
  row's own.
