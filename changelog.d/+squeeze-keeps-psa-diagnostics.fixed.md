- **A stacked batch of PSA runs keeps each run's diagnostics, and no longer
  reads as exact SSA.** `Result.squeeze` (and `squeeze=True`) dropped every
  run's `psa_diagnostics`, so a batch whose rows all used partial scaling
  reported `active: False`. Each run's report is now kept as it was, in
  `Result.psa_diagnostics_per_run`, which a single run answers with its one
  report, and it survives `save` and `load`. The stack's own report says only
  what needs no choice of statistic: `active` if any run used partial
  scaling, the largest `peak_population`, and whether any run crossed the
  activation level; the per-reaction statistics and `speedup` are left empty
  rather than averaged or summed.
