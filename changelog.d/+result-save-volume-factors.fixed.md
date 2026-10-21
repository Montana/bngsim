- **A result saved and loaded again reports the amounts and concentrations it
  had.** `Result.save` kept the stored species values but not the compartment
  sizes, nor the per-sample factors that `as_roadrunner` applies where a size
  changes, so a loaded result read every compartment at size 1 and fixed. A
  species of 4 in a compartment of size 2 had an amount of 4 for 8, and under
  SSA, `[S]` in a compartment growing from 2 to 3.5 read 13.5 for 7.71. The
  file (format 3) now holds both. A file written before this does not say what
  the sizes were, so `as_roadrunner` on it refuses an amount selector, naming
  the reason, and warns on a concentration one. A reloaded result also refuses
  the output sensitivity of an assignment-rule species that a changing size
  rescales, as the run does, where it returned the species' NaN row.
