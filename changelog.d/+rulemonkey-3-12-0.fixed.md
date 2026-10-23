- **`method="rm"` runs on RuleMonkey 3.12.0.** A function that reads one
  listed after it in the XML no longer reads a stale value, a rate law that
  reads `reactant_N()` through a helper function fires, and a table function
  indexed by a function no longer crashes. A model that reads the clock keeps
  its waiting times from a large `t_start`, which is the `rm` half of issue
  #1059, and a `RuleMonkeySession` that runs out of events still reaches
  `t_end`. Arrhenius rules RuleMonkey used to misread stay refused until
  issue #1058.
