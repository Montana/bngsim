- **`ReactionKernel.observables()` no longer returns the previous step's values
  after the state changes.** Once `advance()` had run, `observables()` read the
  step result even after `set_state()` replaced the state, contrary to its
  docstring. For example, a model with A = 100 reported the old step's 6.07.
  The same happened after an `advance()` that raised `StopConditionMet`, which
  moves the state without storing a new result. Both now recompute from the
  live state. The case matters for the predictor-corrector rollback,
  `set_state(saved, time=t0)`. `last_result` is unchanged and still describes
  the step it came from.
