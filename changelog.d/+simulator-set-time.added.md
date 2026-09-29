- **`Simulator.set_time(t)` and `set_state(state, time=t)` roll a stepped
  simulation back to a saved point.** A predictor-corrector coupling loop has to
  redo a step from its saved state *and* time. There was no public way to move
  the clock, so the corrector ran over the next interval, and on a model that
  reads the clock (a time-indexed table, `time()`) it saw the forcing a step
  late. `ReactionKernel.set_state` takes the same `time=`. Unlike `restore`,
  neither call rebuilds the backend, so the warm ODE solver is kept.
