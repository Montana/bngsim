- **SSA/PSA record a sample at a time trigger's threshold on the side ODE
  records it.** In `E1: at time > 2.5: c = 3` beside `species S in c = 3`, the
  trigger is false at 2.5 itself, and ODE and libRoadRunner read [S] = 3 in a
  row at 2.5. SSA and PSA read 1, the value after the event, at any seed, in a
  model with no rate rule and no rate that reads the clock: the event is placed
  to within 1e-12 above 2.5, and a row less than 1e-9 below an event was always
  deferred to read the state after it. Such a row is now deferred only where a
  trigger has already changed at the row's own time, as with `time >= 2.5`,
  which fires at 2.5 and reads 1 there, as in ODE, libRoadRunner, the SBML
  suite and DSMTS. A row 5e-10 before a `>=` event could read the value after
  it with a rate rule or a rate on the clock too, where the step that found the
  event held the row; it now reads the value before.
