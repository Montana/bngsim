- **`steady_state()` reports an amount in a compartment whose size changes
  over the live size, as `run()` does** (part of #787). An amount-valued
  species in a compartment a rate rule or an assignment rule sizes, one a rule
  or a rate rule sets among them, came back over the size the model loaded
  with: `H`, an amount of 2 in `C := 1 + P` with P settling at `kp/kd` = 5,
  read 2.0 where `run()` to the same state and RoadRunner give 2/6 = 0.333.
  Its `dY_ss/dp` row was the stored value's: d[H]/dks 2 for 1/3, and
  d[H]/dkp, which moves only the size, 0 for -0.278. `steady_state()` and
  `steady_state_batch()` now run the passes `run()` does. Where `mask=` leaves
  the size out of the solve and keeps such a species, its row is refused by
  name.
