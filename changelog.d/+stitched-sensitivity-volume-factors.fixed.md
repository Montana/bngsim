- **`compute_all_sensitivities` reports amounts and concentrations as `run()`
  does in a compartment whose size changes.** The stitched result was rebuilt
  from a list of fields that left out the live sizes `Result.as_roadrunner`
  reads, so a conserved amount of 2 in a compartment growing at `c' = 0.2` read
  2, 1, 0.667, a conserved 5 in `c := 2 + time` read 5, 3.33, 2.5, and after an
  event doubled a compartment a concentration species' amount read half, with
  nothing said. The columns and the sensitivity rows were right. The stitched
  result is now the first chunk's with the parameter axis widened, so it
  carries everything each chunk's stamp attached, `Result.state` included, and
  matches a single `run(sensitivity_params=...)` in rate-rule, assignment-rule
  and event-resized compartments and for rule-target species. (#787)
