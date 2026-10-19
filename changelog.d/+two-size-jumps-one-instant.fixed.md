- **SBML: a compartment size that reads a concentration a piecewise size jump
  moves now jumps with it.** Its concentrations kept the values from before
  the jump, with no warning. Beside `cv := piecewise(1, time < 2, 5)` and
  [Sn] = 5, `cw := piecewise(1, time < 2, 1 + Sn)` took [Sw] from 3 to 0.5,
  its new size read with [Sn] from before cv's jump, where libRoadRunner gives
  1.5; `cw := 1 + Sn`, which has no jump of its own, left [Sw] at 3 for 9. The
  same held in a chain of three, on a species threshold, at two equal
  thresholds, and where [Sn] is read only in cw's condition, there with a
  result that changed with the seed. Each jump now rescales the concentrations
  of every compartment whose size reads what it moves, as an event's resize
  does, after that compartment's own jump, interpreted and codegen alike.
  Forward sensitivities follow; a concentration rescaled twice at one instant
  is refused. (#1054)
