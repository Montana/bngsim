- **An SBML conversionFactor the model gives no constant value is refused at
  load, by name.** A species' or the model's `conversionFactor` naming a
  parameter with no value, initialAssignment or assignmentRule was read as 1, so
  the model ran as if it had no factor: `S1 -> S2` at `0.5*S1` gave S1(1) = 0.607
  and no warning. It now raises `UnderSpecifiedModelError` naming the parameter
  and the species, as for a valueless parameter a rate law reads, and as
  RoadRunner does (`BNGSIM_ALLOW_UNSET_PARAMS=1` loads it with the factor at 0).
  A factor that names no parameter, or whose assignment cannot be evaluated at
  load, is refused too: the first was read as 1, and a declared 5 whose
  initialAssignment gave 2 scaled the rates by 5. So is one that a rate rule, an
  event or a moving assignment rule changes during the run, which SBML forbids:
  the rates kept its load value (S1(1) = 0.368 with `cf' = 1`, where RoadRunner
  gives 0.287). A factor with a value, or one an initialAssignment or a rule
  over constants gives, loads as before.
