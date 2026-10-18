- **An SBML conversionFactor the model gives no value is refused at load, by
  name.** A species' or the model's `conversionFactor` naming a parameter with
  no value, initialAssignment or assignmentRule was read as 1, so the model ran
  as if it had no factor: `S1 -> S2` at `0.5*S1` gave S1(1) = 0.607 and no
  warning. It now raises `UnderSpecifiedModelError` naming the parameter and the
  species, as for a valueless parameter a rate law reads, and as RoadRunner does
  (`BNGSIM_ALLOW_UNSET_PARAMS=1` loads it with the factor at 0). A factor that
  names no parameter, or whose assignment cannot be evaluated at load, is
  refused too: the first was read as 1, and a declared 5 whose initialAssignment
  gave 2 scaled the rates by 5. A factor with a value, or one an
  initialAssignment or rule gives, loads as before. `p = rateOf(S)` in an
  initialAssignment now includes S's factor: with a factor of 2 it gave -0.5
  where -1 is right.
