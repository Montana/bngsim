- **SSA ran, and got wrong, a reaction in a compartment sized by a rule that
  reads a constant species in a compartment whose size changes.** A resize keeps
  a species' amount, so the concentration of a constant or boundary species
  moves with its compartment's size, though no rule or event names it. SSA took
  `cv := 1 + 0.1*Kw`, with Kw constant in a compartment `ce` that an event
  doubles, for a size that holds, and ran `J: Sn => ; k*Sn` at cv's size at
  load: the mean [Sn](1.5) of 400 runs was 17.25 for libRoadRunner's 21.97.
  SSA now refuses the reaction by name (`assignment_rule_compartment`), as it
  refuses one in any assignment-rule compartment whose size moves; a rate rule
  on `ce` is refused the same way. The ODE result for this model is also wrong
  (17.18; issues #745 and #1055).
