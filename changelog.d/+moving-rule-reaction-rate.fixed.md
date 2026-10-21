- **An SBML rule that reads a reaction's rate now counts as changing when that
  rate does.** A reaction's id in math is its rate, but bngsim took it for a
  constant. A conversionFactor `cf := J0`, J0 at `0.5*S1`, was frozen at its
  load value: S1(1) = 0.779 for libRoadRunner's 0.800. It is now refused by
  name, as any factor the run changes is. A compartment `cv := 1 + 0.5*J0`, J0
  at `k*A`, was given no rate of change, so its concentrations were not
  diluted ([Sn](2) = 19.67 for 25.80) and SSA ran its reactions at the
  load-time size (mean 19.82) without a word. The ODE now follows the rate
  through the law, including its jumps and events, and matches libRoadRunner;
  SSA refuses the reaction (`assignment_rule_compartment`), and a rule or law
  that reads a rate whose law reads a concentration in a resized compartment
  (`x' = J0` ran to 72.3 for 45.1). A species that no reaction, rule or event
  changes now holds its value, so a rule over it no longer refuses:
  `cf := 2*S3` loads, and SSA runs a compartment sized by `1 + 0.1*S`.
