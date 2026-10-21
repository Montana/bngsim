- **SBML: a compartment sized through a diamond of assignment rules loads in
  linear time.** Where the rules a size reads share rules (`r_i := r_{i-1} +
  r_{i-2}`, each read by the next two), the loader parsed and differentiated
  each rule once per path to it, for the size's rate (the dilution term of a
  concentration in the compartment) and for the jumps of a piecewise size: 22
  such rules took 17.6 s to load, and each four more about 7 times as long.
  Each rule is now parsed and differentiated once: 0.03 s at 22 rules, 0.11 s
  at 80. With non-linear rules (`r_i := sin(r_{i-1}) + cos(r_{i-2})`) the term
  and the jump's rescale were also written out once per path, 22,122 and
  44,839 characters at 16 rules. A rate that several rules read is now a
  function of its own, `_ddt_<rule>`, and the jump's sizes read the rules by
  name, so the text grows with the rules: 542 and 81 characters. The results
  are the same.
