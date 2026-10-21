- **`run()` gives the sensitivity row of an amount an assignment rule sets in a
  compartment whose size changes, and the expression sensitivities of a model
  that reads a rate** (part of #787). `T := 2*P` in `C := 1 + P` reports
  `2P/(1 + P)`, and its row was NaN, listed in `ar_sensitivity_refused`. It is
  now the rule's row put through the quotient rule with the size's, as in
  `steady_state()`: at t = 1, d[T]/dkp is 0.4988 and d[T]/dkd -0.2411, as
  central differences of runs give. One function that read `rateOf` made
  codegen decline the expression sensitivities of the whole model, and the
  dilution term of a concentration in `C := 1 + P` reads the rate of `P`, so
  no expression there had one, `C` included. Now only the functions that read
  a rate, and those that depend on them, are declined: dC/dkp at t = 4 is
  2.7534. A row is still refused by name where codegen declines the rule's or
  the size's own sensitivity.
