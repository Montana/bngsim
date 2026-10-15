- **SBML `rateOf` of a parameter or compartment that no rule sets is 0 wherever
  it is read.** An assignment or rate rule, a kinetic law or an event that read
  `rateOf(p1)`, with `p1` a parameter no rule changes (constant or not, or set
  only by an initialAssignment), failed to load with an ExprTk error about an
  undefined `rate_of__p1`; so did `rateOf(c)` of a compartment nothing resizes.
  The rate is 0 at every time, which an initialAssignment already gave it
  (lanl/bngsim#1045); now a rule, law or event does too.
