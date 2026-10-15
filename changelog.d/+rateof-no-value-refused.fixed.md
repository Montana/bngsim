- **An SBML model that reads a `rateOf` it has no value for is refused at load,
  by name.** Three cases loaded with a wrong rate and nothing said: `rateOf(S)`
  of a concentration in a compartment an assignmentRule sizes (validation rule
  10225), which left out the size's change; `rateOf` of anything but a single
  symbol (rule 10223), read as 0, or as NaN through COPASI's rateOf
  functionDefinition; and a call to a functionDefinition whose body reads
  `rateOf`, which took the rate of the function's own argument, 0, rather than
  of what the caller passed. The error names each, where it is read, and why,
  beside the assignmentRule-variable case refused since lanl/bngsim#1046. A
  functionDefinition nothing calls is left alone.
