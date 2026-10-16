- **An SBML model that reads a `rateOf` it has no value for is refused at load,
  by name.** Beside the assignmentRule-variable case refused since #1046:
  - `rateOf(S)` of a concentration in a compartment an assignmentRule sizes
    (validation rule 10225), a constant or boundary species included, since SBML
    holds its amount and its concentration moves with the size. An initial value
    left out the size's change; a rule had it where the size reads only the time,
    and left it out where the size reads a species. A species with a rate rule of
    its own still loads with that rate.
  - `rateOf` of anything but a single symbol (rule 10223). The csymbol was
    already refused in an initial value; in a rule it, or a call to COPASI's
    rateOf functionDefinition, read 0 with a warning, and the functionDefinition
    in an initial value gave NaN without one.
  - A call to a functionDefinition whose body reads `rateOf`, which took the rate
    of the function's own parameter, 0, rather than of what the caller passed. A
    functionDefinition nothing calls is left alone.

  The error names each, where it is read, and why.
