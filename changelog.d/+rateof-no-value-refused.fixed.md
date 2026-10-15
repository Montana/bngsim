- **An SBML model that reads a `rateOf` it has no value for is refused at load,
  by name.** Beside the assignmentRule-variable case refused since #1046:
  - `rateOf(S)` of a concentration in a compartment an assignmentRule sizes
    (validation rule 10225). An initial value left out the size's change; a rule
    had it where the size reads only the time, and read 0 where it reads a
    species. A species with a rate rule of its own, or a constant one, still
    loads with its rate.
  - `rateOf` of anything but a single symbol (rule 10223). The csymbol was
    already refused in an initial value and read as 0 with a warning in a rule;
    through COPASI's rateOf functionDefinition it was NaN or 0, with no warning.
  - A call to a functionDefinition whose body reads `rateOf`, which took the rate
    of the function's own parameter, 0, rather than of what the caller passed. A
    functionDefinition nothing calls is left alone.

  The error names each, where it is read, and why.
