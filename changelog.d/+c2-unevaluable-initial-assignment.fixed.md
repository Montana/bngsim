- **A parameter with no value, set by an initialAssignment that cannot be
  folded at load, now loads instead of being refused as having "no
  initialAssignment".** `k = 0.5 + S3`, with S3 given no initial value, was
  refused as under-specified, though it has an initialAssignment and
  libRoadRunner runs the model with k = 0.5. bngsim now evaluates the
  assignment at the start, with S3 at 0 and a warning that names it, as it
  already did when k declared a value: `S1 -> S2` at `k*S1` gives
  S1(1) = 0.607, as libRoadRunner does. A parameter the assignment reads that
  has no value is still refused, and is now named alone. A conversionFactor
  whose initialAssignment has no math was read as 0 without a word; it is now
  refused like any factor with no value.
