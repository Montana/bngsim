- **A derivative with an integer past 2^64 in it is right again under both
  backends.** sympy printed the integers of a sensitivity derivative as bare C
  integer literals, so `k = p*10^20` gave dk/dp as `100000000000000000000`.
  `cc` refused to build that source (no forward sensitivity at all), and the
  MIR JIT read it as the 64-bit maximum, so dA/dp came back 82% small with no
  warning (99.99% for `p*6^30`). Integers are now printed as double literals,
  as the analytical-Jacobian printer already did; every other generated
  derivative computes the same doubles as before. An integer exponent that is
  not exactly a double (`p^(2^53+2)`, whose parity rounding would flip at
  p = -1) declines the derivative instead. The MIR JIT's prelude also defines
  `INFINITY`, which sympy prints for `oo`; the JIT refused it as undeclared.
