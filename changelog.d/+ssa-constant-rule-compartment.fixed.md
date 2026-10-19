- **SSA runs a cross-compartment reaction out of or into a compartment an
  assignment rule holds constant.** `A => B; k*A` with A in `C := 2` and B in
  a compartment a rate rule grows was refused (`varvol_non_mass_action`), where
  the same model with `C = 2` declared constant ran, and so was `C*k*A`. A
  rule over constants never moves the size, as a write to what it reads is
  refused, so such a reaction now runs as its static twin does, the same
  trajectory for the same seed, and a law with the size as a factor folds it
  as the twin's does. Over 3,000 runs the mean count of A and the mean [B] are
  within 0.7 standard errors of the exact mean at every sample, and a
  bimolecular `A + B => P` matches an Extrande sampler. A rule that reads the
  clock, a species, or a rate-rule or event target is still refused.
