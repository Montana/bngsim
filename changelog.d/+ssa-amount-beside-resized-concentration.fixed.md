- **SSA/PSA run an SBML monomial across compartments that mixes an amount with
  a concentration in a resized compartment again.** `A => B; k*A`, A an amount
  and B a concentration in a compartment of A's size at load that a rate rule
  or an event resizes, ran under SSA while its ODE divided B's row by the
  load-time size, and was refused once the ODE was right. Its propensity reads
  the amount, and each concentration factor gets the live-volume term a
  monomial over concentrations gets; it matches Extrande and the exact mean.
  With different sizes at load it is still refused, as it was before. (#787)
