- **An SBML initial value of `rateOf(x)` loads where x's rate itself reads
  `rateOf`.** The load works `rateOf(x)` out as a number, from x's rate rule or
  the kinetic laws that change it, and gave a `rateOf` inside those no value: in
  `P0 = rateOf(p)` with `p' = 0.5*rateOf(S3)`, and S3 made at 0.4, the model was
  refused at load where the run and libRoadRunner both read 0.2. A `rateOf` in a
  rate rule or kinetic law is now worked out the same way, through a conversion
  factor or a compartment's dilution as at the top, however deep the chain (1,500
  rate rules, each on the `rateOf` of the one before, fold to 1). What the inner
  rate reads is frozen with the fold, so `set_param` on it is refused rather than
  leaving the value stale. Rates written in terms of each other in a ring are
  still refused, as the run refuses them.
