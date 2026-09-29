- **Large rule-derived networks now factor with the BLAS dense solver instead of
  KLU.** The linear-solver rule tested the Jacobian's density, but KLU factors
  `I - gamma*J`, which on these networks fills 20-35% of `n²` while the
  Jacobian is 1-7% dense. KLU has no BLAS, so it spent most of a run
  refactoring. A model of 256-5,000 species whose estimated LU fill is at least
  10% now takes the BLAS dense factor, if the build has one
  (`bngsim.HAS_LAPACK_DENSE`) and the analytical Jacobian is in use. That is
  3.6-10x faster end to end on the suite's rule-derived networks (fceri_gamma
  47.4 s to 4.7 s); trajectories agree to solver tolerance. Sparse networks
  (metapop_sir_100) stay on KLU, and `force_sparse_linear_solver=True` keeps
  any model there. `jacobian_sparsity` reports the estimate as
  `lu_fill_estimate`.
