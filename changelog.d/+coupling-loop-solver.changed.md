- **Coupling loops keep their linear-solver state across steps.** Each
  `ReactionKernel.advance` / `Simulator.run_until` step restarted the GH #132
  factorization count, so a loop whose steps each factor a few times never
  reached the BLAS factor that `BNGSIM_LAPACK_DENSE=1` enables. A 361-species
  loop ran 2.5x slower than necessary. A `run_until` step now continues the
  count, while an independent `run()` still restarts it, and
  `n_dense_blas_factorizations` counts only the run's own factorizations. The
  warm path also keeps KLU's symbolic analysis across steps instead of redoing
  the ordering at every re-entry. Results are unchanged.
