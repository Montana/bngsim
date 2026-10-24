- **The MIR JIT no longer binds a rate-law function it has no C for to an
  unrelated symbol, or exits the process over one.** The emitter writes some
  ExprTk functions through as bare calls (`equal`, `frac`, `inrange`,
  `iclamp`, `roundn`, `root`, `logn`, `sinc`, a table function's name). `cc`
  refuses such a source, but c2mir declares the call implicitly as `int
  name()`, so with `BNGSIM_CODEGEN_JIT=mir` the name went to the import
  resolver. When nothing exported it, MIR called `exit(1)` and the Python
  process ended mid-run. When something did, the run used it: a framework
  macOS loads into Python exports an `equal`, and `k*(1 + equal(k, 0.5))` gave
  B(1) = 0.787 where the answer is 1.264, with no warning. `MirJit` now
  refuses any import outside its prelude by name, as a `SimulationError` from
  `run()`. Only builds configured with `-DBNGSIM_ENABLE_MIR=ON` are affected.
