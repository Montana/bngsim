- **Forward sensitivity of a log-parameterised rate constant (`k = 10^logk`)
  now runs on the MIR JIT.** sympy writes dk/dlogk as `pow(10, logk)*M_LN10`
  (`M_LN2` for base 2, and so on), and the generated sources defined only
  `M_PI` and `M_E`. The JIT strips `<math.h>`, so with `BNGSIM_CODEGEN_JIT=mir`
  such a run stopped with "undeclared identifier M_LN10" (the benchmark models
  `nfkb` and `RasRaf_stdBNGL_v3` among them). Each standalone generated source
  now defines every `<math.h>` constant sympy's C printer can write, with
  `<math.h>`'s own digits and `#ifndef`-guarded, which also covers MSVC, where
  `<math.h>` defines none of them without `_USE_MATH_DEFINES`.
