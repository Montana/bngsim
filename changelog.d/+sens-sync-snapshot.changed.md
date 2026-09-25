- **Forward sensitivities no longer re-derive every derived parameter on every
  RHS call.** Each call of a sensitivity run synced the model's parameters from
  CVODES and re-evaluated all derived parameters, although every call but a
  difference-quotient probe lands on the same nominal point. The sync now copies
  a snapshot of that point, and a probe re-derives only the derived parameters
  that read the probed one, directly or through another derived parameter. The
  results are bit-identical. On the committed ODE suite with five sensitivity
  parameters, runs are up to 2.1x faster (egfr_net_red, tcr_signaling,
  oscillatory_system); a difference-quotient run on a model with 400 derived
  parameters went from 13.8 ms to 6.3 ms, and its analytic-path twin from
  4.3 ms to 3.65 ms. A model whose derived parameters read `time()` or call
  anything but a stateless math built-in keeps the full re-derivation.
