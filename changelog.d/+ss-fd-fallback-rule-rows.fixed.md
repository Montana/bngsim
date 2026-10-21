- **A steady-state expression sensitivity is no longer NaN because an
  assignment rule's target sits beside it.** Where the compiled evaluator
  declines a function (an `if()`, a `max()`), `steady_state()` differences
  it, and the difference multiplied the NaN row of every species the solve
  leaves out into the function's row, also where the function does not read
  that species. `W := piecewise(P, P < 100, 2*P)` beside `T := 2*P` gave NaN
  where d(W)/d(kp, kd) is (5, -25); a species `mask=` leaves out did the
  same. Such a species now adds nothing to a function that does not read it,
  and a function that does read one keeps its NaN. An amount in a
  compartment whose size such a function sets was refused with a message
  that blamed `mask=` when none was given. It now gets its row, and the
  refusal that remains names the species the size reads and why the solve
  has no row for it.
