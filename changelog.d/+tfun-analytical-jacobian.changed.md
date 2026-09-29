- **Rate laws that read a time- or parameter-indexed table function get the
  analytical Jacobian.** Such a table does not depend on the state, but the
  symbolic core did not recognize it, so any model using one fell back to a
  finite-difference Jacobian. On a 1,319-species network driven by a time
  table, a 12 h run went from 23.4 s to 2.5 s. The compiled Jacobian covers a
  table used as a whole function body (`f() tfun(...)`); a table embedded in
  arithmetic uses the interpreted analytical Jacobian. Tables indexed by an
  observable still use finite differences. Sensitivities are unchanged.
