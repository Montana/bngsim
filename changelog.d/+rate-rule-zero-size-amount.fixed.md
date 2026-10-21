- **An amount species in a rate-rule compartment keeps its amount where the
  size is exactly 0.** `as_roadrunner(["H"])` for a conserved 5 in
  `c' = -1` from 2 read [5, 5, 0, 5] at t = 0, 1, 2, 3. At t = 2 the
  concentration is not divided by the zero size and stays amount/V_static, but
  the amount was that times the live size, 0. It is that times V_static there
  now, so it reads 5, as #728's review made it in an assignment-rule
  compartment. An event that resizes a compartment to 0 was right already.
