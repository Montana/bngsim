- **SSA held the concentration of a species that a rule sets across an event's
  resize, where SBML keeps its amount.** SSA leaves a molecule count alone when
  an event resizes its compartment, and it treated a species that an assignment
  rule or a rate rule sets the same way, though the rule makes it a
  concentration, not a count. With `c` going from 1 to 3 at t = 2.5, `S := 3`
  reported [S] = 1 and S = 3 for 3 and 9, and `S' = 0.4` gave [S](3) = 0.733
  for 0.867. In `c := p` with `p` going from 2 to 4, [S](5) was 1.5 and S was
  12, for 2 and 8. SSA and PSA now rescale such a species at the resize as ODE
  does, and report it as a concentration: all three match ODE and
  libRoadRunner exactly. As in ODE, the rescale is formed when the event
  executes, so it keeps what an event of higher priority at the same instant
  wrote to the species first: [S](3) = 1.867 there, for 1.733 before.
