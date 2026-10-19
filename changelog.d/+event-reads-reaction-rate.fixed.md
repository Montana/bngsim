- **An SBML event that reads a reaction's rate now loads.** A reaction's id in
  math is its rate, and bngsim registers it as one for rules and kinetic laws
  that read it, but not for events: `at J0 < 15: B = 50`, or an assignment
  `B = J0`, failed to load with `J0` an undefined symbol. Both load now and
  match libRoadRunner (B = 50 from t = 1.438 for J0 = 20·e^(-0.2t); B = 16.375
  from t = 1). Under SSA, a trigger that reads a rate whose law reads a
  concentration in a resized compartment is refused by name
  (`variable_compartment_read`), as one that reads the concentration is.
