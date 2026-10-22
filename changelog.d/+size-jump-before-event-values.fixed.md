- **SBML: an event at the instant a piecewise compartment size jumps reads the
  concentrations after the jump.** Its trigger, its values and its delay were
  read before the jump's rescale executed, from the concentration the size
  before gave, with no warning. With `cv := piecewise(1, time < 2, 5)` and
  [Sn] = 5, `at (time >= 2): Sn = Sn + 1` left Sn at 6 and `X = Sn` set X to
  5, where [Sn] at t = 2 is 1 and libRoadRunner gives 2 and 1; a delay read
  from Sn ran 5 times too long, and `time >= 2 && Sn > 3` fired though it is
  false at t = 2. The loader marks its jump events and the ODE engine executes
  them first. Forward sensitivities follow; values frozen between two events'
  fires at that instant are refused. `steady_state()` named the jump's two
  events as "it has 2 events" for a model with none; it names the compartment
  now. (#1054)
