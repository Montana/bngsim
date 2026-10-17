"""bngsim._diffrax_solver — Diffrax ODE solver (pure JAX, no CVODE).

Runs the entire ODE solve in JAX/XLA — RHS, Jacobian (AD), and the
adaptive stepper are all JIT-compiled into a single fused computation.
No Python↔C++ boundary crossings during the solve.

Uses Kvaerno5 (5th-order implicit Runge-Kutta, L-stable) for stiff
biochemical systems, with PID step size control; up to a switch on the clock,
Kvaerno3 (see below).

The RHS, the parameter values and the initial state all come from the built
model (issue #803 step 4). They used to be re-read from the ``.net`` text by
codegen's private parser: a synthesis reaction's rate was multiplied by the last
species, a model without observables crashed, and an initial concentration
written as an expression was looked up as a parameter name.

A rate law that switches on the clock, ``if(time() > t*, ...)``, jumps there,
and the run is split at each such time and restarted just past it (issue #865),
each piece before one solved with Kvaerno3, whose stages stay within its step:
stepped across, a jump cost several times CVODE's error, and at a tight
tolerance the whole step budget. A switch on the model's state, whose times are
not known before the run, is refused, as events are.

Optional dependency: ``pip install diffrax``.
"""

from __future__ import annotations

import logging

import numpy as np

from bngsim._jax_rhs import (
    _as_model,
    check_rhs_against_engine,
    generate_jax_rhs,
    jax_available,
)

logger = logging.getLogger("bngsim")

# ─── Availability check ────────────────────────────────────────────

_DIFFRAX_AVAILABLE: bool | None = None


def diffrax_available() -> bool:
    """Check if diffrax is importable (cached)."""
    global _DIFFRAX_AVAILABLE
    if _DIFFRAX_AVAILABLE is None:
        if not jax_available():
            _DIFFRAX_AVAILABLE = False
        else:
            try:
                import diffrax  # noqa: F401

                _DIFFRAX_AVAILABLE = True
            except ImportError:
                _DIFFRAX_AVAILABLE = False
    return _DIFFRAX_AVAILABLE


# ─── Diffrax solver ─────────────────────────────────────────────────


def run_diffrax(
    model,
    param_dict: dict[str, float] | None = None,
    t_start: float = 0.0,
    t_end: float = 100.0,
    n_points: int = 101,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 100000,
) -> dict:
    """Run an ODE simulation using Diffrax (pure JAX).

    Parameters
    ----------
    model : Model or str
        The built model, or a ``.net`` path loaded with ``Model.from_net``.
    param_dict : dict[str, float], optional
        Parameter overrides, applied with ``set_param`` to a clone of the model,
        so derived parameters and parameter-valued initial conditions follow.
        Every other parameter keeps the model's value. Naming a derived
        parameter pins it at the value given. The primary parameters are applied
        first and the derived ones after, whatever the order of the keys, so a
        dict naming every parameter -- this function's contract before issue
        #803 -- runs on exactly those values.
    t_start, t_end : float
        Time interval.
    n_points : int
        Number of output time points.
    rtol, atol : float
        Solver tolerances.
    max_steps : int
        Maximum internal solver steps: of the run, or, for a run split at its
        switches, of each piece and each solver tried on it. ``n_steps`` counts
        every step taken, a piece's failed attempt included.

    Returns
    -------
    dict
        Keys: 'time' (n_points,), 'species' (n_points, n_sp),
        'species_names' list[str], 'n_steps' int.

    Raises
    ------
    ValueError
        If the model has events, or a rate law that switches on the model's
        state, neither of which this solver implements; if its switches on the
        clock cannot all be placed (see :func:`_switch_times`); or if it has a
        construct the JAX RHS does not implement or misdescribes (see
        :func:`bngsim._jax_rhs.check_rhs_against_engine`).
    """
    if not diffrax_available():
        raise ImportError(
            "Diffrax is required for method='diffrax'. Install with: pip install diffrax"
        )

    import diffrax
    import jax
    import jax.numpy as jnp

    # A clone, put back to its initial state: the run starts from the initial
    # state (get_initial_state below), and its switch times have to be of that
    # state too. A counter clock's crossing is placed from the counter's present
    # value, which a run on the model before this one has moved, and then no
    # switch was found and the run stepped across it (issue #865).
    model = _as_model(model).clone()
    model.reset()
    if param_dict:
        derived = {
            n for n, e in zip(model.param_names, model.param_is_expression, strict=True) if e
        }
        # Primaries first: a derived parameter set before one of its inputs would
        # be recomputed over (set_param re-derives), and a key order would decide.
        ordered = sorted(param_dict.items(), key=lambda kv: kv[0] in derived)
        for name, value in ordered:
            model.set_param(name, value)
    if model.n_events:
        raise ValueError(
            f"run_diffrax does not implement events, and this model has {model.n_events}; "
            "use Simulator(method='ode')."
        )
    state_switches = model.state_switch_root_conditions()
    if state_switches:
        raise ValueError(
            "run_diffrax does not integrate a rate law that switches on the model's state, "
            "whose switch times are not known before the run, and this model has "
            f"{list(state_switches)}; use Simulator(method='ode')."
        )
    switch_ts = _switch_times(model, float(t_start), float(t_end))
    rhs = generate_jax_rhs(model)
    span = float(t_end) - float(t_start)
    check_rhs_against_engine(
        model, rhs, times=[float(t_start) + f * span for f in (0.0, 0.37, 0.81, 1.0)]
    )

    # Parameters in the model's order, derived ones at their evaluated values --
    # the array the RHS indexes -- and the model's own initial state.
    core = model._core
    params = jnp.array([core.get_param(n) for n in core.param_names], dtype=jnp.float64)
    y0 = jnp.array(np.asarray(core.get_initial_state()), dtype=jnp.float64)
    sp_names = list(model.species_names)

    # Output times
    ts = jnp.linspace(t_start, t_end, n_points)
    out_ts = np.asarray(ts)

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    term = diffrax.ODETerm(lambda t, y, args: rhs(y, t, args))

    # Issue #865. The run is split at the times its rate laws switch at
    # (_switch_times): each piece ends at the float just before a switch, and the
    # next begins at the float just after it, so that no step is taken across the
    # jump (to within the few floats a crossing the engine scans is placed to).
    # Kvaerno5 reads one of its stages at 1.23 steps, past the end of the step,
    # and in a piece's last step that is past the switch: a step that ends just
    # before one read the other branch, and was turned down until the steps ran
    # out. That held for diffrax's own jump_ts as well. So a piece that ends at a
    # switch is solved with Kvaerno3, whose stages all lie within its step
    # (c <= 1), and the last piece, and a run with no switch, with Kvaerno5 as
    # before. Reading the time clamped to the piece instead kept the branch, but
    # froze the rate laws' smooth dependence on the time in each last step, past
    # the tolerance, which the error estimate does not see. One compiled solve per
    # solver serves every piece: its bounds, start state and output times are
    # arguments, the output times padded with the piece's end, the last of which
    # is the state the next piece starts from.
    #
    # Kvaerno3 cannot take a step as short as some pieces are: its Newton
    # iteration reads roundoff as divergence, the step is turned down and shrunk,
    # and so on. A switch 1e-6 after t_start, or 1e-12 before t_end, made such a
    # piece. A piece next to a switch is solved under a least step of 128 floats
    # of the run's time, so such a solve fails within a few dozen steps and not
    # max_steps, and again with Tsit5, which has no Newton iteration and whose
    # stages also lie within its step. A run with no switch is one piece, with
    # main's solve exactly, raising what it raised.
    def _compiled(solver, controller, throw):
        @jax.jit
        def _solve(a, b, y_a, seg_ts, params):
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=a,
                t1=b,
                dt0=jnp.minimum((b - a) / 100.0, 0.01),
                y0=y_a,
                args=params,
                saveat=diffrax.SaveAt(ts=seg_ts),
                stepsize_controller=controller,
                max_steps=max_steps,
                throw=throw,
            )
            return sol.ys, sol.stats["num_steps"], sol.result == diffrax.RESULTS.successful

        return _solve

    if switch_ts:
        scale = max(abs(float(t_start)), abs(float(t_end)), 1.0)
        piece_controller = diffrax.PIDController(
            rtol=rtol, atol=atol, dtmin=128.0 * np.finfo(np.float64).eps * scale
        )
        solve_last = _compiled(diffrax.Kvaerno5(), piece_controller, throw=False)
        solve_to_a_switch = _compiled(diffrax.Kvaerno3(), piece_controller, throw=False)
        solve_explicitly = _compiled(diffrax.Tsit5(), piece_controller, throw=False)
    else:
        solve_last = _compiled(diffrax.Kvaerno5(), stepsize_controller, throw=True)

    # Piece i runs from just past switch i-1 to just short of switch i. A switch
    # is a group of crossings within the ODE path's one instant (_same_instant),
    # its lo the first and its hi the last; the piece before it ends short of lo
    # and the one after begins past hi, so no piece is turned backward or to
    # nothing by two crossings a float apart.
    n_pieces = len(switch_ts) + 1
    species_out = np.empty((n_points, len(sp_names)), dtype=np.float64)
    y_a = y0
    n_steps = 0
    for i in range(n_pieces):
        a = float(t_start) if i == 0 else float(np.nextafter(switch_ts[i - 1][1], np.inf))
        last = i == n_pieces - 1
        b = float(t_end) if last else float(np.nextafter(switch_ts[i][0], -np.inf))
        # A piece reports the output times up to the end of its switch, the first
        # also t_start. One on a switch is the piece's before it, read at the float
        # just before: the state does not jump there, only its derivative. A run
        # with no switch is one piece, forward or backward, and reports them all.
        if n_pieces == 1:
            mine = np.arange(n_points)
        else:
            after = switch_ts[i - 1][1] if i else -np.inf
            upto = float(t_end) if last else switch_ts[i][1]
            mine = np.flatnonzero((out_ts > after) & (out_ts <= upto))
        seg_ts = np.full(n_points + 1, b)
        seg_ts[: mine.size] = np.clip(out_ts[mine], min(a, b), max(a, b))
        if i > 0:
            # A counter clock reaches its threshold at the switch only to within
            # the integration's error, and a few floats short the next piece read
            # the branch that had just ended: a dozen rejected steps, and an
            # accepted one on the wrong branch. It is put just past its threshold,
            # as the core puts it before CVODE restarts (issue #82). (A counter
            # that runs ahead reads the next branch before the piece ends instead,
            # which this cannot repair.)
            y_np = np.array(y_a, dtype=np.float64)
            for idx, threshold in switch_ts[i - 1][2]:
                y_np[idx] = max(y_np[idx], float(np.nextafter(threshold, np.inf)))
            y_a = jnp.asarray(y_np)
        args = (
            jnp.float64(a),
            jnp.float64(b),
            y_a,
            jnp.asarray(seg_ts, dtype=jnp.float64),
            params,
        )
        ys, n, ok = (solve_last if last else solve_to_a_switch)(*args)
        n_steps += int(n)
        if switch_ts and not bool(ok):
            ys, n, ok = solve_explicitly(*args)
            n_steps += int(n)
            if not bool(ok):
                raise ValueError(
                    f"run_diffrax could not integrate the piece of the run from {a!r} to "
                    f"{b!r}, next to a switch, with an implicit or an explicit solver within "
                    f"max_steps={max_steps}; use Simulator(method='ode')."
                )
        ys = np.asarray(ys)
        species_out[mine] = ys[: mine.size]
        y_a = jnp.asarray(ys[-1])

    return {
        "time": np.asarray(ts),
        "species": species_out,
        "species_names": sp_names,
        "n_steps": n_steps,
    }


def _switch_times(model, t_start: float, t_end: float) -> list[tuple[float, float, tuple]]:
    """The instants in ``(t_start, t_end)`` at which a rate law of *model* switches
    on the clock, which run_diffrax splits the run at (issue #865), as
    ``(lo, hi, counters)``: crossings within one instant of each other
    (``_same_instant``) are one switch from ``lo`` to ``hi``, and *counters* are the
    counter clocks that cross there, as ``(species index, threshold)``.

    A condition on the clock, ``if(time() > 1.37, 5, 0.1)``, is a jump in the
    right-hand side, and the step-size controller stepped across it: several
    times CVODE's error at ``rtol=1e-6``, and at ``rtol=1e-8`` it spent its whole
    step budget there and failed. The crossings are constants of the run, the
    ones CVODE stops on (:func:`bngsim._switch_sensitivity.crossings_with_periodic`).
    These are the crossings themselves, a periodic condition's included, not the
    stops CVODE places a little past one. A condition only a reporting function
    reads changes an output, not the trajectory, which is all run_diffrax returns,
    and is left out: a split there cost a model whose rate laws never switch
    hundreds of steps, and changed its answer.

    A condition a rate law reads that switches more often than a run's budget of
    stops has none placed, and one that cannot be placed, or a failure to place
    them, would leave the run stepping across them: each raises ``ValueError``
    rather than return the trajectory #865 did. So does a run backward in time over a model with a
    switch on the clock, whose times are placed forward.
    """
    conditions = model.time_discontinuity_conditions()
    if conditions:
        conditions = _conditions_the_run_switches_on(model, conditions)
    if not conditions:
        return []
    if t_end < t_start:
        raise ValueError(
            "run_diffrax integrates backward in time only a model with no switch on the "
            "clock; this one has one, and its switch times are placed forward. Use "
            "Simulator(method='ode') or run forward."
        )
    from bngsim._switch_sensitivity import (
        _PLAIN_INSTANT_FLOOR,
        _rewrite_counter_clock,
        _same_instant,
        crossings_with_periodic,
        switch_condition_scope,
    )

    # A condition on two counters, or on a counter that is not a clock, is one
    # no resolver can place, and the resolvers pass it over without a word: the
    # run then stepped across it, which is the issue again.
    scope = switch_condition_scope(model._core)
    unplaced = [
        c for c in conditions if _rewrite_counter_clock(model._core, c, scope, t_start) is None
    ]
    if unplaced:
        raise ValueError(
            f"run_diffrax cannot place the times these conditions switch at: {unplaced}; "
            "use Simulator(method='ode')."
        )
    over_budget: list = []
    try:
        crossings, _stops, periodic_at = crossings_with_periodic(
            model._core,
            t_start,
            t_end,
            conditions,
            touches=getattr(model, "_event_disc_conditions", frozenset()),
            over_budget=over_budget,
        )
    except Exception as e:
        raise ValueError(
            f"run_diffrax could not place the times this model's rate laws switch at ({e}); "
            "use Simulator(method='ode')."
        ) from e
    if over_budget:
        raise ValueError(
            "run_diffrax cannot stop at every switch of this model: "
            f"{len(over_budget)} condition(s) switch more often than a run may stop, or "
            "too irregularly to count; use Simulator(method='ode')."
        )
    found = sorted(
        (float(c.time), c) for c in (*crossings, *periodic_at) if t_start < c.time < t_end
    )
    groups: list[list] = []
    for t, c in found:
        if groups and _same_instant(groups[-1][1], t, _PLAIN_INSTANT_FLOOR):
            groups[-1][1] = t
        else:
            groups.append([t, t, set()])
        if c.clock_species_idx >= 0:
            groups[-1][2].add((int(c.clock_species_idx), float(c.threshold)))
    return [(lo, hi, tuple(sorted(counters))) for lo, hi, counters in groups]


def _conditions_the_run_switches_on(model, conditions) -> list:
    """Those of *conditions* a rate law reads, or a loader registered as the
    run's own, as the ODE path counts them (``Simulator._switches_the_run``)."""
    from bngsim._switch_sensitivity import conditions_a_rate_law_reads

    registered = set(getattr(model, "_time_disc_conditions", None) or ())
    asked = [c for c in conditions if c not in registered]
    try:
        read = conditions_a_rate_law_reads(model._core, asked)
    except Exception as e:  # pragma: no cover - defensive: then all of them count
        logger.debug("which conditions a rate law reads: not read (%s)", e)
        read = frozenset(asked)
    return [c for c in conditions if c in registered or c in read]
