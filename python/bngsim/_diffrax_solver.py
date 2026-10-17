"""bngsim._diffrax_solver — Diffrax ODE solver (pure JAX, no CVODE).

Runs the entire ODE solve in JAX/XLA — RHS, Jacobian (AD), and the
adaptive stepper are all JIT-compiled into a single fused computation.
No Python↔C++ boundary crossings during the solve.

Uses Kvaerno5 (5th-order implicit Runge-Kutta, L-stable) for stiff
biochemical systems, with PID step size control.

The RHS, the parameter values and the initial state all come from the built
model (issue #803 step 4). They used to be re-read from the ``.net`` text by
codegen's private parser: a synthesis reaction's rate was multiplied by the last
species, a model without observables crashed, and an initial concentration
written as an expression was looked up as a parameter name.

A rate law that switches on the clock, ``if(time() > t*, ...)``, jumps there,
and the run is split at each such time and restarted just past it (issue #865):
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
        Maximum internal solver steps.

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

    model = _as_model(model)
    if param_dict:
        model = model.clone()
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
    jump_ts = _switch_times(model, float(t_start), float(t_end))
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

    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # Issue #865. The run is split at the times its rate laws switch at
    # (_switch_times): each piece ends at the float just before a switch, and the
    # next begins at the float just after it, so no step is taken across the
    # jump. diffrax also reads the vector field at trial times past the end of
    # the interval it solves over (to 1.43 on [0, 1.37), at rtol=1e-6), and a
    # switch just past the end turned every step near it down until the steps
    # ran out, both for a piece ending there and for diffrax's own jump_ts. So
    # within a piece the right-hand side is read at the time clamped to the
    # piece, and only ever sees the branch the piece is on. One compiled solve
    # serves every piece: its bounds, start state and output times are
    # arguments, the output times padded with the piece's end, the last of
    # which is the state the next piece starts from.
    @jax.jit
    def _solve(a, b, y_a, seg_ts, params):
        def piece_rhs(t, y, args):
            return rhs(y, jnp.clip(t, a, b), args)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(piece_rhs),
            solver,
            t0=a,
            t1=b,
            dt0=jnp.minimum((b - a) / 100.0, 0.01),
            y0=y_a,
            args=params,
            saveat=diffrax.SaveAt(ts=seg_ts),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        return sol.ys, sol.stats["num_steps"]

    edges = [float(t_start), *jump_ts, float(t_end)]
    species_out = np.empty((n_points, len(sp_names)), dtype=np.float64)
    y_a = y0
    n_steps = 0
    for i in range(len(edges) - 1):
        a = edges[0] if i == 0 else float(np.nextafter(edges[i], np.inf))
        b = edges[-1] if i == len(edges) - 2 else float(np.nextafter(edges[i + 1], -np.inf))
        # A piece reports the output times in (edges[i], edges[i + 1]], the first
        # also t_start. One on a switch is the piece's before it, read at the float
        # just before: the state does not jump there, only its derivative.
        mine = np.flatnonzero((out_ts <= edges[i + 1]) & ((out_ts > edges[i]) | (i == 0)))
        seg_ts = np.full(n_points + 1, b)
        seg_ts[: mine.size] = np.clip(out_ts[mine], a, b)
        ys, n = _solve(
            jnp.float64(a), jnp.float64(b), y_a, jnp.asarray(seg_ts, dtype=jnp.float64), params
        )
        ys = np.asarray(ys)
        species_out[mine] = ys[: mine.size]
        y_a = jnp.asarray(ys[-1])
        n_steps += int(n)

    return {
        "time": np.asarray(ts),
        "species": species_out,
        "species_names": sp_names,
        "n_steps": n_steps,
    }


def _switch_times(model, t_start: float, t_end: float) -> list[float]:
    """The instants in ``(t_start, t_end)`` at which a rate law of *model* switches
    on the clock, for diffrax's ``jump_ts`` (issue #865).

    A condition on the clock, ``if(time() > 1.37, 5, 0.1)``, is a jump in the
    right-hand side, and the step-size controller stepped across it: several
    times CVODE's error at ``rtol=1e-6``, and at ``rtol=1e-8`` it spent its whole
    step budget there and failed. The crossings are constants of the run, the
    ones CVODE stops on (:func:`bngsim._switch_sensitivity.crossings_with_periodic`).
    These are the crossings themselves, a periodic condition's included, not the
    stops CVODE places a little past one.

    A condition that switches more often than a run's budget of stops has none
    placed, and a failure to place them would leave the run stepping across
    them: both raise ``ValueError`` rather than return the trajectory #865 did.
    """
    conditions = model.time_discontinuity_conditions()
    if not conditions:
        return []
    from bngsim._switch_sensitivity import crossings_with_periodic

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
            f"{len(over_budget)} condition(s) switch more often than a run allows; "
            "use Simulator(method='ode')."
        )
    return sorted({float(c.time) for c in (*crossings, *periodic_at) if t_start < c.time < t_end})
