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
        If the model has events, which this solver does not implement, or a
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

    # Diffrax RHS adapter: f(t, y, args) where args = params
    def diffrax_rhs(t, y, args):
        return rhs(y, t, args)

    # Build solver
    term = diffrax.ODETerm(diffrax_rhs)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # JIT-compile and run
    @jax.jit
    def _solve(y0, params):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_start,
            t1=t_end,
            dt0=min((t_end - t_start) / 100.0, 0.01),
            y0=y0,
            args=params,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        return sol.ys, sol.stats["num_steps"]

    species_out, n_steps = _solve(y0, params)

    return {
        "time": np.asarray(ts),
        "species": np.asarray(species_out),
        "species_names": sp_names,
        "n_steps": int(n_steps),
    }
