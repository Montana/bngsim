"""bngsim._jax_rhs — JAX-based ODE RHS and AD Jacobian for BNGsim.

Generates a JAX-traced RHS function from a .net file, then uses
``jax.jacfwd`` to compute exact dense Jacobians via automatic
differentiation. This provides exact Jacobians for ALL rate law types
(Elementary, Functional, MichaelisMenten) without manual derivatives.

Architecture:
  1. generate_jax_rhs(net_path) -> Callable: Parse .net, build JAX RHS
  2. generate_jax_jacobian(net_path) -> Callable: jacfwd(rhs) wrapper
  3. screen_for_discontinuities(net_path) -> bool: Check for floor/ceil/etc.

The JAX Jacobian is fed back to CVODE via the existing user-Jacobian
callback mechanism (dense matrix). JAX runs on CPU only.

Optional dependency: ``pip install bngsim[jax]``.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any

from bngsim._codegen import _BUILTIN_CONSTANT_VALUES, _classify_rate_law, _parse_net_file

logger = logging.getLogger("bngsim")

# ─── Availability check ─────────────────────────────────────────────────────

_JAX_AVAILABLE: bool | None = None


def jax_available() -> bool:
    """Check if JAX is importable (cached).

    Also enables 64-bit precision (required for CVODE compatibility).
    """
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is None:
        try:
            import jax

            # Enable 64-bit precision — CVODE uses double, and float32
            # Jacobians would corrupt Newton convergence.
            jax.config.update("jax_enable_x64", True)
            import jax.numpy  # noqa: F401

            _JAX_AVAILABLE = True
        except ImportError:
            _JAX_AVAILABLE = False
    return _JAX_AVAILABLE


# ─── Discontinuity screening ────────────────────────────────────────────────

# Functions that produce useless AD gradients (piecewise constant).
_DISCONTINUOUS_FUNCS = {"floor", "ceil", "rint", "round", "Heaviside"}


def screen_for_discontinuities(net_path: str) -> list[str]:
    """Scan function expressions for constructs that defeat AD.

    Returns a list of problematic function names found, or empty list
    if the model is safe for JAX AD.
    """
    model = _parse_net_file(net_path)
    problems = []
    for _, name, expr in model["functions"]:
        for disc_fn in _DISCONTINUOUS_FUNCS:
            if re.search(rf"\b{disc_fn}\b", expr):
                problems.append(f"function '{name}' uses {disc_fn}()")
    return problems


# ─── Expression translator (.net expression -> JAX/Python) ──────────────────

# Stand-in for `time()` while the model-name substitutions run, so no parameter
# or observable can rewrite the clock (issue #659). Not a valid BNG identifier,
# so it cannot collide with a model name.
_CLOCK_SYM = "__bngsim_clock__"

# A zero-arg call — an observable, parameter or any other scalar written as
# `name()` (issue #28). `time()` is the only zero-argument built-in and is
# rewritten before this runs.
_EMPTY_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*\)")

# The engine's math functions and their jax.numpy spelling. Applied in a single
# pass (see _translate_expr_jax), so no entry's replacement can be rewritten by
# another entry's pattern.
_JAX_MATH_FUNCS: dict[str, str] = {
    "ln": "jnp.log",
    "log": "jnp.log",
    "sqrt": "jnp.sqrt",
    "exp": "jnp.exp",
    "sin": "jnp.sin",
    "cos": "jnp.cos",
    "tan": "jnp.tan",
    "asin": "jnp.arcsin",
    "acos": "jnp.arccos",
    "atan": "jnp.arctan",
    "abs": "jnp.abs",
    "min": "jnp.minimum",
    "max": "jnp.maximum",
    "pow": "jnp.power",
    # Not jnp.round, which rounds a half to EVEN — see _jax_round below.
    "rint": "__bngsim_rint__",
    "round": "__bngsim_round__",
    "trunc": "jnp.trunc",
    "floor": "jnp.floor",
    "ceil": "jnp.ceil",
    # GH #565 — the engine's reserved list (reserved_names(), src/expression.cpp)
    # carries these too, and jax.numpy spells every one of them directly. Their
    # absence was not a decision: `log10` never matched the `log` rule (`1` is a
    # word character, so the boundary held), so it reached the generated source
    # untranslated and raised NameError inside the RHS at solve time, on a model
    # the ODE backend and jacobian="auto" both handle.
    "log10": "jnp.log10",
    "log2": "jnp.log2",
    "sinh": "jnp.sinh",
    "cosh": "jnp.cosh",
    "tanh": "jnp.tanh",
    "asinh": "jnp.arcsinh",
    "acosh": "jnp.arccosh",
    "atanh": "jnp.arctanh",
    "sign": "jnp.sign",
    "sgn": "jnp.sign",
}

# The engine's reserved constants, bound on every expression it compiles. Same
# story as the functions above: nothing substitutes them, so `_pi` reached the
# RHS as a bare name and raised NameError there. The physical constants have no
# jax.numpy name, so they are emitted as the values the engine binds.
_JAX_CONSTANTS: dict[str, str] = {
    **{name: repr(value) for name, value in _BUILTIN_CONSTANT_VALUES.items()},
    "_pi": "jnp.pi",
    "_e": "jnp.e",
    # The bare spelling of the clock. `time()` is parked above as _CLOCK_SYM;
    # the engine accepts `time` without parentheses too, and _BUILTIN_IDENT_MAP
    # (the C path) maps it the same way. (`t()` is not the clock — issue #659.)
    "time": "t",
}

_JAX_NAMES: dict[str, str] = {**_JAX_MATH_FUNCS, **_JAX_CONSTANTS}


# The engine's two roundings, neither of which jax.numpy spells. jnp.round
# rounds a half to EVEN, so mapping either name onto it moved every exact half:
# round(2.5) came out 2 here and 3 in the engine. They are not quite each other
# either. `rint` is the engine's own adapter over C std::round
# (src/expression.cpp), halves away from zero; `round` is ExprTk's
# floor(x + 0.5), or ceil(x - 0.5) below zero, which agrees except where the
# addition itself rounds — round(0.49999999999999994) is 1, rint of it is 0.
# Each needs its argument twice, so the RHS namespace binds them as helpers.
def _jax_round(x: Any) -> Any:
    """ExprTk's ``round``: ``floor(x + 0.5)``, or ``ceil(x - 0.5)`` below zero."""
    import jax.numpy as jnp

    return jnp.where(x < 0, jnp.ceil(x - 0.5), jnp.floor(x + 0.5))


def _jax_rint(x: Any) -> Any:
    """The engine's ``rint``: C ``std::round``, a half rounded away from zero."""
    import jax.numpy as jnp

    whole = jnp.trunc(x)  # x - whole is exact, so the test below is too
    return jnp.where(jnp.abs(x - whole) >= 0.5, whole + jnp.sign(x), whole)


_JAX_HELPERS: dict[str, Any] = {"__bngsim_round__": _jax_round, "__bngsim_rint__": _jax_rint}

# What the generated RHS binds around the eval (see generate_jax_rhs), plus the
# Python keywords an expression may legitimately contain. Every other bare name
# left after translation is a NameError waiting for solve time.
_JAX_EVAL_NAMES = frozenset({"jnp", "t", "params", "obs", "y", *_JAX_HELPERS})
_PY_KEYWORDS = frozenset({"and", "or", "not", "if", "else", "True", "False", "None"})

# An identifier that is not an attribute access: `jnp.log` is one name, not two.
_BARE_IDENT_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)")

# Quoted text is data, not a name — a table function's file name would
# otherwise be read as a pile of undefined identifiers.
_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")

# Longest name first so `asin` wins over `sin`, and no match may start straight
# after a word character or a `.` — the latter keeps an already-emitted
# `jnp.log` from being read as the function `log` should this ever run twice.
_JAX_MATH_RE = re.compile(
    r"(?<![\w.])(" + "|".join(sorted(map(re.escape, _JAX_NAMES), key=len, reverse=True)) + r")\b"
)


def _translate_expr_jax(
    expr: str,
    param_names: dict[str, int],
    obs_names: dict[str, int],
    func_names_set: set[str],
    func_order: list[str],
) -> str:
    """Translate a .net function expression to JAX-compatible Python.

    Replaces:
      - parameter names -> params[idx]
      - observable names -> obs[idx]
      - function names -> func_<name> (local variable)
      - a scalar written as a zero-arg call (`Atot()`) -> the scalar (issue #28)
      - time() -> t
      - if(cond,a,b) -> jnp.where(cond,a,b)
      - ln() -> jnp.log()
      - common math -> jnp.<func>()
      - && -> & (JAX boolean), || -> | (JAX boolean)
    """
    c = expr

    # Replace logical operators FIRST
    c = c.replace("&&", " & ")
    c = c.replace("||", " | ")

    # The clock, parked under a placeholder no model name can collide with.
    #
    # Issue #659 — `time` is the only clock symbol the evaluator binds; `t` is
    # deliberately left free as an ordinary model identifier
    # (src/expression.cpp), and `t()` is that identifier written as a zero-arg
    # call. Rewriting both to a bare `t` here put the clock and a model symbol
    # named `t` into the same token, and the observable pass below then rewrote
    # it: in a model with an observable `t`, `time()` came out as `obs[i]` — the
    # clock silently replaced by a population. The placeholder survives every
    # substitution below and is spent last.
    c = re.sub(r"\btime\s*\(\s*\)", _CLOCK_SYM, c)

    # A scalar written as a zero-arg call (issue #28): BNGL accepts `Atot()`
    # wherever `Atot` is valid and BNG2.pl preserves whichever the user wrote,
    # so the engine strips the parens for any registered scalar
    # (strip_empty_parens, src/expression.cpp). Do the same before the name
    # passes below, or `Atot()` becomes `obs[1]()` — a call on a JAX array —
    # and `t()` becomes `obs[3]()`. `time()` is already gone, and it is the only
    # zero-arg built-in, so every remaining empty argument list is a scalar.
    c = _EMPTY_CALL_RE.sub(r"\1", c)

    # Replace if(cond, a, b) -> jnp.where(cond, a, b)
    # This handles nested if() via repeated application
    for _ in range(10):  # max nesting depth
        new_c = re.sub(r"\bif\s*\(", "jnp.where(", c)
        if new_c == c:
            break
        c = new_c

    # Replace function references: funcName() or bare funcName
    # Must do BEFORE parameter replacement
    for fname in func_order:
        safe = _safe_py_name(fname)
        c = re.sub(rf"\b{re.escape(fname)}\(\)", f"func_{safe}", c)
        c = re.sub(rf"\b{re.escape(fname)}\b", f"func_{safe}", c)

    # Replace observable names (longest first to avoid partial match)
    for name in sorted(obs_names.keys(), key=len, reverse=True):
        idx = obs_names[name]
        c = re.sub(
            rf"(?<!func_)\b{re.escape(name)}\b",
            f"obs[{idx}]",
            c,
        )

    # Replace parameter names (longest first)
    for name in sorted(param_names.keys(), key=len, reverse=True):
        idx = param_names[name]
        c = re.sub(
            rf"(?<!obs\[)(?<!func_)\b{re.escape(name)}\b",
            f"params[{idx}]",
            c,
        )

    # Replace math functions with jnp equivalents, in ONE pass (GH #564).
    # Run as a sequence of re.sub calls, each rule saw what the rules before it
    # had already emitted: `ln` became `jnp.log`, and the very next rule's
    # `\blog\b` matched the `log` inside it — `.` is not a word character, so
    # the boundary holds there — leaving `jnp.jnp.log` and an
    # "AttributeError: module 'jax.numpy' has no attribute 'jnp'" at
    # evaluation. One pass over the source cannot rewrite its own output, which
    # ends the whole class of collision rather than the one instance of it.
    c = _JAX_MATH_RE.sub(lambda m: _JAX_NAMES[m.group(1)], c)

    # Replace ^ with ** for exponentiation
    c = c.replace("^", "**")

    # Spend the clock placeholder last: `t` is the JAX RHS's own time argument,
    # and nothing above may rewrite it (issue #659).
    c = c.replace(_CLOCK_SYM, "t")

    # After the clock is spent, not before: the scan below rejects any bare name
    # it cannot evaluate, and `__bngsim_clock__` is one until it becomes `t`.
    _reject_untranslated_names(expr, c)

    return c


def _reject_untranslated_names(expr: str, translated: str) -> None:
    """Raise if anything survived translation that the RHS cannot evaluate.

    The generated RHS evaluates each translated body with ``{"__builtins__":
    {}}`` and a namespace holding only ``jnp``/``t``/``params``/``obs``/``y``
    and the previously computed ``func_*`` locals, so a name that reaches it
    untranslated is a guaranteed ``NameError`` — raised deep inside the solve,
    naming a symbol the caller never wrote in Python (GH #565). Saying so here,
    while the expression is still in hand, costs nothing and names the function
    and the model text it came from.
    """
    scanned = _STRING_LITERAL_RE.sub("''", translated)
    unknown = sorted(
        {
            name
            for name in _BARE_IDENT_RE.findall(scanned)
            if name not in _JAX_EVAL_NAMES
            and name not in _PY_KEYWORDS
            and not name.startswith("func_")
        }
    )
    if unknown:
        raise ValueError(
            f"jacobian='jax' cannot translate {', '.join(repr(u) for u in unknown)} "
            f"in the expression {expr!r}: no jax.numpy equivalent is mapped for it. "
            "Use jacobian='auto' (the default), which evaluates this model through "
            "the engine instead."
        )


def _safe_py_name(name: str) -> str:
    """Convert a BNG name to a safe Python identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# ─── JAX RHS generator ──────────────────────────────────────────────────────


def generate_jax_rhs(net_path: str) -> Any:
    """Generate a JAX-traced RHS function from a .net file.

    The returned function has signature::

        rhs(y: jnp.ndarray, t: float, params: jnp.ndarray) -> jnp.ndarray

    where y is species (n_species,), params is (n_params,), and the
    return is dydt (n_species,).

    All operations use jnp so the function is JAX-traceable for AD.

    Parameters
    ----------
    net_path : str
        Path to the .net file.

    Returns
    -------
    Callable
        JAX-traceable RHS function.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    import jax.numpy as jnp

    model = _parse_net_file(net_path)
    params_list = model["parameters"]
    species_list = model["species"]
    reactions = model["reactions"]
    observables = model["observables"]
    functions = model["functions"]

    n_sp = len(species_list)
    n_params = len(params_list)

    # Build index maps (0-based)
    param_idx = {name: i for i, (_, name, _, _) in enumerate(params_list)}
    func_names_set = {name for _, name, _ in functions}
    func_order = [name for _, name, _ in functions]
    {name: i for i, (_, name, _) in enumerate(functions)}
    obs_idx = {name: i for i, (_, name, _) in enumerate(observables)}

    # Fixed species (0-based indices)
    fixed_sp = frozenset(sp[0] - 1 for sp in species_list if sp[3])

    # Pre-build observable weight matrix as dense array
    # obs[k] = sum of weight[k, j] * y[j]
    obs_weights = []
    for _, _name, entries in observables:
        row = [0.0] * n_sp
        for factor, sp_i in entries:
            row[sp_i - 1] = factor
        obs_weights.append(row)

    # Pre-translate function expressions to JAX Python
    func_exprs = []
    for _, name, expr in functions:
        jax_expr = _translate_expr_jax(expr, param_idx, obs_idx, func_names_set, func_order)
        func_exprs.append((name, jax_expr))

    # Pre-classify reactions and build stoichiometry
    rxn_data = []
    for _, reactants, products, rate_law, _comment in reactions:
        kind = _classify_rate_law(rate_law, func_names_set)
        rxn_data.append((reactants, products, kind))

    # Build the RHS function source as a closure
    # We create numpy arrays for the weight matrix at build time
    obs_w_array = jnp.array(obs_weights, dtype=jnp.float64)

    def rhs(y, t, params):
        """JAX-traced ODE RHS: dy/dt = f(y, t, params)."""
        # Compute observables: obs = W @ y
        obs = obs_w_array @ y

        # Evaluate functions in dependency order
        func_vals = {}
        for fname, jax_expr in func_exprs:
            _safe_py_name(fname)
            # Build local namespace for eval
            local_ns = {
                **_JAX_HELPERS,
                "jnp": jnp,
                "t": t,
                "params": params,
                "obs": obs,
                "y": y,
            }
            # Add previously computed functions
            for prev_name, prev_val in func_vals.items():
                local_ns[f"func_{_safe_py_name(prev_name)}"] = prev_val
            val = eval(jax_expr, {"__builtins__": {}}, local_ns)  # noqa: S307
            func_vals[fname] = val

        # Compute derivatives
        dydt = jnp.zeros(n_sp, dtype=y.dtype)

        for reactants, products, kind in rxn_data:
            if kind[0] == "elementary":
                _, pname, sf = kind
                p_i = param_idx.get(pname, -1)
                rate = params[p_i] * sf if p_i >= 0 else 0.0
                for ri in reactants:
                    rate = rate * y[ri - 1]
            elif kind[0] == "functional":
                _, fname, sf = kind
                rate = func_vals[fname] * sf
                for ri in reactants:
                    rate = rate * y[ri - 1]
            elif kind[0] == "mm":
                _, kcat_name, km_name, sf = kind
                kcat_i = param_idx.get(kcat_name, -1)
                km_i = param_idx.get(km_name, -1)
                kcat = params[kcat_i] if kcat_i >= 0 else 0.0
                km = params[km_i] if km_i >= 0 else 0.0
                if len(reactants) >= 2:
                    e = y[reactants[0] - 1]
                    s = y[reactants[1] - 1]
                    # Stable positive root of x² − delta·x − km·s = 0 (GH #89):
                    # ½(delta + D) cancels for delta < 0, so that branch uses the
                    # conjugate form. The denominator is masked before the divide
                    # so the unselected branch cannot inject a NaN into a tangent.
                    delta = s - km - e
                    d_mm = jnp.sqrt(delta * delta + 4.0 * km * s)
                    neg = delta < 0.0
                    denom = jnp.where(neg, d_mm - delta, 1.0)
                    s_free = jnp.where(neg, 2.0 * km * s / denom, 0.5 * (delta + d_mm))
                    # GH #93: no clamp on s_free (it is negative exactly when s
                    # is, and the rate continues smoothly there), but the rate's
                    # denominator is guarded — it vanishes when km*e == 0. Masked
                    # before the divide for the same reason as above: an unmasked
                    # 0/0 in the unselected branch NaNs the tangent.
                    kps = km + s_free
                    live = kps > 0.0
                    rate = jnp.where(live, sf * kcat * s_free * e / jnp.where(live, kps, 1.0), 0.0)
                else:
                    rate = 0.0
            else:
                rate = 0.0

            # Accumulate stoichiometry
            for ri in reactants:
                if ri > 0:
                    dydt = dydt.at[ri - 1].add(-rate)
            for pi in products:
                if pi > 0:
                    dydt = dydt.at[pi - 1].add(rate)

        # Zero fixed species derivatives
        for si in fixed_sp:
            dydt = dydt.at[si].set(0.0)

        return dydt

    # Attach metadata (function attributes — Any-typed to satisfy mypy)
    rhs_obj: Any = rhs
    rhs_obj.n_species = n_sp
    rhs_obj.n_params = n_params
    rhs_obj.fixed_species = fixed_sp

    return rhs_obj


# ─── JAX Jacobian wrapper ───────────────────────────────────────────────────


def generate_jax_jacobian(net_path: str) -> Any:
    """Generate a JAX AD Jacobian function from a .net file.

    Returns a function::

        jac_fn(y: ndarray, t: float, params: ndarray) -> ndarray

    that computes the exact N×N dense Jacobian ∂f/∂y via forward-mode AD.

    Parameters
    ----------
    net_path : str
        Path to the .net file.

    Returns
    -------
    Callable
        Function that returns (n_species, n_species) Jacobian matrix.

    Raises
    ------
    ImportError
        If JAX is not installed.
    RuntimeError
        If model contains discontinuous functions (floor/ceil/etc.).
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    # Screen for discontinuities — warn but don't reject.
    # CVODE uses the Jacobian for Newton convergence, not solution
    # correctness. A locally-zero Jacobian at a discontinuity just
    # means CVODE takes smaller steps there (same as FD behavior).
    problems = screen_for_discontinuities(net_path)
    if problems:
        logger.warning(
            "Model functions may produce zero JAX AD gradients at "
            "discontinuities (CVODE will adapt step size):\n  %s",
            "\n  ".join(problems),
        )

    import jax

    rhs = generate_jax_rhs(net_path)

    # Forward-mode AD: differentiate RHS w.r.t. y (argnums=0)
    # Returns (n_species, n_species) Jacobian matrix
    jac_fn_raw = jax.jacfwd(rhs, argnums=0)

    def jac_fn(y, t, params):
        """Compute dense Jacobian J[i][j] = ∂f_i/∂y_j."""
        return jac_fn_raw(y, t, params)

    # Attach metadata (function attributes — Any-typed to satisfy mypy)
    jac_fn_obj: Any = jac_fn
    jac_fn_obj.n_species = rhs.n_species
    jac_fn_obj.n_params = rhs.n_params
    jac_fn_obj.rhs = rhs

    return jac_fn_obj


# ─── Prepare JAX Jacobian for CVODE ─────────────────────────────────────────


def prepare_jax_jacobian(net_path: str) -> tuple:
    """Prepare a JAX Jacobian evaluator for use with CVODE.

    Returns a tuple (jac_fn, n_species, param_values) where jac_fn
    is a callable that takes (y_flat, t, param_flat) and returns
    a flat row-major Jacobian array suitable for C++ consumption.

    Parameters
    ----------
    net_path : str
        Path to the .net file.

    Returns
    -------
    tuple
        (evaluate_jacobian, n_species) where evaluate_jacobian is
        a callable (y_flat, t, param_flat) -> flat_jac_array.
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    import jax
    import jax.numpy as jnp
    import numpy as np

    jac_fn = generate_jax_jacobian(net_path)
    n_sp = jac_fn.n_species

    # JIT-compile for speed
    jac_fn_jit = jax.jit(jac_fn)

    # Warm up with dummy data to trigger compilation
    dummy_y = jnp.ones(n_sp, dtype=jnp.float64)
    dummy_p = jnp.ones(jac_fn.n_params, dtype=jnp.float64)
    # warmup may fail with dummy params; that's OK
    with contextlib.suppress(Exception):
        _ = jac_fn_jit(dummy_y, 0.0, dummy_p)

    def evaluate_jacobian(y_flat, t, param_flat):
        """Evaluate Jacobian, return column-major flat array for CVODE.

        CVODE dense matrix is column-major (Fortran order).
        """
        y_jax = jnp.array(y_flat, dtype=jnp.float64)
        p_jax = jnp.array(param_flat, dtype=jnp.float64)
        J = jac_fn_jit(y_jax, t, p_jax)
        # J is (n_sp, n_sp) with J[i][j] = df_i/dy_j
        # CVODE dense matrix is column-major: column j, row i
        # np.asfortranarray gives column-major layout
        J_np = np.asarray(J, dtype=np.float64)
        return J_np.flatten(order="F")

    return evaluate_jacobian, n_sp
