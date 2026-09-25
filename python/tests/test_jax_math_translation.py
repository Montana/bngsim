"""GH #564 — the jnp rewrite must not rewrite its own output.

``_translate_expr_jax`` mapped the engine's math functions with one ``re.sub``
per function, in sequence, so each rule saw what the rules before it had
emitted. ``ln`` became ``jnp.log``, and the very next rule's ``\\blog\\b``
matched the ``log`` inside that — ``.`` is not a word character, so the
boundary holds there — leaving ``jnp.jnp.log``. Every model with an ``ln()`` in
a function block then died under ``jacobian="jax"`` with "AttributeError:
module 'jax.numpy' has no attribute 'jnp'", while the same model integrated
fine on the default Jacobian.

The rewrite is now a single pass over the source, which cannot re-read its own
replacements. That ends the class of collision rather than the one instance:
any future pair of names where one's replacement contains the other is safe by
construction, not by ordering.

The translator is pure text and needs no JAX, so those tests run everywhere;
the end-to-end pair is skipped when JAX is absent.
"""

from __future__ import annotations

import contextlib
import io
import re

import numpy as np
import pytest
from bngsim._jax_rhs import _translate_expr_jax

try:  # JAX is an optional extra
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    JAX_AVAILABLE = False

PARAMS = {"k1": 0}
OBS = {"B": 0}


def _t(expr: str) -> str:
    return _translate_expr_jax(expr, PARAMS, OBS, set(), [])


# ── The rewrite ──────────────────────────────────────────────────────────────


def test_the_issue_expression():
    assert _t("k1*ln(1+B)") == "params[0]*jnp.log(1.0+obs[0])"


@pytest.mark.parametrize(
    "expr, want",
    [
        ("ln(B)", "jnp.log(obs[0])"),
        ("log(B)", "jnp.log(obs[0])"),
        # the collision needs only one of them, but both spellings in one
        # expression is the case that pins the pass down
        ("ln(B)+log(B)", "jnp.log(obs[0])+jnp.log(obs[0])"),
        ("ln(log(B))", "jnp.log(jnp.log(obs[0]))"),
    ],
)
def test_the_logarithms(expr, want):
    assert _t(expr) == want


@pytest.mark.parametrize(
    "expr, want",
    [
        ("sqrt(B)", "jnp.sqrt(obs[0])"),
        ("exp(B)", "jnp.exp(obs[0])"),
        ("sin(B)", "jnp.sin(obs[0])"),
        ("cos(B)", "jnp.cos(obs[0])"),
        ("tan(B)", "jnp.tan(obs[0])"),
        # the inverse trio must not be read as the bare one plus a stray 'a'
        ("asin(B)", "jnp.arcsin(obs[0])"),
        ("acos(B)", "jnp.arccos(obs[0])"),
        ("atan(B)", "jnp.arctan(obs[0])"),
        ("abs(k1)", "jnp.abs(params[0])"),
        ("min(B,k1)", "jnp.minimum(obs[0],params[0])"),
        ("max(B,k1)", "jnp.maximum(obs[0],params[0])"),
        ("pow(B,2)", "jnp.power(obs[0],2.0)"),
        ("rint(B)", "__bngsim_rint__(obs[0])"),
        ("floor(B)", "jnp.floor(obs[0])"),
        ("ceil(B)", "jnp.ceil(obs[0])"),
        # nesting, and an inverse wrapping its own bare form
        ("atan(tan(B))", "jnp.arctan(jnp.tan(obs[0]))"),
        ("sqrt(exp(B))", "jnp.sqrt(jnp.exp(obs[0]))"),
    ],
)
def test_every_other_function(expr, want):
    assert _t(expr) == want


def test_no_replacement_is_rewritten_by_another_rule():
    """The property, asked of the output rather than of one function: every
    ``jnp.`` is followed by a real jax.numpy name, whatever the expression
    mixes. ``jnp.jnp.log`` is what this file exists to keep out."""
    out = _t("ln(B)+log(B)+sqrt(min(exp(B),max(abs(k1),pow(B,2))))+atan(sin(B))")
    emitted = re.findall(r"jnp\.(\w+)", out)
    assert emitted, "nothing was translated, so the assertion below proves nothing"
    assert set(emitted) == {
        "log",
        "sqrt",
        "minimum",
        "exp",
        "maximum",
        "abs",
        "power",
        "arctan",
        "sin",
    }


def test_a_longer_name_is_not_partially_matched():
    """A name that merely starts with a mapped one keeps its own identity:
    `log10` is translated as itself (GH #565 added it), and a name that is in
    no table at all is refused by its own name rather than half-rewritten into
    `jnp.log` plus a stray tail."""
    assert _t("log10(B)") == "jnp.log10(obs[0])"
    with pytest.raises(ValueError, match="logistic"):
        _t("logistic(B)")


def test_a_model_name_that_merely_contains_one_is_left_alone():
    out = _translate_expr_jax("explicit*k1", {"k1": 0}, {"explicit": 0}, set(), [])
    assert out == "obs[0]*params[0]"


# ── End to end ───────────────────────────────────────────────────────────────

NET = """begin parameters
    1 k1      0.5  # Constant
end parameters
begin functions
    1 law() k1*{body}
end functions
begin species
    1 B() 10
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 B                    1
end groups
"""


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("body", ["ln(1+B)", "log(1+B)", "ln(1+B)+sqrt(B)"])
def test_the_jax_jacobian_matches_the_default(tmp_path, body):
    """The failure the issue reports: this raised "module 'jax.numpy' has no
    attribute 'jnp'" while the same model ran on the default Jacobian."""
    import bngsim

    net = tmp_path / "m.net"
    net.write_text(NET.format(body=body))

    def _run(**kw):
        with contextlib.redirect_stderr(io.StringIO()):
            model = bngsim.Model.from_net(str(net))
            return bngsim.Simulator(model, method="ode", **kw).run(t_span=(0.0, 5.0), n_points=6)

    default = np.asarray(_run().species)[:, 0]
    with_jax = np.asarray(_run(jacobian="jax").species)[:, 0]
    assert with_jax == pytest.approx(default, rel=1e-6, abs=1e-8)
    # ...and the model actually moved, so the agreement is not two flat lines.
    assert default[-1] < default[0]
