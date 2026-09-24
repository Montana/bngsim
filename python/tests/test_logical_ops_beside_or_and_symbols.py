"""`||` / `&&` stay operators in a model that declares `or` / `and` (issue #770).

A BNGL/.net model may name a parameter or observable `or` or `and`; bngsim
registers it under the mangled key `r_or` / `r_and` (issue #18). The ExprTk
compiler used to turn `||` / `&&` into the keywords `or` / `and` *first* and
remap identifiers *second*, so the remap captured the operator it had just
inserted:

* `(x>2) || on` compiled as `(x>2) r_or on`, which ExprTk reads as the implicit
  product `(x>2)*r_or*on`: a silently wrong rate;
* `(a)||(b)`, BNG2.pl's usual spelling, tripped the "declared model symbol used
  as a function call" guard and the model refused to load, on every engine,
  since the compiled (codegen) path loads through the same compiler.

The sympy printer that writes the derived Jacobian back to ExprTk text emitted
the keywords too, so such a model also lost its analytical Jacobian.

The operators are now replaced after the remap, and the printer writes
`&&` / `||`. Every model here is A -> 0 at rate k with k = 1 whenever the
condition holds, so A(1) = 100 e^-1.
"""

from __future__ import annotations

import contextlib
import io
import math
import textwrap
from pathlib import Path

import numpy as np
import pytest
from bngsim import Model, Simulator

A1 = 100.0 * math.exp(-1.0)

#: (extra parameters, rate law). Each declares `and` and/or `or` with a value
#: that would change the answer if it were read in place of the operator.
CASES = {
    "and_as_flag": ({"on": 1, "and": 0}, "if(((k>0.5)&&on),k,0)"),
    "and_scales_product": ({"on": 1, "and": 4}, "k*(on&&1)"),
    "or_bare_operand": ({"on": 1, "or": 0}, "if((k>5)||on,k,0)"),
    "or_parenthesized": ({"or": 0}, "if((k>5)||(k>0.5),k,0)"),
    "both_symbols_and_both_ops": ({"on": 1, "and": 0, "or": 0}, "if((k>5)||((k>0.5)&&on),k,0)"),
}


def _net(tmp_path: Path, extra: dict[str, float], body: str, name: str) -> str:
    params = {"k": 1, **extra}
    plines = "\n".join(f"  {i} {n} {v}" for i, (n, v) in enumerate(params.items(), start=1))
    text = textwrap.dedent(
        """
        begin parameters
        PARAMS
        end parameters
        begin species
          1 A() 100
        end species
        begin functions
          1 f() BODY
        end functions
        begin reactions
          1 1 0 f
        end reactions
        begin groups
          1 Atot 1
        end groups
        """
    ).strip()
    path = tmp_path / f"{name}.net"
    path.write_text(text.replace("PARAMS", plines).replace("BODY", body) + "\n")
    return str(path)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
def test_operator_is_not_the_symbol(tmp_path, case, codegen):
    extra, body = CASES[case]
    model = Model.from_net(_net(tmp_path, extra, body, case))
    sim = Simulator(model, method="ode", codegen=codegen)
    # The codegen leg must really run compiled code, not fall back to ExprTk.
    assert sim.codegen_backend in (("cc", "mir") if codegen else ("exprtk",))
    res = sim.run(t_span=(0, 1), n_points=2)
    assert float(res.species[-1, 0]) == pytest.approx(A1, rel=1e-6)


@pytest.mark.parametrize("case", sorted(CASES))
def test_ssa_matches_the_renamed_control(tmp_path, case):
    """SSA shares the interpreter's compiler. Renaming the symbols cannot change
    a model whose operators are operators, so seed for seed the trajectories
    must be identical."""
    extra, body = CASES[case]
    renamed = {("r_" + n if n in ("and", "or") else n): v for n, v in extra.items()}
    nets = (_net(tmp_path, extra, body, case), _net(tmp_path, renamed, body, case + "_ctl"))
    finals = [
        [
            float(
                Simulator(Model.from_net(net), method="ssa")
                .run(t_span=(0, 1), n_points=2, seed=s)
                .species[-1, 0]
            )
            for s in range(20)
        ]
        for net in nets
    ]
    assert finals[0] == finals[1]


def test_the_symbols_still_read_as_values(tmp_path):
    """The other half: a bare `and` / `or` is still the model's parameter."""
    net = _net(tmp_path, {"and": 0.25, "or": 0.75}, "k*(and+or)", "bare")
    res = Simulator(Model.from_net(net), method="ode").run(t_span=(0, 1), n_points=2)
    assert float(res.species[-1, 0]) == pytest.approx(A1, rel=1e-6)


def test_printer_writes_c_style_logicals():
    """The derived Jacobian's text uses `&&` / `||`, which compile() turns into
    the keywords only after the remap."""
    sp = pytest.importorskip("sympy")
    from bngsim._jacobian import sympy_to_exprtk

    x, y = sp.symbols("x y")
    both = sp.Piecewise((x, sp.And(x > 1, y < 2)), (0, True))
    either = sp.Piecewise((x, sp.Or(x > 1, y < 2)), (0, True))
    assert sympy_to_exprtk(both) == "if(((x > 1) && (y < 2)),x,0)"
    assert sympy_to_exprtk(either) == "if(((x > 1) || (y < 2)),x,0)"


@pytest.mark.parametrize("declare", [False, True], ids=["plain", "with_and_or"])
def test_analytical_jacobian_survives_the_symbols(tmp_path, declare):
    """d/dA of a rate gated by `&&` is a Piecewise with an And condition. With
    `and` / `or` declared, the printed keyword used to be captured, and the
    model silently fell back to the finite-difference Jacobian."""
    pytest.importorskip("sympy")
    extra = {"kb": 0.5, **({"and": 0, "or": 0} if declare else {})}
    params = "\n".join(f"  {i} {n} {v}" for i, (n, v) in enumerate({"k": 1, **extra}.items(), 1))
    net = tmp_path / "jac.net"
    net.write_text(
        f"begin parameters\n{params}\nend parameters\n"
        "begin species\n  1 A() 50\n  2 B() 10\nend species\n"
        "begin functions\n  1 f() if(Atot>1 && Atot<200, k*Atot*Btot/(1+Atot), 0)\nend functions\n"
        "begin reactions\n  1 1 2 f\n  2 2 1 kb\nend reactions\n"
        "begin groups\n  1 Atot 1\n  2 Btot 2\nend groups\n"
    )
    model = Model.from_net(str(net))
    with contextlib.redirect_stderr(io.StringIO()):
        assert model.prepare_analytical_jacobian()
    y = np.array([50.0, 10.0])
    jac = model.jacobian(y)
    assert jac.source == "analytical"
    fd = np.asarray(model._core.fill_dense_fd_jacobian(0.0, y))
    np.testing.assert_allclose(np.asarray(jac), fd, rtol=1e-6, atol=1e-6)
