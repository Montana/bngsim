"""Issue #1126: a sensitivity run on a pulse made of ``min()``, ``max()`` or ``abs()``.

A counter ``0 -> T`` at ``k_clock = 2``, read as ``Tobs``, and ``X -> Y`` at
``kx*(1/w)*max(0,1-abs(Tobs-c)/(w/2))``, a triangle of area 0.5 in T. Y =
10(1 - exp(-kx*0.5/k_clock)) whatever ``c`` and ``w`` are, so dY/dk_clock =
-0.973501, and dY/dc = dY/dw = 0 once the pulse is past.

The bends had no analytic sensitivity RHS (the rate law "uses unsupported
construct: max()"), and on CVODES' difference quotient the clock-rate column of
a triangle of width 0.01 was -5.97 at rtol 1e-6 and -0.91 at 1e-9, and did not
converge. A bend's derivative has no delta in it, so the analytic path needs
nothing at the bend that it does not have, and ``∂f/∂p`` is now taken through
the same kink rewrite the state Jacobian uses (issue #507).
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 k_clock {k_clock!r}
    2 kx {kx!r}
    3 w {w!r}
    4 c {c!r}
end parameters
begin functions
    1 pulse() {law}
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 pulse
end reactions
begin groups
    1 Tobs 1
end groups
"""

LAWS = {
    "max-abs": "kx*(1/w)*max(0,1-abs(Tobs-c)/(w/2))",
    "min-max": "kx*(1/w)*max(0,min((Tobs-c+w/2)/(w/2),(c+w/2-Tobs)/(w/2)))",
}
PARAMS = ["kx", "k_clock", "c", "w"]


def _model(tmp_path, law: str, **values) -> bngsim.Model:
    v = {"k_clock": 2.0, "kx": 1.0, "w": 0.1, "c": 50.0, **values}
    path = tmp_path / f"pulse_{abs(hash(tuple(sorted(v.items()))))}.net"
    path.write_text(NET.format(law=law, **v))
    return bngsim.Model.from_net(str(path))


def _run(model, sens=None, rtol=1e-8):
    return bngsim.Simulator(model, method="ode", sensitivity_params=sens).run(
        t_span=(0.0, 30.0), n_points=4, rtol=rtol, atol=rtol / 100
    )


IF_SPELLING = (
    "kx*(1/w)*if(Tobs<c-w/2,0,if(Tobs<c,(Tobs-c+w/2)/(w/2),if(Tobs<c+w/2,(c+w/2-Tobs)/(w/2),0)))"
)


def _clock_column(tmp_path, law: str, w: float, rtol: float) -> float:
    run = _run(_model(tmp_path, law, w=w), ["kx", "k_clock"], rtol=rtol)
    return float(np.asarray(run.sensitivities)[-1, 2, 1])


@pytest.mark.parametrize("rtol", [1e-6, 1e-8])
@pytest.mark.parametrize("w", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("name", list(LAWS))
def test_the_clock_rate_column_is_the_if_spellings(tmp_path, name, w, rtol):
    """The same triangle written with if(), which was already on the analytic
    path. At w = 0.01 and rtol 1e-6 the bends gave -5.97 beside its -1.04."""
    bend = _clock_column(tmp_path, LAWS[name], w, rtol)
    assert bend == pytest.approx(_clock_column(tmp_path, IF_SPELLING, w, rtol), rel=2e-3)


@pytest.mark.parametrize("w", [0.1, 1.0])
@pytest.mark.parametrize("name", list(LAWS))
def test_the_clock_rate_column_converges(tmp_path, name, w):
    exact = -10 * (0.5 / 4) * math.exp(-0.5 / 2)
    assert _clock_column(tmp_path, LAWS[name], w, 1e-8) == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize("name", list(LAWS))
def test_the_sensitivity_rhs_is_analytic(tmp_path, name):
    model = _model(tmp_path, LAWS[name])
    _run(model, ["kx", "c"])
    assert model._codegen_sens_decline is None


@pytest.mark.parametrize(
    "law",
    ["if(max(Tobs-c,0)==0,kx,0)", "if(abs(Tobs-c)<w/2,kx,0)", "kx*max(0,Tobs-c)*(Tobs>c+1)"],
)
def test_a_bend_inside_a_condition_is_still_declined(tmp_path, law):
    """A bend in a condition makes crossings of its own, which nothing reads:
    ``max(T − c, 0) == 0`` holds over an interval, and on the analytic path a
    model of that shape gave dY/dk = 0 for 2.46 with no error."""
    sim = bngsim.Simulator(_model(tmp_path, law), method="ode", sensitivity_params=["kx"])
    assert not sim.has_analytic_sens_rhs


@pytest.mark.parametrize(
    "law",
    [
        LAWS["max-abs"],
        "kx*0.001*max(0,Tobs-c)",
        "kx*0.01*abs(Tobs-c)",
        "kx*0.01*min(Tobs,c)",
        "kx*(1/w)*max(0,1-abs(time()-c/2)/(w/2))",
    ],
)
def test_every_column_is_a_difference_of_runs(tmp_path, law):
    """Parameters inside the bends too, on a counter and on the clock."""
    base = {"k_clock": 2.0, "kx": 1.0, "w": 1.0, "c": 50.0}
    run = _run(_model(tmp_path, law, **base), PARAMS, rtol=1e-10)
    sens = np.asarray(run.sensitivities)[:, 2, :]
    for j, name in enumerate(PARAMS):
        h = 1e-5 * base[name]
        y = [
            np.asarray(_run(_model(tmp_path, law, **{**base, name: v}), rtol=1e-10).species)[:, 2]
            for v in (base[name] + h, base[name] - h)
        ]
        np.testing.assert_allclose(sens[:, j], (y[0] - y[1]) / (2 * h), atol=5e-5, err_msg=name)
