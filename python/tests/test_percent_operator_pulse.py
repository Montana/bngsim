"""Issue #1127: a pulse written with ExprTk's ``%`` operator.

``((Tobs+49.99)%100)-((Tobs+50)%100)+0.01`` is the ``mod()`` pulse of #1068
spelled with the operator: height 100 and width 0.01 in T at T = 50. Every
stop and root scanner read named calls only, so on a clock it got no stop and
on a counter no root, and one step spanned it: Y = 0 with no error, where the
``mod()`` spelling was right. The scanners now read ``a % b`` as ``mod(a, b)``,
which is the same function in ExprTk (``std::fmod``).

T = k*t, so the pulse lasts 0.01/k and Y = 10(1 - exp(-kx/k)).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import _spell_mod

NET = """begin parameters
    1 k_clock {k}
    2 kx 1
end parameters
begin functions
    1 pulse() kx*({law})
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

PULSES = {
    "parenthesized": "((Tobs+49.99)%100)-((Tobs+50)%100)+0.01",
    "bare": "(Tobs+49.99)%100-(Tobs+50)%100+0.01",
    # 2*(T + 49.99) % 200 / 2 is mod(T + 49.99, 100): % reads its left operand
    # from the product before it, and the product goes on after it.
    "inside a product": "2*(Tobs+49.99)%200/2-(Tobs+50)%100+0.01",
}
# The same pulse on time(), which runs at 1.
CLOCK = "(time()+49.99)%100-(time()+50)%100+0.01"


def _model(tmp_path, law: str, k: float) -> bngsim.Model:
    path = tmp_path / "pulse.net"
    path.write_text(NET.format(k=k, law=law))
    return bngsim.Model.from_net(str(path))


def _y(model: bngsim.Model, k: float, **kw) -> float:
    run = bngsim.Simulator(model, method="ode", **kw).run(t_span=(0.0, 60.0 / k), n_points=3)
    return float(np.asarray(run.species)[-1, 2])


@pytest.mark.parametrize("k", [2.0, 1.0, 0.5])
@pytest.mark.parametrize("name", list(PULSES))
def test_the_pulse_is_not_stepped_over(tmp_path, name, k):
    y = _y(_model(tmp_path, PULSES[name], k), k)
    assert y == pytest.approx(10 * (1 - np.exp(-1 / k)), rel=1e-5)


def test_the_pulse_on_time_is_not_stepped_over(tmp_path):
    model = _model(tmp_path, CLOCK, 1.0)
    assert model.time_discontinuity_conditions()
    assert _y(model, 1.0) == pytest.approx(10 * (1 - np.exp(-1.0)), rel=1e-5)


def test_the_operator_is_read_as_the_call_it_is(tmp_path):
    """Stops on a clock and roots on a counter, the same as ``mod()``'s."""
    call = "mod(Tobs+49.99,100)-mod(Tobs+50,100)+0.01"
    for k in (1.0, 2.0):
        mine = _model(tmp_path, PULSES["parenthesized"], k)
        theirs = _model(tmp_path, call, k)
        assert mine.time_discontinuity_conditions() or mine.state_switch_root_conditions()
        assert len(mine.time_discontinuity_conditions()) == len(
            theirs.time_discontinuity_conditions()
        )
        assert len(mine.state_switch_root_conditions()) == len(
            theirs.state_switch_root_conditions()
        )


def test_a_sensitivity_run_on_the_clock_does_not_step_over_the_pulse(tmp_path):
    """Before, the generated C read ``%`` on doubles and did not compile."""
    run = bngsim.Simulator(
        _model(tmp_path, PULSES["bare"], 1.0), method="ode", sensitivity_params=["kx"]
    ).run(t_span=(0.0, 60.0), n_points=3)
    assert float(np.asarray(run.species)[-1, 2]) == pytest.approx(10 * (1 - np.exp(-1)), rel=1e-5)
    dy = float(np.asarray(run.sensitivities)[-1, 2, 0])
    assert dy == pytest.approx(10 * np.exp(-1), rel=1e-4)


def test_a_sensitivity_run_on_a_counter_is_refused_as_the_call_is(tmp_path):
    """Off the clock the pulse is a step over state on the difference quotient,
    which ``mod()``'s spelling is refused for too."""
    for law in (PULSES["bare"], "mod(Tobs+49.99,100)-mod(Tobs+50,100)+0.01"):
        sim = bngsim.Simulator(_model(tmp_path, law, 2.0), method="ode", sensitivity_params=["kx"])
        with pytest.raises(bngsim.SensitivityUnsupportedError):
            sim.run(t_span=(0.0, 30.0), n_points=3)


@pytest.mark.parametrize(
    ("text", "spelled"),
    [
        ("k1*X", "k1*X"),
        ("a%b", "mod(a, b)"),
        ("a*b % c - d", "mod(a*b, c) - d"),
        ("a % b % c", "mod(mod(a, b), c)"),
        ("x - a%b*c", "x - mod(a, b)*c"),
        ("a/b%c", "mod(a/b, c)"),
        ("2^x%3", "mod(2^x, 3)"),
        ("a ^ -b % c", "mod(a ^ -b, c)"),
        ("-a%b", "mod(-a, b)"),
        ("1e-3%x + 2", "mod(1e-3, x) + 2"),
        ("a%(b+c)", "mod(a, (b+c))"),
        ("f(a%b, c)", "f(mod(a, b), c)"),
        ("if(T%2<1, 1, 0)", "if(mod(T, 2)<1, 1, 0)"),
        ("x%y and z", "mod(x, y) and z"),
        ('tfun("a%b.tsv", T)', 'tfun("a%b.tsv", T)'),
        # ExprTk groups with [] and {} as it does with ().
        ("{a+b}%c", "mod({a+b}, c)"),
        ("[a+b]%c - d", "mod([a+b], c) - d"),
        ("tfun([0,1],[0,3], h%2)", "tfun([0,1],[0,3], mod(h, 2))"),
        # Brackets that do not pair up: not ExprTk, so left as written.
        ("a%b)c", "a%b)c"),
        ("f(a%b", "f(a%b"),
        ("(a%b]", "(a%b]"),
    ],
)
def test_the_spelling(text, spelled):
    assert _spell_mod(text) == spelled


DERIVED = """begin parameters
    1 kx 1
    2 base {base}
    3 n 2.9999999
    4 thr {thr}
    5 kd {kd}
end parameters
begin functions
    1 f() kx*if(time() > thr + 10 && time() < thr + 10.01, 100, 0)
end functions
begin species
    1 X() 10
    2 Y() 0
    3 Z() 10
end species
begin reactions
    1 1 2 f
    2 3 0 kd
end reactions
"""


def _derived(tmp_path, *, base: float = 13.0, thr: str = "3", kd: str = "1") -> bngsim.Model:
    path = tmp_path / "derived.net"
    path.write_text(DERIVED.format(base=base, thr=thr, kd=kd))
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize("base", [13.0, -13.0])
def test_a_derived_threshold_written_with_the_operator_is_still_placed(tmp_path, base):
    """A derived parameter is a constant of the run, so its ``%`` is no switch:
    ``thr = base % 10`` crosses where its value says, exactly, as a literal
    does, and the SSA holds the rate between the crossings. Read as ``mod()``
    it was not placed at all. At base = -13 the value is ExprTk's, -3, and not
    the 7 a floored modulus gives."""
    from bngsim._switch_sensitivity import fixed_time_crossings, piecewise_constant_time_functions

    model = _derived(tmp_path, base=base, thr="base%10")
    at = 10 + float(np.fmod(base, 10))
    conditions = model.time_discontinuity_conditions()
    assert fixed_time_crossings(model._core, 0.0, 30.0, conditions) == pytest.approx(
        [at, at + 0.01]
    )
    assert piecewise_constant_time_functions(model._core, 0.0, 30.0, conditions) == ["f"]
    run = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 30.0), n_points=3)
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(10 * (1 - np.exp(-1)), rel=1e-5)


def test_a_derived_rate_constant_written_with_the_operator_is_refused_as_the_call_is(tmp_path):
    """``kd = n % 3`` jumps in n, and the difference quotient over n reads across
    the jump: refused, as ``mod(n, 3)`` is, where it was a CVODE failure."""
    for kd in ("n%3", "mod(n,3)"):
        sim = bngsim.Simulator(_derived(tmp_path, kd=kd), method="ode", sensitivity_params=["n"])
        with pytest.raises(bngsim.SensitivityUnsupportedError):
            sim.run(t_span=(0.0, 1.0), n_points=3)


def _random_expression(rng, depth: int = 0) -> str:
    def atom() -> str:
        pick = rng.integers(0, 5 if depth < 3 else 2)
        if pick == 0:
            return "time()"
        if pick == 1:
            return repr(float(rng.choice([0.3, 1.5, 2, 7, 1e-1, 2.5e0])))
        if pick == 2:
            return f"({_random_expression(rng, depth + 1)})"
        if pick == 3:
            a, b = _random_expression(rng, depth + 1), _random_expression(rng, depth + 1)
            return f"max({a}, {b})"
        return f"if(time()%{rng.choice([2, 3])} < 1, {_random_expression(rng, depth + 1)}, 1)"

    def unary() -> str:
        text = atom() + (f"^{atom()}" if rng.random() < 0.15 else "")
        return ("-" if rng.random() < 0.2 else "") + text

    def term() -> str:
        text = unary()
        for _ in range(rng.integers(0, 3)):
            text += str(rng.choice(["*", "/", "%", "%", " % "])) + unary()
        return text

    text = term()
    for _ in range(rng.integers(0, 3)):
        text += str(rng.choice(["+", "-", " - "])) + term()
    return text


def test_the_spelling_evaluates_as_the_operator_does(tmp_path):
    """Bit for bit, on the model's own evaluator, over random texts."""
    core = _model(tmp_path, PULSES["bare"], 2.0)._core
    times = np.linspace(-7.0, 9.0, 97)
    rng = np.random.default_rng(1127)
    for _ in range(300):
        text = _random_expression(rng)
        spelled = _spell_mod(text)
        assert "%" not in spelled
        np.testing.assert_array_equal(
            core.evaluate_over_times(spelled, times),
            core.evaluate_over_times(text, times),
            err_msg=f"{text!r} spelled {spelled!r}",
        )
