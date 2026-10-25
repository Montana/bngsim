"""A bend root whose ``if()`` branch is off at the crossing restarts nothing
(issue #1147).

#1089 roots every ``min``/``max``/``abs`` call over state a rate law reads, and
the rulehub corpus's division guard ``if(c, k*S/max(S, 0.001), 0)`` is a
common one. Where ``c`` is on, the bend at ``S = 0.001`` is a kink worth a
restart. Where ``c`` is off, the rate law is 0 on both sides, yet the run
restarted there: in sp_kalman_filter and cs_pid_controller every crossing was
of that kind, and Jacobian evaluations rose ×1.5 and ×1.27 with accuracy
unchanged (#1139). Each call root now carries the branches it stands in, and a
plain run steps on through a crossing whose every branch is off.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import _switch_sensitivity as ss

# S rises from 0 through its guard at 0.001 with the branch off (time < 50).
_OFF = (
    "species S = 0; k = 0.5\nJ1: -> S; 1\n"
    "J2: S -> ; piecewise(k*S*S/max(S, 0.001), time > 50, 0)\n"
)
# S decays through it with the branch on.
_ON = "species S = 2; k = 0.5\nJ2: S -> ; piecewise(k*S*S/max(S, 0.001), time < 50, 0)\n"


def _run(text, roots=True, t_end=10.0, rtol=1e-8, atol=1e-8, monkeypatch=None, codegen=False):
    if not roots:
        monkeypatch.setattr(bngsim.Model, "_state_switch_roots_beside", lambda self, covered: ())
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", codegen=codegen)
    if codegen:
        assert sim.codegen_backend != "none"
    run = sim.run(t_span=(0.0, t_end), n_points=int(t_end) + 1, rtol=rtol, atol=atol)
    stats = run.solver_stats
    return (
        np.asarray(run.species)[:, 0],
        (stats["n_steps"], stats["n_rhs_evals"], stats["n_jac_evals"]),
    )


class TestBranchGuard:
    def test_a_then_branch_is_its_condition(self):
        text = "if(c>1, k/max(S,0.001), 0)"
        assert ss._branch_guard(text, text.index("max")) == "((c>1)!=0)"

    def test_an_else_branch_is_its_negation(self):
        text = "if(c>1, 0, k/max(S,0.001))"
        assert ss._branch_guard(text, text.index("max")) == "((c>1)==0)"

    def test_nested_branches_multiply(self):
        text = "if(a>1, if(b<2, 0, max(S,0.001)), 0)"
        assert ss._branch_guard(text, text.index("max")) == "((a>1)!=0)*((b<2)==0)"

    def test_a_condition_is_read_as_exprtk_reads_it(self):
        """ExprTk takes a branch wherever its condition is nonzero, so a
        condition that is a number, not a comparison, is tested against 0."""
        text = "if(F, k/max(S,0.001), 0)"
        assert ss._branch_guard(text, text.index("max")) == "((F)!=0)"

    def test_outside_every_branch_is_no_guard(self):
        text = "k/max(S,0.001) + if(a>1, 1, 0)"
        assert ss._branch_guard(text, text.index("max")) is None

    @pytest.mark.parametrize(
        ("text", "guard"),
        [
            # an if() that closed before the call holds nothing of it
            ("if(c>1, 1, 0) + k/max(S,0.001)", None),
            ("if(max(S,0.001)>1, 1, 0)", None),  # in the condition itself
            ("if(c>1, k/max(S,0.001))", None),  # not three arguments
            ("if(a>1, if(b<2, 1, 2) + max(S,0.001), 0)", "((a>1)!=0)"),
            ("if(a>1, 0, if (b<2, max(S,0.001), 0))", "((a>1)==0)*((b<2)!=0)"),
            ("if(a>1, (k, max(S,0.001), 0)", None),  # an if( never closed
        ],
    )
    def test_the_edges_of_an_if(self, text, guard):
        assert ss._branch_guard(text, text.rindex("max")) == guard


def test_the_root_carries_its_branch():
    model = bngsim.Model.from_antimony_string(_OFF)
    covered = model.time_discontinuity_conditions()
    assert model._state_switch_roots_beside(covered) == ("(S)<(0.001)",)
    assert model._state_switch_guards_beside(covered) == ("(((time()>50))!=0)",)


_MANY_CALLS_NET = """begin parameters
    1 k 1
end parameters
begin functions
    1 f() {law}
end functions
begin species
    1 X() 5
end species
begin reactions
    1 0 1 f
end reactions
begin groups
    1 Xo 1
end groups
"""


def test_a_branch_holding_many_calls_is_read_once(tmp_path):
    """The guard of each call used to be read afresh from the text: every
    if() before the call matched to its close paren and its arguments split,
    which for the one if() holding all of them is the whole rate law each
    time. 2,000 calls took 10.7 s, before any run and on every plain one;
    the text is now read once."""
    import time

    n = 2000
    path = tmp_path / "many_calls.net"
    path.write_text(_MANY_CALLS_NET.format(law=f"if(Xo>3,{'+'.join(['k*max(Xo-1,0)'] * n)},2*k)"))
    model = bngsim.Model.from_net(str(path))
    covered = model.time_discontinuity_conditions()
    began = time.perf_counter()
    roots = model._state_switch_roots_beside(covered)
    elapsed = time.perf_counter() - began
    assert roots == ("Xo>3", "(Xo-1)<(0)")
    assert model._state_switch_guards_beside(covered) == (
        "",
        f"max({', '.join(['((Xo>3)!=0)'] * n)})",
    )
    assert elapsed < 2.0


def test_a_reading_outside_every_branch_leaves_the_root_unguarded():
    """The same bend, once inside the branch and once outside it."""
    text = _OFF.replace("time > 50, 0)", "time > 50, 0) + 0.1*max(S, 0.001)")
    model = bngsim.Model.from_antimony_string(text)
    covered = model.time_discontinuity_conditions()
    assert model._state_switch_guards_beside(covered) == ("",)


def test_a_crossing_with_its_branch_off_steps_as_if_unrooted(monkeypatch):
    """The issue's Kalman and PID shape: the run's counters are the run's
    without the root (main took 11, 17 and 11 for 9, 12 and 9)."""
    y, counters = _run(_OFF)
    y0, counters0 = _run(_OFF, roots=False, monkeypatch=monkeypatch)
    assert counters == counters0
    np.testing.assert_allclose(y, y0, rtol=1e-12)
    t = np.arange(11.0)
    np.testing.assert_allclose(y, t, rtol=1e-6)  # S = t: J2 is off throughout


def test_a_crossing_with_its_branch_on_still_restarts(monkeypatch):
    """S decays through 0.001 at t = 2·ln(2000) with the branch on: the root
    restarts there, so the counters are not the unrooted run's."""
    _y, counters = _run(_ON, t_end=30.0)
    _y0, counters0 = _run(_ON, roots=False, t_end=30.0, monkeypatch=monkeypatch)
    assert counters != counters0


def _has_cc() -> bool:
    from bngsim import _codegen

    try:
        _codegen._find_c_compiler()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_cc(), reason="no C compiler available")
def test_the_compiled_backend_takes_the_guards_too(monkeypatch):
    """The guards are read by the core, beside the roots, whatever computes the
    right-hand side; #1148's runs were all interpreted. Compiled, the off branch
    steps as if unrooted and the on branch still restarts."""
    y, counters = _run(_OFF, codegen=True)
    _y0, counters0 = _run(_OFF, roots=False, monkeypatch=monkeypatch, codegen=True)
    assert counters == counters0
    np.testing.assert_allclose(y, np.arange(11.0), rtol=1e-6)
    monkeypatch.undo()
    _y, on = _run(_ON, t_end=30.0, codegen=True)
    _y0, on0 = _run(_ON, roots=False, t_end=30.0, monkeypatch=monkeypatch, codegen=True)
    assert on != on0


@pytest.mark.parametrize("text", [_OFF, _ON], ids=["off", "on"])
def test_a_run_with_sensitivities_is_unchanged(text):
    """The guards are a plain run's: a run with columns roots as before."""
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
        t_span=(0.0, 10.0), n_points=11, rtol=1e-8, atol=1e-10
    )
    assert np.all(np.isfinite(np.asarray(run.sensitivities)))


# The rulehub division guard as a .net function, its branch gated by a number F
# rather than a comparison. ExprTk runs the then-branch wherever F != 0, so every
# nonzero F is the same ODE, and S decays through the bend with the branch on.
_NUMERIC_NET = """begin parameters
    1 k 0.5
    2 F {F}
end parameters
begin species
    1 S() 2
end species
begin functions
    1 r() if(F, k*Sobs/max(Sobs,0.001), 0)
end functions
begin reactions
    1 1 0 r #_R1
end reactions
begin groups
    1 Sobs 1
end groups
"""


def _net_counters(tmp_path, F, monkeypatch=None, codegen=False):
    if monkeypatch is not None:
        monkeypatch.setattr(bngsim.Model, "_state_switch_roots_beside", lambda self, covered: ())
    path = tmp_path / f"numeric_{F}.net"
    path.write_text(_NUMERIC_NET.format(F=F))
    run = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=codegen).run(
        t_span=(0.0, 30.0), n_points=31, rtol=1e-8, atol=1e-8
    )
    stats = run.solver_stats
    return stats["n_steps"], stats["n_rhs_evals"], stats["n_jac_evals"]


@pytest.mark.parametrize("codegen", [False, True], ids=["exprtk", "codegen"])
@pytest.mark.parametrize("F", ["0.3", "-1", "2"])
def test_a_numeric_condition_is_on_where_exprtk_takes_its_branch(
    tmp_path, monkeypatch, F, codegen
):
    """F = 0.3 and F = -1 run the branch as F = 1 does, so the crossing restarts
    there too. Read against 0.5, (F) called those two off and stepped over the
    kink: the counters were the unrooted run's. The guard is the evaluator's to
    read whichever backend runs the right-hand side."""
    rooted = _net_counters(tmp_path, F, codegen=codegen)
    assert rooted == _net_counters(tmp_path, "1", codegen=codegen)
    assert rooted != _net_counters(tmp_path, F, monkeypatch=monkeypatch, codegen=codegen)


@pytest.mark.parametrize(
    ("second", "restarts"), [("time > 60", False), ("time < 5", True)], ids=["both-off", "one-on"]
)
def test_two_branches_on_one_surface_restart_where_either_is_on(monkeypatch, second, restarts):
    """Two rate laws read the same bend, each in its own branch: the root's
    guard is either branch on, max(g1, g2). With both off at the crossing the
    run steps as the unrooted one; with one on it restarts."""
    text = _OFF + f"J3: S -> ; piecewise(0.5*k*S*S/max(S, 0.001), {second}, 0)\n"
    model = bngsim.Model.from_antimony_string(text)
    (guard,) = model._state_switch_guards_beside(model.time_discontinuity_conditions())
    assert guard.startswith("max(")
    _y, counters = _run(text)
    _y0, counters0 = _run(text, roots=False, monkeypatch=monkeypatch)
    assert (counters != counters0) is restarts


def test_a_guard_that_does_not_compile_restarts(monkeypatch):
    """The core cannot read a guard it cannot compile, so it restarts at the
    crossing as it did before guards: the counters are the unguarded run's."""
    _y, guarded = _run(_OFF)
    monkeypatch.setattr(
        bngsim.Model,
        "_state_switch_guards_beside",
        lambda self, covered: ("",) * len(self._state_switch_roots_beside(covered)),
    )
    _y, unguarded = _run(_OFF)
    monkeypatch.setattr(
        bngsim.Model,
        "_state_switch_guards_beside",
        lambda self, covered: ("((",) * len(self._state_switch_roots_beside(covered)),
    )
    _y, broken = _run(_OFF)
    assert broken == unguarded != guarded


def test_a_cleared_memo_does_not_take_the_guards(tmp_path):
    """The memo the guards live in is shared with clones and cleared whole at its
    cap. The model kept its roots in hand and lost the guards, so every later
    run restarted at the crossings they step through."""
    from bngsim._model import _DERIVATION_MEMO_MAX, _remember

    model = bngsim.Model.from_antimony_string(_OFF)
    covered = model.time_discontinuity_conditions()
    guards = model._state_switch_guards_beside(covered)
    assert any(guards)
    for i in range(_DERIVATION_MEMO_MAX):
        _remember(model._state_switch_root_memo, ("another writer", i), ())
    assert ("guards", covered) not in model._state_switch_root_memo
    assert model._state_switch_guards_beside(covered) == guards
