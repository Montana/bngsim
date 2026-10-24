"""A run that spends a CVODE step batch says nothing of it on stderr (issue
#1124, item 13).

``max_steps`` is a batch per output point, not a ceiling: CVODE returns
CV_TOO_MUCH_WORK when one is spent, and the run resumes where it stood
(issue #54), raising by name only where a batch makes no progress. SUNDIALS
logged "mxstep steps taken before reaching tout" for every batch all the
same: about 110 lines from a sensitivity run at rtol 1e-11 that succeeded,
with ``C1 := 2 + r*P`` sizing a compartment. That one message is dropped;
every other CVODE error is still logged.

Captured at the file descriptor: SUNDIALS writes to the C ``stderr``.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_DECAY = "species S = 5; k = 0.5; J: S -> ; k*S;"


@pytest.mark.parametrize("sens", [False, True], ids=["plain", "sensitivity"])
def test_a_spent_batch_is_not_logged(capfd, sens):
    """S = 5·e^(-t/2), in batches of 5 steps: dozens of them, each a
    CV_TOO_MUCH_WORK that the run resumes from."""
    model = bngsim.Model.from_antimony_string(_DECAY)
    kw = {"sensitivity_params": ["k"]} if sens else {}
    result = bngsim.Simulator(model, method="ode", **kw).run(
        t_span=(0.0, 10.0), n_points=3, max_steps=5, rtol=1e-10, atol=1e-12
    )
    t = np.asarray(result.time)
    s = np.asarray(result.species)[:, 0]
    np.testing.assert_allclose(s, 5 * np.exp(-0.5 * t), rtol=1e-7)
    assert result.solver_stats["n_steps"] > 50
    assert "mxstep" not in capfd.readouterr().err


@pytest.mark.filterwarnings("ignore")
def test_a_failure_is_still_logged(capfd):
    """S' = S² from 1 has no solution past t = 1: CVODE's own account of the
    failure still reaches stderr, beside the run's error."""
    model = bngsim.Model.from_antimony_string("species S = 1; J: -> S; S^2;")
    with pytest.raises(bngsim.SimulationError):
        bngsim.Simulator(model, method="ode").run(t_span=(0.0, 2.0), n_points=3)
    err = capfd.readouterr().err
    assert "[ERROR]" in err
    assert "mxstep" not in err


def test_the_message_can_be_kept(capfd, monkeypatch):
    """BNGSIM_LOG_SPENT_BATCHES=1 keeps SUNDIALS' own handler, and the line."""
    monkeypatch.setenv("BNGSIM_LOG_SPENT_BATCHES", "1")
    bngsim.Simulator(bngsim.Model.from_antimony_string(_DECAY), method="ode").run(
        t_span=(0.0, 10.0), n_points=3, max_steps=5, rtol=1e-10, atol=1e-12
    )
    assert "mxstep steps taken before reaching tout" in capfd.readouterr().err


def test_the_variable_is_honoured_on_a_reused_simulator(capfd, monkeypatch):
    """The handler was chosen only where the warm path built its persistent
    objects, so a Simulator that had already run kept its filter whatever the
    variable said afterwards (#1142's review). It is chosen on every run."""
    monkeypatch.delenv("BNGSIM_LOG_SPENT_BATCHES", raising=False)
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(_DECAY), method="ode")
    kw = dict(t_span=(0.0, 10.0), n_points=3, max_steps=5, rtol=1e-10, atol=1e-12)
    sim.run(**kw)
    assert "mxstep" not in capfd.readouterr().err
    monkeypatch.setenv("BNGSIM_LOG_SPENT_BATCHES", "1")
    sim.run(**kw)
    assert "mxstep steps taken before reaching tout" in capfd.readouterr().err
    monkeypatch.delenv("BNGSIM_LOG_SPENT_BATCHES")
    sim.run(**kw)
    assert "mxstep" not in capfd.readouterr().err
