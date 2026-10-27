"""A sensitivity run's rows where CVODE returns an output time outside its last
step (issue #1301).

After the sensitivity RHS refuses a state (GH #395's recoverable return),
CVODES can return the time asked for from a step that does not span it.
CV_NORMAL interpolates to tout and does not check that the interpolation
succeeded: its CVodeGetDky refused ("Illegal value for t"), the state vector was
left as it was, and the row took the state of the call before, at the time
asked for. The per-row CVodeGetSens then failed too, and the row's
sensitivities were left at zero, with no error: 24 rows of 101 on
MODEL2001200002 (Mpekris 2017). Its second message, a solver "at t = 2.6e-312",
is SUNDIALS 7.2.1's CVodeGetSensDky1 printing its MSGCV_BAD_T with no
arguments, not a stale handle.

Each such row is now read where the run stands, as a sensitivity run records
its rows, and counted in solver_stats["n_rows_outside_last_step"].
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# MODEL2001200002 (Mpekris 2017), 7 species, as #1300's
# test_sens_rhs_at_the_clamped_state.py writes it.
MPEKRIS = """
model mpekris2017
  D := d_im*((L/T)^lamda_im/(s + (L/T)^lamda_im))*T;
  J1:  => C; G*S_ftr*C + p_CT*Csc + p_IT*I;
  J2: T => ; c*N*T + D + (p_TC + p_TI)*T;
  J3:  => Csc; a_CSC*GS_CSC*Csc + p_TC*T + p_IC*I;
  J4: Csc => ; c_CSC*N*Csc + D + (p_CT + p_CI)*Csc;
  J5:  => I; a_I*GS_I*I + p_TI*T + p_CI*Csc;
  J6: I => ; c*N*I + D + (p_IT + p_IC)*I;
  J7:  => N; sigma_nk + (g_NK*T*T/(h + T*T))*N;
  J8: N => ; f_NK*N + p_im*N*T + lamda_reg*Treg*N;
  J9:  => L; (j_T8*D*D/(k_im + D*D))*L + r*N*T;
  J10: L => ; m_T8*L + q*L*T + lamda_reg*Treg*L;
  J11:  => Treg; g_reg*Treg;
  J12: Treg => ; m_reg*Treg;
  C = 1; T = 1; Csc = 1; I = 1; N = 1; L = 1; Treg = 1;
  G = 0.1; S_ftr = 0.1; GS_CSC = 0.1; c_CSC = 0.1; GS_I = 0.1; a_CSC = 1; a_I = 1;
  c = 3.23e-07; d_im = 1.43; lamda_im = 1.36; s = 2.73; sigma_nk = 13000;
  f_NK = 0.0412; m_T8 = 0.02; m_reg = 0.02; g_NK = 0.025; g_reg = 0.0375;
  h = 20200000; p_im = 1e-07; k_im = 20200000; q = 3.42e-10; r = 1.1e-07;
  lamda_reg = 100; p_TC = 0.55; p_CT = 1; p_CI = 0.58; p_IC = 0.96; p_TI = 0.21;
  p_IT = 1; j_T8 = 0.0375;
end
"""

_RUN = {"t_span": (0, 100), "n_points": 101, "rtol": 1e-9, "atol": 1e-12}


@pytest.fixture(scope="module")
def sens_run():
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(MPEKRIS),
        method="ode",
        sensitivity_params=["sigma_nk", "g_reg"],
    )
    return sim.run(**_RUN)


def test_the_run_reaches_rows_outside_the_last_step(sens_run):
    """The guard: this model reaches the path at all. A change that stops it
    (#1300 evaluates the sensitivity RHS at the value RHS's clamped state) makes
    the tests below pass for nothing, so this one fails instead and asks for
    another model."""
    assert sens_run.solver_stats["n_rows_outside_last_step"] > 0


def test_no_row_of_sensitivities_is_left_at_zero(sens_run):
    s = np.asarray(sens_run.sensitivities)
    zero = [i for i in range(1, len(s)) if not np.any(s[i])]
    assert zero == [], f"rows whose sensitivities were never read: {zero}"
    assert np.isfinite(s).all()


def test_each_row_is_read_where_the_run_stands(sens_run):
    """A row read where the run stood is at or past the time asked for, within
    one output interval of it, and the grid stays in order."""
    t = np.asarray(sens_run.time)
    grid = np.linspace(0, 100, 101)
    assert (t >= grid - 1e-9).all()
    assert (t - grid < 1.0).all()
    assert (np.diff(t) > 0).all()


def test_a_run_without_sensitivities_counts_none():
    r = bngsim.Simulator(bngsim.Model.from_antimony_string(MPEKRIS), method="ode").run(**_RUN)
    assert r.solver_stats["n_rows_outside_last_step"] == 0
    np.testing.assert_array_equal(np.asarray(r.time), np.linspace(0, 100, 101))
