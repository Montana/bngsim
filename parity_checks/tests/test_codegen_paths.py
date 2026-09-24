"""Lock the codegen-path parity sweep's measuring instruments (issue #803).

The sweep's verdicts are only as good as its metrics and its arm wiring, so each
is pinned here against a case whose answer is known:

metrics
  * a column that is roundoff on both sides (1e-20 against states of 1e5) is not
    an O(1) disagreement once the reference run's atol is the floor
  * the scaled sensitivity error is invariant under rescaling a parameter (it
    measures S*p), and an all-zero species does not divide by zero
  * Richardson extrapolation recovers S exactly from two central differences
    whose error is c*h^2 (the truncation-limited case a single step misreads)
  * per-species atol follows each species' own scale, so a 1e12-count species
    does not set atol=1 for its order-1 neighbours
parameter choice
  * a primary behind a derived rate constant (the chain rule the .net path was
    kept for) comes first, then a Functional-law primary, then a direct one
  * a function-owned parameter slot (issue #227) is never chosen
classification
  * any model/.net difference leads the verdict; a tight re-run that did not
    finish is not reported as a disagreement
arm wiring (needs a C compiler)
  * the net and model arms really take the two different codegen paths and
    agree with the interpreter on the derived-rate-constant fixture
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _metrics import per_species_atol, pick_sens_params, richardson, sens_err, traj_err

REPO = Path(__file__).resolve().parents[2]
CELL = REPO / "parity_checks" / "codegen_paths" / "_cell.py"


# ── metrics ──────────────────────────────────────────────────────────────────


def test_traj_err_identical_is_zero_and_shape_or_nan_mismatch_is_inf():
    a = np.random.default_rng(0).random((5, 3))
    assert traj_err(a, a.copy()) == 0.0
    assert traj_err(a[:, :2], a) == float("inf")
    b = a.copy()
    b[2, 1] = np.nan
    assert traj_err(b, a) == float("inf")
    assert traj_err(b, b.copy()) == 0.0


def test_traj_err_roundoff_column_is_not_a_disagreement_with_the_atol_floor():
    # egfr_path: every observable sits at ~1e-20 (species that are numerically
    # zero), so the two runs' roundoff differs by 100% of the column.
    ref = np.array([[0.0, 4e-20], [0.0, -2e-21]])
    a = np.array([[0.0, -3e-20], [0.0, 5e-20]])
    assert traj_err(a, ref) > 1.0
    assert traj_err(a, ref, abs_floor=1e-10) < 1e-9


def test_traj_err_relative_to_each_column():
    ref = np.array([[1.0, 1e6], [2.0, 2e6]])
    a = ref * (1 + 1e-3)
    assert traj_err(a, ref) == pytest.approx(1e-3, rel=1e-6)


def test_sens_err_is_the_scaled_sensitivity_and_survives_a_zero_species():
    y = np.array([[1.0, 0.0], [2.0, 0.0]])  # species 1 identically zero
    S = np.zeros((2, 2, 1))
    S[1, 0, 0] = 5.0
    ref = S.copy()
    ref[1, 0, 0] = 5.5
    # |dS| * p / ymax = 0.5 * 4 / 2
    assert sens_err(S, ref, [4.0], y) == pytest.approx(1.0)
    # Rescaling the parameter by c rescales S by 1/c: the measure is unchanged.
    assert sens_err(S / 10, ref / 10, [40.0], y) == pytest.approx(1.0)
    ref2 = S.copy()
    ref2[0, 1, 0] = 1e-30  # a zero species' column
    assert np.isfinite(sens_err(S, ref2, [1.0], y))


def test_richardson_recovers_s_from_truncation_limited_differences():
    S, c, h = 3.0, 7.0, 1e-4
    fd_coarse, fd_fine = S + c * (10 * h) ** 2, S + c * h**2
    assert richardson(fd_coarse, fd_fine) == pytest.approx(S, abs=1e-14)
    assert abs(fd_fine - S) > 1e-8  # the single fine step alone is off


def test_per_species_atol_follows_each_species_scale():
    y = np.array([[1e12, 1.0, 0.0], [5e11, 2.0, 0.0]])
    atol = per_species_atol(y)
    assert atol[0] == pytest.approx(1.0)
    assert atol[1] == pytest.approx(2e-12)
    assert atol[2] == pytest.approx(1e-12)  # identically zero: floored, never 0


# ── parameter choice ─────────────────────────────────────────────────────────


def _cd():
    return {
        "parameters": [
            {"name": "kon", "is_const": True, "expression": "1"},
            {"name": "chi", "is_const": True, "expression": "10"},
            {"name": "koff", "is_const": True, "expression": "0.5"},
            {"name": "_rateLaw1", "is_const": False, "expression": "chi*kon"},
            {"name": "Km", "is_const": True, "expression": "3"},
            {"name": "f", "is_const": True, "expression": "0"},  # function-owned slot
        ],
        "functions": [{"name": "f", "expression": "koff*A/(Km+A)"}],
        "reactions": [
            {"type": "elementary", "rate_param_indices": [3]},
            {"type": "elementary", "rate_param_indices": [2]},
            {"type": "functional", "function_name": "f", "rate_param_indices": [5]},
        ],
    }


def test_pick_sens_params_orders_chain_then_functional_then_direct():
    P, fam = pick_sens_params(_cd(), max_p=3)
    assert P[0] in ("chi", "kon")  # behind the derived rate constant
    assert P[1] in ("Km", "koff")  # inside the Functional law
    assert fam == {"n_chain": 2, "n_func": 2, "n_direct": 1}


def test_pick_sens_params_never_chooses_a_function_owned_slot():
    P, _ = pick_sens_params(_cd(), max_p=10)
    assert "f" not in P
    assert "_rateLaw1" not in P  # derived, not primary
    assert set(P) == {"chi", "kon", "koff", "Km"}


# ── classification ───────────────────────────────────────────────────────────


def test_classify_leads_with_a_path_difference_and_not_with_an_unfinished_rerun():
    import codegen_paths as cp

    first = {"model_vs_interp": 1e-2, "model_vs_net": 0.0, "model_sens_vs_interp": 1e-2}
    assert cp.classify(first, {"model_vs_interp": 1e-7}).startswith("trajectory within")
    assert "PATHS DIFFER" in cp.classify({**first, "model_vs_net": 0.5}, {"model_vs_interp": 1e-7})
    label = cp.classify(first, {"model_vs_interp": 1e-7})
    assert "sensitivity-run trajectory: tight re-run did not finish" in label
    assert "still differ" not in label


# ── arm wiring ───────────────────────────────────────────────────────────────


def _have_cc() -> bool:
    try:
        from bngsim._codegen import _find_c_compiler

        _find_c_compiler()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_cc(), reason="needs bngsim and a C compiler for codegen")
@pytest.mark.parametrize("sens", [False, True], ids=["plain", "sensitivity"])
def test_arms_take_distinct_codegen_paths_and_agree(tmp_path, sens):
    """The positive control the sweep's comparisons rest on: two arms that
    silently ran the same code would report perfect agreement."""
    net = REPO / "tests" / "data" / "derived_rate_const.net"
    spec = json.dumps({"t_end": 5.0, "n_points": 6, "sens_params": ["kon", "chi"]})
    env = {**os.environ, "BNGSIM_CODEGEN_CACHE_DIR": str(tmp_path / "cg")}
    arms = ("interp", "net_sens", "model_sens") if sens else ("interp", "net", "model")
    out = {}
    for arm in arms:
        subprocess.run(
            [sys.executable, str(CELL), str(net), arm, str(tmp_path / arm), spec],
            check=True,
            env=env,
            capture_output=True,
        )
        meta = json.loads((tmp_path / f"{arm}.json").read_text())
        assert meta["status"] == "OK", meta
        out[arm] = (meta, np.load(tmp_path / f"{arm}.npz"))
    n, m = arms[1], arms[2]
    assert out[n][0]["backend"] == out[m][0]["backend"] == "cc"
    assert out[n][0]["sim_net_path"] and not out[m][0]["sim_net_path"]
    assert out[n][0]["so"] != out[m][0]["so"]
    ref = out["interp"][1]["species"]
    for arm in (n, m):
        assert traj_err(out[arm][1]["species"], ref) < 1e-6
    if sens:
        # A = 1 - B, dA/dt = -chi*kon*A + koff*B: the chain rule through
        # _rateLaw1 = chi*kon makes dA/dkon = chi * dA/d_rateLaw1 = (chi/kon) dA/dchi.
        s = out[m][1]["sens"]
        np.testing.assert_allclose(s[:, :, 0], 10.0 * s[:, :, 1], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(out[n][1]["sens"], s, rtol=1e-8, atol=1e-14)
