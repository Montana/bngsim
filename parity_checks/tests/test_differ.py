"""Lock the shared cross-engine comparison protocol (`_core/differ.py`).

`differ.deterministic_verdict` / `ensemble_verdict` are the verdict both parity
suites gate on (issue #69 worries about this protocol drifting), so each
contract below is anchored to an oracle, not to a frozen number:

deterministic_verdict
  * identical arrays pass with zero residual (the trivial floor)
  * a concentrated relative blow-up past the hard ceiling is a DIFF even when
    it is a vanishing fraction of cells (ceilings beat the budget)
  * a few soft cells within the fail-fraction budget are forgiven; one more
    than the budget allows is a DIFF (the budget boundary, parametrized)
  * an underflow-scale disagreement (both sides below the file-scale floor) is
    forgiven by the near-zero backstop
  * a cell where exactly one side is non-finite (NaN/inf vs a finite value) is
    an unconditional hard fail — the regression guard for the bug where the
    near-zero backstop silently forgave it
  * both-NaN is a zero-diff pass
  * the verdict is invariant under a positive global rescaling of both inputs
    (metamorphic: the protocol is scale-aware), and `forgive_mask` only ever
    loosens (superset property)
  * a shape mismatch raises
  * the movement gate (issue #873): a frozen engine, and a 4% error on one
    species, fail where a clamped reservoir, a nearly constant one or a large
    baseline sets the scale; an engine that agrees passes in each, and so does
    0.9% of every species; a reference where nothing moves is ``no_signal``;
    the mirror image fails too (an engine that lets a clamped species drift,
    or makes one the reference never makes); the forgive mask still applies,
    and the rr/amici row comment carries the gate's numbers

ensemble_verdict
  * equal means pass; a sub-particle cell (one side floored to 0, the other a
    small nonzero with tiny SEM — the BIOMD2 class in miniature) is a DIFF
  * a difference within K SEM passes; the relative escape hatch passes a
    deterministic cell whose SEM has collapsed
  * a near-zero cell is skipped (counts as a pass)
  * one-side-NaN mean fails, both-NaN passes
  * the frac_pass>=0.99 gate is exact at the boundary (parametrized)
  * a shape mismatch raises
"""

from __future__ import annotations

import numpy as np
import pytest
from _core import differ


def _passed(a, b, **kw):
    v = differ.deterministic_verdict(np.asarray(a, float), np.asarray(b, float), **kw)
    return v["passed"], v


# --------------------------------------------------------------------------- #
# deterministic_verdict
# --------------------------------------------------------------------------- #
def test_identical_arrays_pass_with_zero_residual():
    """a == b ⇒ PASS, no failing cells, zero reported residual."""
    rng = np.random.default_rng(0)
    a = rng.uniform(1e-3, 10.0, size=(50, 6))
    ok, v = _passed(a, a.copy())
    assert ok
    assert v["n_fail"] == 0
    assert v["max_rel"] == 0.0 and v["max_abs"] == 0.0


def test_concentrated_relative_blowup_is_diff_despite_tiny_fraction():
    """One 20% divergence on a unit-magnitude cell is a DIFF.

    It is 1/500 of cells — far inside the fail-fraction budget — but the hard
    relative ceiling (0.05) is unconditional, so the budget cannot forgive it.
    Oracle: a concentrated real divergence must never be averaged away.
    """
    a = np.ones((100, 5))
    b = np.ones((100, 5))
    b[0, 0] = 1.2  # 16.7% relative on a magnitude-carrying cell
    ok, v = _passed(a, b)
    assert not ok
    assert v["n_hard_fail"] == 1
    assert v["max_rel"] == pytest.approx(0.2 / 1.2, rel=1e-6)


def _swinging(n_t=100, n_c=5):
    """Columns that each swing between -1 and 1. A 1% relative shift is then half
    a percent of the column's movement, inside MOVE_CEILING, so the magnitude
    gates decide; on a flat column the movement gate would (issue #873)."""
    t = np.linspace(0.0, 4.0 * np.pi, n_t)
    return np.column_stack([np.sin(t + k) for k in range(n_c)])


@pytest.mark.parametrize("n_soft", [3, 50, 300])
def test_subceiling_column_forgiven_regardless_of_count(n_soft):
    """A column whose worst divergence is a 1% relative shift (above REL_TOL=1e-4,
    below HARD_REL_CEILING=0.05) at its own peak scale never diverges past the
    ceiling, so the peak-relative significance gate forgives it OUTRIGHT — no
    matter how many of its cells trip the tight per-cell soft bar. This is the
    decayed-tail / integrator-noise class the gate was added for (rr_check forgives
    it on reldiv<0.05): an arbitrarily large fraction of sub-ceiling soft cells is
    not a divergence. (Before the gate this DIFFed once the soft fraction passed
    FAIL_FRAC_BUDGET — the behavior change this test pins.)

    The columns move (``_swinging``): a 1% blip on a column that is flat on the
    other side is a movement the other engine never made, which the movement
    gate fails whichever side it is on (issue #873)."""
    b = _swinging()
    a = b.copy()
    flat = b.reshape(-1)
    flat[:n_soft] *= 1.01  # 1% relative on a unit-magnitude column: < HARD_REL_CEILING
    ok, v = _passed(a, b)
    assert ok
    assert v["n_hard_fail"] == 0
    assert v["max_rel"] == 0.0  # nothing survives to the verdict


def test_budget_forgives_soft_cells_within_a_real_column():
    """The fail-fraction budget still forgives a few soft cells that coexist with
    a genuine (real) divergence in the same column. The column has one 10% hard
    cell (so it is a real divergence the gate does NOT forgive) plus n soft 1%
    cells. The model DIFFs either way (the hard cell), but the budget controls
    whether the soft cells are reported as forgiven vs counted — the budget
    bookkeeping the gate keeps. (On moving columns, as above; rows 12, 37 and 62
    are where column 0 is at +-1.)"""
    b = _swinging()
    a = b.copy()
    b[12, 0] *= 1.10  # hard: 9.1% of col peak > HARD_REL_CEILING → column is real
    b[37, 0] *= 1.01  # 2 soft cells (2/500 = 0.004 <= FAIL_FRAC_BUDGET)
    b[62, 0] *= 1.01
    ok, v = _passed(a, b)
    assert not ok  # the hard cell DIFFs regardless
    assert v["n_hard_fail"] == 1
    assert v["budget_forgiven"] == 2  # the 2 soft cells stay within budget


def test_near_zero_backstop_forgives_underflow_disagreement():
    """Two columns: a unit-magnitude one (sets the file scale ~1) and a column
    where both sides are ~1e-15 but differ by 2x. The tiny column is below the
    file-scale near-zero floor, so its (relatively huge) disagreement is
    underflow, not signal — forgiven. Oracle: sub-1e-12-of-scale noise is not a
    divergence."""
    a = np.array([[1.0, 1e-15], [1.0, 2e-15], [1.0, 3e-15]])
    b = np.array([[1.0, 2e-15], [1.0, 4e-15], [1.0, 6e-15]])
    ok, _ = _passed(a, b)
    assert ok


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_one_side_nonfinite_is_unconditional_hard_fail(bad):
    """A cell where exactly one engine produced NaN/inf while the other gave a
    finite value is an unambiguous divergence and must DIFF — regardless of
    magnitude. Regression guard: np.maximum collapses the cell's colmag to 0,
    so without an explicit guard the near-zero backstop forgives it and a real
    blow-up passes silently (the bug this test was written to catch)."""
    a = np.ones((5, 2))
    b = np.ones((5, 2))
    a[0, 0] = bad
    ok, v = _passed(a, b)
    assert not ok
    assert v["n_hard_fail"] >= 1


def test_both_nan_cell_is_a_pass():
    """Both engines NaN at the same cell ⇒ zero diff there, no fail."""
    a = np.ones((4, 2))
    b = np.ones((4, 2))
    a[1, 1] = b[1, 1] = np.nan
    ok, _ = _passed(a, b)
    assert ok


@pytest.mark.parametrize("c", [1.0, 1e3, 1e6])
def test_verdict_is_invariant_under_positive_rescaling(c):
    """Scaling both inputs by c>0 scales absd, the file scale, col_peak and
    colmag identically, so every per-cell tolerance term scales with the data:
    the pass/fail verdict is invariant. Metamorphic oracle for scale-awareness.
    Mixed case: mostly-agreeing with two 1%-soft cells and one 10%-hard cell."""
    a = np.ones((100, 5))
    b = np.ones((100, 5))
    b[0, 0] = 1.01
    b[1, 0] = 1.01
    b[2, 1] = 1.10  # hard
    base_ok, base = _passed(a, b)
    scaled_ok, scaled = _passed(c * a, c * b)
    assert scaled_ok is base_ok
    assert scaled["n_hard_fail"] == base["n_hard_fail"]


def test_decayed_tail_relative_divergence_is_forgiven():
    """A species that peaks at the file scale then decays to a tiny tail, where
    the two engines disagree RELATIVELY on that tail (per-cell reld ~0.9, hard
    under the old per-cell rule), is forgiven by the peak-relative gate:
    |a-b|/col_peak ≈ 1e-3 ≪ HARD_REL_CEILING — the tail is measured against the
    species' OWN peak, not its vanishing value. Oracle: a divergence below the
    species' dynamic range is integrator underflow, the artifact class the gate
    was added for (rr_check forgives it on reldiv<0.05)."""
    n = 50
    a = np.ones((n, 2))  # col1 = 1.0 sets the file scale; col0 peaks at 1.0
    a[10:, 0] = 1e-4  # col0 decays to a 1e-4 tail (peak stays 1.0)
    b = a.copy()
    b[10:, 0] = 1.1e-3  # tail disagrees: per-cell reld ≈ 0.91, |a-b|=1e-3=1e-3*peak
    ok, v = _passed(a, b)
    assert ok
    assert v["n_hard_fail"] == 0
    assert v["max_rel"] == 0.0


def test_significant_species_past_ceiling_still_fails():
    """A meaningful-magnitude species (signif > SIGNIF_FLOOR) that diverges past
    HARD_REL_CEILING at its own peak scale is a real divergence the gate must NOT
    forgive. col0 carries ~12% of the file scale (well above the 1e-3 floor) and
    disagrees ~17% at its peak — a hard fail, DIFF."""
    n = 50
    a = np.ones((n, 2))
    a[:, 1] = 10.0  # scale-setter: file peak = 10
    a[:, 0] = 1.0  # col0 peak ≈ 0.1*scale → signif ≈ 0.1 ≫ SIGNIF_FLOOR
    b = a.copy()
    b[5, 0] = 1.2  # 0.2/1.2 = 16.7% of col0 peak > HARD_REL_CEILING
    ok, v = _passed(a, b)
    assert not ok
    assert v["n_hard_fail"] == 1
    assert v["max_rel"] == pytest.approx(0.2 / 1.2, rel=1e-6)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_one_side_nonfinite_on_subsignificance_column_is_still_hard(bad):
    """A one-side-non-finite cell is an unconditional hard fail even when its
    column is far below the dynamic range (signif ≪ SIGNIF_FLOOR): a blow-up on
    one engine is never forgiven by the significance gate. Regression guard that
    the dynamic-range forgiveness explicitly excludes one_side_nonfinite."""
    n = 20
    a = np.ones((n, 2))
    a[:, 1] = 1e6  # scale-setter: file peak = 1e6
    a[:, 0] = 1e-3  # col0 ≈ 1e-9 of scale → sub-significant
    b = a.copy()
    b[0, 0] = bad  # one engine blows up on the sub-significance column
    ok, v = _passed(a, b)
    assert not ok
    assert v["n_hard_fail"] >= 1


def test_forgive_mask_only_loosens():
    """forgive_mask forgives cells *before* the gate/budget, so masking a genuine
    divergence can only flip DIFF→PASS, never the reverse. Here one 10% cell on a
    significant column is a real divergence (DIFF); masking it → PASS. Confirms
    omitting the mask (rr_parity passes None) is strictly tighter."""
    a = np.ones((100, 5))
    b = np.ones((100, 5))
    b[0, 0] = 1.10  # real: column diverges past HARD_REL_CEILING at its own peak
    assert not _passed(a, b)[0]  # DIFF
    mask = np.zeros((100, 5), dtype=bool)
    mask[0, 0] = True
    assert _passed(a, b, forgive_mask=mask)[0]  # forgiven → PASS


def test_deterministic_shape_mismatch_raises():
    with pytest.raises(ValueError):
        differ.deterministic_verdict(np.ones((3, 2)), np.ones((3, 3)))


# --------------------------------------------------------------------------- #
# ensemble_verdict
# --------------------------------------------------------------------------- #
def _ens(ma, sa, mb, sb):
    return differ.ensemble_verdict(
        np.asarray(ma, float),
        np.asarray(sa, float),
        np.asarray(mb, float),
        np.asarray(sb, float),
    )


def test_ensemble_equal_means_pass():
    """Identical ensemble means ⇒ every cell passes the K-sigma test."""
    rng = np.random.default_rng(1)
    m = rng.uniform(1.0, 100.0, size=(20, 4))
    s = np.full_like(m, 0.5)
    v = _ens(m, s, m.copy(), s.copy())
    assert v["passed"] and v["frac_pass"] == 1.0


def test_ensemble_sub_particle_cell_is_diff():
    """The BIOMD2 sub-particle class in miniature: bngsim floors to 0, RR keeps
    a small nonzero value with a tiny SEM. The mean difference is many SEM and
    far above the relative escape hatch, so the cell fails and frac_pass < 0.99.
    Oracle: a 0-vs-nonzero ensemble split is a genuine ensemble divergence."""
    mean_a = np.zeros((10, 3))
    mean_b = np.full((10, 3), 1e-5)
    sem = np.full((10, 3), 1e-9)
    v = _ens(mean_a, sem, mean_b, sem)
    assert not v["passed"]
    assert v["frac_pass"] == 0.0
    assert np.isinf(v["max_z"]) or v["max_z"] > 100


def test_ensemble_within_k_sigma_passes():
    """Means differing by 2 combined SEM pass at K=3. se = sqrt(sa²+sb²)."""
    mean_a = np.full((5, 2), 10.0)
    sa = sb = np.full((5, 2), 1.0)
    se = np.sqrt(sa**2 + sb**2)
    mean_b = mean_a + 2.0 * se  # 2 sigma < K=3
    v = _ens(mean_a, sa, mean_b, sb)
    assert v["passed"]


def test_ensemble_k_for_scales_with_sqrt_n():
    """Effect-size-preserving gate (GH #190): K = ENSEMBLE_K·sqrt(N/base), unchanged
    at the calibration N, 2× at 4× the replicates, sqrt(10)× at 10×."""
    assert differ.ensemble_k_for(differ.ENSEMBLE_K_BASE_N) == pytest.approx(differ.ENSEMBLE_K)
    assert differ.ensemble_k_for(4 * differ.ENSEMBLE_K_BASE_N) == pytest.approx(
        2.0 * differ.ENSEMBLE_K
    )
    assert differ.ensemble_k_for(100) == pytest.approx(
        differ.ENSEMBLE_K * (100 / differ.ENSEMBLE_K_BASE_N) ** 0.5
    )


def test_ensemble_verdict_honors_scaled_k():
    """A 5σ difference FAILS at the base K=3 but PASSES at the N=100 gate (K≈9.49):
    more replicates sharpen precision without flagging the same meaningful diff."""
    mean_a = np.full((5, 2), 10.0)
    sa = sb = np.full((5, 2), 1.0)
    se = np.sqrt(sa**2 + sb**2)
    mean_b = mean_a + 5.0 * se
    assert not differ.ensemble_verdict(mean_a, sa, mean_b, sb)["passed"]
    scaled = differ.ensemble_verdict(mean_a, sa, mean_b, sb, k=differ.ensemble_k_for(100))
    assert scaled["passed"]
    assert scaled["k"] == pytest.approx(differ.ensemble_k_for(100))


def test_ensemble_relative_escape_hatch_passes_collapsed_sem():
    """A near-deterministic cell (SEM → 0) whose means differ by < REL_TOL is
    passed by the relative escape hatch, even though the sigma test alone would
    reject any nonzero difference at zero SEM. Oracle: effectively-deterministic
    agreement should not be punished for having no sampling spread."""
    mean_a = np.full((5, 2), 1.0)
    mean_b = np.full((5, 2), 1.0 + 5e-5)  # 5e-5 < REL_TOL (1e-4)
    zero = np.zeros((5, 2))
    v = _ens(mean_a, zero, mean_b, zero)
    assert v["passed"]


def test_ensemble_near_zero_cell_is_skipped():
    """A cell where both means are below NEAR_ZERO_REL*scale is skipped (treated
    as a pass): with one O(100) cell setting the scale, an O(1e-9*scale) cell
    that disagrees does not drag frac_pass down."""
    mean_a = np.array([[100.0, 1e-8]])
    mean_b = np.array([[100.0, 5e-8]])  # tiny cell disagrees 5x, but near-zero
    sem = np.array([[1.0, 1e-12]])
    v = _ens(mean_a, sem, mean_b, sem)
    assert v["passed"] and v["frac_pass"] == 1.0


def test_ensemble_one_side_nan_fails_both_nan_passes():
    """One-side-NaN mean is a fail; both-NaN is a pass (the cells are aligned
    missing values, not a divergence)."""
    mean_a = np.array([[1.0, np.nan], [1.0, 1.0]])
    mean_b = np.array([[1.0, 2.0], [1.0, 1.0]])
    sem = np.full((2, 2), 0.1)
    assert not _ens(mean_a, sem, mean_b, sem)["passed"]
    mean_a2 = np.array([[1.0, np.nan]])
    mean_b2 = np.array([[1.0, np.nan]])
    assert _ens(mean_a2, np.full((1, 2), 0.1), mean_b2, np.full((1, 2), 0.1))["passed"]


@pytest.mark.parametrize(
    "n_fail, expect_pass",
    [
        (1, True),  # 99/100 pass = 0.99 == ENSEMBLE_PASS_FRAC → PASS
        (2, False),  # 98/100 = 0.98 < 0.99 → DIFF
    ],
)
def test_ensemble_frac_pass_gate_boundary(n_fail, expect_pass):
    """The model passes iff >= 99% of cells pass. 100 cells ⇒ the gate sits
    between 1 and 2 failing cells. Failing cells are a 0-vs-nonzero split."""
    n = 100
    mean_a = np.full((n, 1), 5.0)
    mean_b = np.full((n, 1), 5.0)
    sem = np.full((n, 1), 0.01)
    mean_b[:n_fail, 0] = 50.0  # gross, many-sigma, not near-zero, not rel-ok
    v = _ens(mean_a, sem, mean_b, sem)
    assert v["passed"] is expect_pass
    assert v["frac_pass"] == pytest.approx((n - n_fail) / n)


def test_ensemble_shape_mismatch_raises():
    with pytest.raises(ValueError):
        differ.ensemble_verdict(np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 3)))


# --------------------------------------------------------------------------- #
# The movement gate (issue #873)
# --------------------------------------------------------------------------- #
def _trajectory(n=121, *, reservoir=None, baseline=0.0):
    """A reference with a substrate decaying from 6e5, a product rising to it,
    and optionally a reservoir column and a large baseline under the moving ones:
    the shapes of #873's three mechanisms."""
    t = np.linspace(0.0, 3600.0, n)
    s = 6.022e5 * np.exp(-t / 900.0)
    cols = [baseline + s, baseline + (6.022e5 - s), baseline + 514.0 * (1 - np.exp(-t / 300.0))]
    if reservoir is not None:
        cols.append(reservoir(t))
    return np.column_stack(cols)


_MECHANISMS = {
    # (a) a clamped species sets the file scale (catalysis.bngl, $ATP = 6.022e8)
    "clamped": dict(reservoir=lambda t: np.full_like(t, 6.022e8)),
    # (b) a nearly constant unclamped one does (moves 1e-8 of its peak)
    "nearly-constant": dict(reservoir=lambda t: 6.022e8 * (1 - 1e-8 * t / 3600.0)),
    # (c) the moving species sit on a baseline 1e3 times what they move by:
    #     they move by 0.1% of their peak, never 5% of it
    "baseline": dict(baseline=6.022e8),
}


@pytest.mark.parametrize("shape", sorted(_MECHANISMS))
def test_a_frozen_engine_fails(shape):
    """An engine that never moves off the initial state. On catalysis.bngl the
    verdict was PASS with max_rel 0.0, 960 of 1210 cells past the per-cell
    tolerance."""
    ref = _trajectory(**_MECHANISMS[shape])
    frozen = np.tile(ref[0], (ref.shape[0], 1))
    ok, v = _passed(frozen, ref)
    assert not ok
    assert v["n_hard_fail"] > 0
    assert v["max_err_over_movement"] >= 0.5
    assert not v["no_signal"]


@pytest.mark.parametrize("shape", sorted(_MECHANISMS))
def test_a_four_percent_error_on_one_species_fails(shape):
    """One significant moving species 4% off throughout: 68% of such cases
    passed the corpus before the movement gate."""
    ref = _trajectory(**_MECHANISMS[shape])
    a = ref.copy()
    a[:, 1] = ref[:, 1] + 0.04 * (ref[:, 1] - ref[0, 1])
    assert not _passed(a, ref)[0]


@pytest.mark.parametrize("shape", sorted(_MECHANISMS))
def test_an_engine_that_agrees_passes(shape):
    """Control. 1e-6 of each column's movement, well inside the census p99
    (4.4e-5), passes in every shape."""
    ref = _trajectory(**_MECHANISMS[shape])
    span = ref.max(axis=0) - ref.min(axis=0)
    a = ref + 1e-6 * span * np.sin(np.arange(ref.shape[0]))[:, np.newaxis]
    ok, v = _passed(a, ref)
    assert ok
    assert v["max_err_over_movement"] == pytest.approx(1e-6, rel=1e-2)


def test_an_error_under_the_movement_ceiling_passes():
    """Control. Every species ×1.009 passed all 582 corpus references, and
    still does: 0.9% of a species that moves its whole peak."""
    t = np.linspace(0.0, 10.0, 101)
    ref = np.column_stack([1.0 - np.exp(-t), np.exp(-t)])
    ok, v = _passed(1.009 * ref, ref)
    assert ok
    assert v["max_err_over_movement"] < differ.MOVE_CEILING


def test_a_reference_that_does_not_move_is_no_signal():
    """Nothing moves (the 5 STATIC corpus models): a frozen engine is right,
    and the verdict says it could not have told otherwise."""
    ref = np.tile([6.022e8, 6.022e7, 6.022e5], (50, 1))
    ok, v = _passed(ref.copy(), ref)
    assert ok
    assert v["no_signal"]
    assert v["max_err_over_movement"] == 0.0


def test_the_movement_gate_is_invariant_under_positive_rescaling():
    """Metamorphic, as for the magnitude gates."""
    ref = _trajectory(**_MECHANISMS["clamped"])
    frozen = np.tile(ref[0], (ref.shape[0], 1))
    for c in (1e-6, 1.0, 1e6):
        assert not _passed(c * frozen, c * ref)[0]


def test_forgive_mask_forgives_a_movement_fail():
    """The step-shift mask still applies to cells the movement gate fails, and
    the #316 audit credits it with them.

    The reservoir at 100 keeps the 0.02 error under the absolute ceiling (1e-2 of
    the file peak) and the column under HARD_REL_CEILING, so the magnitude gates
    forgive the cell and only the movement gate fails it (2% of the range).
    Without the reservoir the cell is an absolute-ceiling breach, and the test
    would pass on main."""
    t = np.linspace(0.0, 10.0, 101)
    ref = np.column_stack([np.full_like(t, 100.0), 1.0 - np.exp(-t)])
    a = ref.copy()
    a[40, 1] += 0.02
    ok, v = _passed(a, ref)
    assert not ok
    assert v["n_hard_fail"] == 1
    mask = np.zeros(ref.shape, dtype=bool)
    mask[40, 1] = True
    ok, v = _passed(a, ref, forgive_mask=mask)
    assert ok
    assert v["passed_without_forgive"] is False
    assert v["n_forgive_rescued"] == 1


def test_an_engine_that_lets_a_clamped_species_drift_fails():
    """The mirror of a frozen engine: the reference holds the reservoir still and
    the engine lets it drift by 0.9% over the run. That is under HARD_REL_CEILING
    and the absolute ceiling, so the magnitude gates forgive it, and a gate that
    looked at the reference's movement alone would never see the column."""
    ref = _trajectory(**_MECHANISMS["clamped"])
    a = ref.copy()
    a[:, 3] *= 1.0 - 0.009 * np.linspace(0.0, 1.0, ref.shape[0])
    ok, v = _passed(a, ref)
    assert not ok
    assert v["max_err_over_movement"] >= 0.5


@pytest.mark.parametrize("shape", sorted(_MECHANISMS))
def test_an_engine_that_makes_a_species_the_reference_never_makes_fails(shape):
    """The reference never makes the last species; the engine makes 5e5 of it,
    under the significance bar of a 6e8 reservoir or baseline."""
    ref = _trajectory(**_MECHANISMS[shape])
    ref = np.column_stack([ref, np.zeros(ref.shape[0])])
    a = ref.copy()
    a[:, -1] = 5e5 * (1.0 - np.exp(-np.linspace(0.0, 12.0, ref.shape[0])))
    ok, v = _passed(a, ref)
    assert not ok
    assert v["max_err_over_movement"] >= 0.5


def test_nothing_moving_on_either_side_is_no_signal_but_still_judged():
    """``no_signal`` says the verdict could not see dynamics, not that it passed:
    two flat trajectories 10% apart still DIFF."""
    ref = np.tile([6.022e8, 6.022e5, 100.0], (50, 1))
    a = ref.copy()
    a[:, 1] *= 1.10
    ok, v = _passed(a, ref)
    assert not ok
    assert v["no_signal"]


@pytest.mark.parametrize("module", ["rr_run", "amici_run"])
def test_the_rr_and_amici_rows_carry_the_movement_numbers(module):
    """Both runners write ``differ.ode_row_comment``, so the amici rows say NO
    SIGNAL too (the matrices' humanize_comment reads it there)."""
    runner = pytest.importorskip(module)
    t = np.linspace(0.0, 10.0, 11)
    names = ["A", "B"]
    flat = np.tile([5.0, 7.0], (t.size, 1))
    comment = runner._compare_ode((t, flat, names), (t, flat.copy(), names))[2]
    assert comment.startswith("2 sp; fail 0/22 (hard 0, soft 0, forgiven 0); err/movement")
    assert "NO SIGNAL" in comment
    moving = np.column_stack([np.exp(-t), 1.0 - np.exp(-t)])
    comment = runner._compare_ode((t, moving, names), (t, moving.copy(), names))[2]
    assert "err/movement 0.0e+00" in comment
    assert "NO SIGNAL" not in comment
