// bngsim/include/bngsim/dq_jacobian.hpp -- the dense difference quotients
// SUNDIALS forms when it is given no Jacobian, formed by bngsim instead so that
// a column that steps across a jump in a rate law can be mended (issue #1002).
//
// CVODE (cvLsDenseDQJac) and KINSOL (kinLsDenseDQJac) each difference the
// right-hand side one column at a time when no Jacobian callback is installed:
// `bngsim` used to rely on that wherever a model has no closed form or the
// caller asked for `jacobian="fd"`. Both read Δf/h where the step crosses a
// threshold (fd_jacobian.hpp says what that does to a run). `dense_dq_past_jumps`
// is the loop both of them run, with the solver's increment passed in, so a
// column that crosses nothing is the solver's own to the bit: the same
// increment, the same right-hand-side call, `(1/inc)·(f1 − f0)` as
// N_VLinearSum forms it, and on a failed call the same early return that leaves
// the probed component where it stands (the solver then rewinds it).
//
// What it adds is the jump test of fd_jacobian.hpp: one point further on in
// every column and, where a row may step across a jump, two points back. Those
// go through `side_f`, which the caller makes leave no trace a run reads.
#pragma once

#include "bngsim/fd_jacobian.hpp"

#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace bngsim {

// Fills the column-major n×n `jac` with the solver's difference quotient at `y`
// (writable: probed in place and restored, as the solver does), where `f0` is f
// at `y`. The solver's loop runs first, whole and as it is; the jump test runs
// over the matrix it left, which gives each row's terms (fd_jump_floor).
//
//   inc_of(j, y_j)  the solver's increment for column j;
//   solver_f()      f at the current `y` into `f1`, as the solver calls it; its
//                   nonzero return ends the loop and is returned;
//   side_f(out)     f at the current `y` into `out`; false where that is not
//                   usable (a failure, or a value that is not finite).
//
// `f2`, `fm1`, `fm2`, `terms` are n-long scratch rows. `by_motion`: an
// integrator's matrix, mended only in the columns fd_jump_mendable takes.
// Returns the solver's code; `*n_mended` counts the entries mended.
template <class Inc, class SolverF, class SideF>
inline int dense_dq_past_jumps(long n, double *y, const double *f0, const double *f1, double *f2,
                               double *fm1, double *fm2, double *terms, double *jac, bool by_motion,
                               Inc &&inc_of, SolverF &&solver_f, SideF &&side_f, int *n_mended) {
    const auto col_of = [&](long j) {
        return jac + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
    };
    for (long j = 0; j < n; ++j) {
        double *col = col_of(j);
        const double yj = y[j];
        const double inc = inc_of(j, yj);
        y[j] += inc;
        const int retval = solver_f();
        if (retval != 0) {
            return retval;
        }
        y[j] = yj;
        const double inc_inv = 1.0 / inc;
        for (long i = 0; i < n; ++i) {
            col[i] = inc_inv * (f1[i] - f0[i]);
        }
    }

    for (long i = 0; i < n; ++i) {
        terms[i] = std::abs(f0[i]);
    }
    for (long j = 0; j < n; ++j) {
        const double *col = col_of(j);
        for (long i = 0; i < n; ++i) {
            terms[i] += std::abs(col[i] * y[j]);
        }
    }
    for (long j = 0; j < n; ++j) {
        double *col = col_of(j);
        const double yj = y[j];
        const double inc = inc_of(j, yj);
        if (by_motion && !fd_jump_mendable(f0[j], inc)) {
            continue;
        }
        y[j] = yj + 2.0 * inc;
        bool ok = side_f(f2);
        y[j] = yj;
        if (!ok) {
            continue;
        }
        // f(y + h) is f0 + h·col to the rounding of the column formula.
        bool may = false;
        for (long i = 0; i < n && !may; ++i) {
            const double d01 = col[i] * inc;
            may = fd_step_may_cross_jump(d01, f2[i] - f0[i] - d01, fd_jump_floor(terms[i]));
        }
        if (!may) {
            continue;
        }
        y[j] = yj - inc;
        ok = side_f(fm1);
        if (ok) {
            y[j] = yj - 2.0 * inc;
            ok = side_f(fm2);
        }
        y[j] = yj;
        if (!ok) {
            continue;
        }
        for (long i = 0; i < n; ++i) {
            const double d01 = col[i] * inc;
            const double d12 = f2[i] - f0[i] - d01;
            const double dmm = fm1[i] - fm2[i];
            const double floor = fd_jump_floor(terms[i]);
            if (fd_step_crosses_jump(dmm, d01, d12, floor)) {
                col[i] = fd_entry_past_jump(dmm, f0[i] - fm1[i], d12, floor, inc);
                ++*n_mended;
            }
        }
    }
    return 0;
}

// The two adapters below take `solver_f(out)` and `side_f(out)` over an
// N_Vector, and `scratch`, the caller's rows kept between calls (3·n).

// CVODE's dense quotient (cvLsDenseDQJac, SUNDIALS 7.2) at its Jacobian setup:
// increment max(√uround·|y_j|, minInc/ewt_j) with minInc = 1000·|h|·uround·n·
// ‖f‖_WRMS (1 where ‖f‖ = 0). The error weights and the step are CVODE's own
// (CVodeGetErrWeights, CVodeGetCurrentStep: cv_next_h, which CVODE sets at
// every change of cv_h ahead of a setup). bngsim sets no inequality
// constraints, the one part of CVODE's left out. tmp1 holds f1, tmp2 the
// weights, tmp3 the row one step further on.
template <class SolverF, class SideF>
inline int cvode_dense_dq_past_jumps(void *cvode_mem, N_Vector y, N_Vector fy, SUNMatrix J,
                                     N_Vector tmp1, N_Vector tmp2, N_Vector tmp3,
                                     std::vector<double> &scratch, SolverF &&solver_f,
                                     SideF &&side_f, int *n_mended) {
    const sunindextype n = SUNDenseMatrix_Columns(J);
    if (CVodeGetErrWeights(cvode_mem, tmp2) != CV_SUCCESS) {
        return -1;
    }
    sunrealtype h = 0.0;
    if (CVodeGetCurrentStep(cvode_mem, &h) != CV_SUCCESS) {
        return -1;
    }
    const sunrealtype uround = SUN_UNIT_ROUNDOFF;
    const sunrealtype srur = SUNRsqrt(uround);
    const sunrealtype fnorm = N_VWrmsNorm(fy, tmp2);
    const sunrealtype min_inc = (fnorm != 0.0) ? (1000.0 * SUNRabs(h) * uround * n * fnorm) : 1.0;
    const double *ewt = N_VGetArrayPointer(tmp2);
    const auto un = static_cast<std::size_t>(n);
    if (scratch.size() != 3 * un) {
        scratch.assign(3 * un, 0.0);
    }
    N_Vector side = N_VCloneEmpty(tmp1);
    if (side == nullptr) {
        return -1;
    }
    const int rc = dense_dq_past_jumps(
        static_cast<long>(n), N_VGetArrayPointer(y), N_VGetArrayPointer(fy),
        N_VGetArrayPointer(tmp1), N_VGetArrayPointer(tmp3), scratch.data(), scratch.data() + un,
        scratch.data() + 2 * un, SUNDenseMatrix_Data(J), /*by_motion=*/true,
        [&](long j, double yj) {
            const sunrealtype by_self = srur * SUNRabs(yj);
            const sunrealtype by_floor = min_inc / ewt[j];
            return by_self > by_floor ? by_self : by_floor;
        },
        [&] { return solver_f(tmp1); },
        [&](double *out) {
            N_VSetArrayPointer(out, side);
            return side_f(side);
        },
        n_mended);
    N_VSetArrayPointer(nullptr, side);
    N_VDestroy(side);
    return rc;
}

// KINSOL's dense quotient (kinLsDenseDQJac, SUNDIALS 7.2): increment
// √uround·max(|u_j|, 1/uscale_j), signed as u_j, where √uround is KINSOL's
// sqrt_relfunc at its default (bngsim never sets the relative error of F).
// `uscale` is the scaling vector the solve was given. tmp1 holds f1; tmp2's
// data, the row one step further on.
template <class SolverF, class SideF>
inline int kinsol_dense_dq_past_jumps(N_Vector u, N_Vector fu, SUNMatrix J, N_Vector tmp1,
                                      N_Vector tmp2, const double *uscale,
                                      std::vector<double> &scratch, SolverF &&solver_f,
                                      SideF &&side_f, int *n_mended) {
    const sunindextype n = SUNDenseMatrix_Columns(J);
    const sunrealtype sqrt_relfunc = SUNRsqrt(SUN_UNIT_ROUNDOFF);
    const auto un = static_cast<std::size_t>(n);
    if (scratch.size() != 3 * un) {
        scratch.assign(3 * un, 0.0);
    }
    N_Vector side = N_VCloneEmpty(tmp1);
    if (side == nullptr) {
        return -1;
    }
    const int rc = dense_dq_past_jumps(
        static_cast<long>(n), N_VGetArrayPointer(u), N_VGetArrayPointer(fu),
        N_VGetArrayPointer(tmp1), N_VGetArrayPointer(tmp2), scratch.data(), scratch.data() + un,
        scratch.data() + 2 * un, SUNDenseMatrix_Data(J), /*by_motion=*/false,
        [&](long j, double uj) {
            const sunrealtype ujscale = 1.0 / uscale[j];
            const sunrealtype sign = (uj >= 0.0) ? 1.0 : -1.0;
            const sunrealtype mag = SUNRabs(uj);
            return sqrt_relfunc * (mag > ujscale ? mag : ujscale) * sign;
        },
        [&] { return solver_f(tmp1); },
        [&](double *out) {
            N_VSetArrayPointer(out, side);
            return side_f(side);
        },
        n_mended);
    N_VSetArrayPointer(nullptr, side);
    N_VDestroy(side);
    return rc;
}

} // namespace bngsim
