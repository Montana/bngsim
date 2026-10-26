// bngsim/include/bngsim/dq_jacobian.hpp -- the dense difference quotient CVODE
// forms when it is given no Jacobian, formed by bngsim instead so that a row
// whose step crosses a jump in a rate law can be mended (issue #1002).
//
// With no Jacobian callback, CVODE differences the right-hand side one column at
// a time (cvLsDenseDQJac): every dense run and steady-state march with
// `jacobian="fd"`, or with one law the closed form declines, ran on that. It
// reads Δf/h where the step crosses a threshold; fd_jacobian.hpp says what that
// did to a run. `cvode_dense_dq_past_jumps` is CVODE's loop first, whole and as
// it is, so a matrix that crosses nothing is CVODE's to the bit: the same
// increment, the same right-hand-side call, `(1/inc)·(f1 − f0)` as N_VLinearSum
// forms it, and on a failed call the same early return that leaves the probed
// component where it stands (CVODE then rewinds it). The jump test of
// fd_jacobian.hpp runs after it, over the matrix it left.
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

// The dense J at CVODE's Jacobian setup, for a callback handed (y, fy, J, tmp1,
// tmp2, tmp3). CVODE's increment is max(√uround·|y_j|, minInc/ewt_j), with
// minInc = 1000·|h|·uround·n·‖fy‖_WRMS (1 where that norm is 0), from CVODE's
// own error weights and step (CVodeGetErrWeights; CVodeGetCurrentStep returns
// cv_next_h, which CVODE sets at every change of cv_h ahead of a setup). bngsim
// sets no inequality constraints, the one part of CVODE's left out.
//
//   solver_f(out)  f at `y` into `out`, as CVODE calls it; its nonzero return
//                  ends the loop and is returned;
//   side_f(out)    f at `y` into `out` for the jump test, leaving nothing a run
//                  reads afterwards; false where that is not usable (a failure,
//                  or a value that is not finite).
//
// tmp1 holds f(y + h·e_j), tmp2 the weights and tmp3 f(y + 2h·e_j); `scratch`
// (3·n, kept between calls) the two backward rows and each row's terms.
// `*n_mended` counts the entries mended.
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
    auto inc_of = [&](sunindextype j, double yj) {
        const sunrealtype by_self = srur * SUNRabs(yj);
        const sunrealtype by_floor = min_inc / ewt[j];
        return by_self > by_floor ? by_self : by_floor;
    };

    double *yd = N_VGetArrayPointer(y);
    const double *f0 = N_VGetArrayPointer(fy);
    const double *f1 = N_VGetArrayPointer(tmp1);
    for (sunindextype j = 0; j < n; ++j) {
        double *col = SUNDenseMatrix_Column(J, j);
        const double yj = yd[j];
        const double inc = inc_of(j, yj);
        yd[j] += inc;
        const int retval = solver_f(tmp1);
        if (retval != 0) {
            return retval;
        }
        yd[j] = yj;
        const double inc_inv = 1.0 / inc;
        for (sunindextype i = 0; i < n; ++i) {
            col[i] = inc_inv * (f1[i] - f0[i]);
        }
    }

    const auto un = static_cast<std::size_t>(n);
    if (scratch.size() != 3 * un) {
        scratch.assign(3 * un, 0.0);
    }
    double *f2 = N_VGetArrayPointer(tmp3);
    double *fm1 = scratch.data();
    double *fm2 = scratch.data() + un;
    double *terms = scratch.data() + 2 * un;
    for (sunindextype i = 0; i < n; ++i) {
        terms[i] = std::abs(f0[i]);
    }
    for (sunindextype j = 0; j < n; ++j) {
        const double *col = SUNDenseMatrix_Column(J, j);
        for (sunindextype i = 0; i < n; ++i) {
            terms[i] += std::abs(col[i] * yd[j]);
        }
    }
    N_Vector side = N_VCloneEmpty(tmp1);
    if (side == nullptr) {
        return -1;
    }
    auto side_at = [&](sunindextype j, double yj, double m, double *out) {
        yd[j] = yj + m * inc_of(j, yj);
        N_VSetArrayPointer(out, side);
        const bool ok = side_f(side);
        yd[j] = yj;
        return ok;
    };
    for (sunindextype j = 0; j < n; ++j) {
        double *col = SUNDenseMatrix_Column(J, j);
        const double yj = yd[j];
        const double inc = inc_of(j, yj);
        if (!fd_jump_mendable(f0[j], inc) || !side_at(j, yj, 2.0, f2)) {
            continue;
        }
        // f(y + h) is f0 + h·col to the rounding of the column formula.
        bool may = false;
        for (sunindextype i = 0; i < n && !may; ++i) {
            const double d01 = col[i] * inc;
            may = fd_step_may_cross_jump(d01, f2[i] - f0[i] - d01, fd_jump_floor(terms[i]));
        }
        if (!may || !side_at(j, yj, -1.0, fm1) || !side_at(j, yj, -2.0, fm2)) {
            continue;
        }
        for (sunindextype i = 0; i < n; ++i) {
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
    N_VSetArrayPointer(nullptr, side);
    N_VDestroy(side);
    return 0;
}

} // namespace bngsim
