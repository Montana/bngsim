// bngsim/include/bngsim/sparse_jacobian.hpp — the sparse (KLU) linear-solver
// route, shared by the time-course solver and the steady-state march (#128)
//
// Everything a CVODE session needs in order to factor its Newton matrix with
// KLU instead of a dense LU, in one place because there are now two sessions
// that do it: CvodeSimulator::run() and SteadyStateMarcher. Before issue #128
// the steady-state march always factored densely — ss_make_dense_linsol's own
// comment said so — while run() on the same model routed to KLU, which on the
// 1000+ species corpus models is 2.0-3.9x of wall clock.
//
// Three things live here, and they are here rather than in one of the two .cpp
// files for the same reason: a routing rule, a matrix layout and a difference
// formula that two solvers must agree on are exactly the things that drift when
// each keeps its own copy.
//
//   * route_to_sparse_linear_solver — the dense/sparse decision itself.
//   * install_csc_structure — reinstating the CSC pattern SUNMatZero_Sparse
//     clears, which every sparse Jacobian callback must do before it fills.
//   * colored_fd_jacobian — the Curtis-Powell-Reid difference quotient, for a
//     sparse-routed model with no closed-form Jacobian (KLU has no built-in
//     difference quotient to fall back on: CVODE's covers dense and banded
//     matrices only).

#pragma once

#include "bngsim/lapack_dense_linsol.hpp"
#include "bngsim/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace bngsim {

// ── The dense/sparse decision ────────────────────────────────────────────────
//
// Species count threshold for auto-selecting the sparse solver: KLU when
// N >= SPARSE_THRESHOLD AND density < SPARSE_DENSITY_MAX. Many rule-based
// models have relatively dense Jacobians (chromatic number ≈ N, density often
// 10-30%), making KLU's sparse factorization slower than a dense LU. Only
// models with truly sparse Jacobians (metapopulation, compartmental transport)
// tend to benefit. The density cutoff reflects internal benchmarking.
//
// The choice of dense backend once this says "dense" (built-in dense LU vs the
// GH #84 BLAS dgetrf solver) is a separate question, answered by
// should_use_lapack_dense() in bngsim/lapack_dense_linsol.hpp — except for a
// model the LU-fill test below moved off KLU, which always takes the BLAS
// factor (make_fill_routed_dense_linear_solver).
inline constexpr int SPARSE_THRESHOLD = 50;
inline constexpr double SPARSE_DENSITY_MAX = 0.10; // 10%

// ── LU fill: a sparse Jacobian whose factor is not ───────────────────────────
//
// SPARSE_DENSITY_MAX tests the Jacobian, but KLU factors I - gamma*J, and what
// it costs is the density of the *factor*. On the rule-derived networks of
// benchmarks/_dev/suite_ode.json with 354-3,744 species the Jacobian is 1-7%
// dense while its LU fills to at least 20-35% of n^2 (hub species — a free
// enzyme, an adaptor, a ligand — couple to most complexes). KLU is left-looking with no
// BLAS, so on those models it spends >90% of a run in klu_l_refactor, and the
// BLAS dense factor (GH #84) is 3.6-10x faster end to end; a genuinely sparse
// network (metapop_sir_100, fill 2%) keeps KLU 6x ahead. See
// benchmarks/_dev/bench_linear_solver_fill.py for the sweep.
//
// So a model the size/density test sends to KLU goes dense instead when all of:
//   * a BLAS dense factor is linked (the built-in unblocked LU is not reliably
//     faster than KLU even at this fill — multisite_phos 2.4x slower);
//   * LU_FILL_DENSE_MIN_N <= n <= LU_FILL_DENSE_MAX_N (below, KLU and the dense
//     factor tie: SHP2_base_model, 149 species, 26% fill; above, two dense n^2
//     copies stop being a reasonable amount of memory — 400 MB at the cap);
//   * the analytical Jacobian is complete and requested (auto / analytical):
//     the dense route's only other Jacobian is CVODE's difference quotient, n
//     RHS evaluations, where the sparse route colors its differences;
//   * KLU's symbolic analysis estimates fill >= LU_FILL_DENSE_MIN (an AMD
//     estimate, low on these nets — compute_lu_fill_estimate, model_builder.cpp).
// Neither force flag is overridden: force_sparse_linear_solver keeps KLU, and
// is the escape hatch for a model this rule misjudges.
inline constexpr double LU_FILL_DENSE_MIN = 0.10;
inline constexpr int LU_FILL_DENSE_MIN_N = 256;
inline constexpr int LU_FILL_DENSE_MAX_N = 5000;

// Can the fill test change this model's route? True only for a model the size
// and density test would send to KLU, in the size window above, on a build with
// both KLU and a BLAS dense factor. Callers use it to skip the fill estimate —
// a KLU symbolic analysis — for every model it cannot affect.
inline bool lu_fill_matters(const JacobianSparsity &sp, int ns) {
#ifdef BNGSIM_HAS_KLU
    return lapack_dense_available() && !sp.empty() && ns >= SPARSE_THRESHOLD &&
           sp.density < SPARSE_DENSITY_MAX && ns >= LU_FILL_DENSE_MIN_N &&
           ns <= LU_FILL_DENSE_MAX_N;
#else
    (void)sp;
    (void)ns;
    return false;
#endif
}

// The fill estimate a routing decision needs, or -1 when the fill test does not
// apply: outside lu_fill_matters, or without a complete analytical Jacobian the
// requested strategy will use. `Model` is NetworkModel (a template only so this
// header need not include model.hpp); estimated_lu_fill() is computed once per
// model and shared across clones.
template <class Model>
double routing_lu_fill(const Model &model, int ns, const std::string &jacobian_strategy) {
    const auto &sp = model.jacobian_sparsity();
    if (!lu_fill_matters(sp, ns))
        return -1.0;
    if (!(jacobian_strategy == "auto" || jacobian_strategy == "analytical") ||
        !model.analytical_jacobian_complete())
        return -1.0;
    return model.estimated_lu_fill();
}

// Does the fill test send this model to the dense BLAS factor instead of KLU?
// `lu_fill` is routing_lu_fill()'s answer; -1 (not applicable) never does.
inline bool fill_routes_to_dense(const JacobianSparsity &sp, int ns, double lu_fill,
                                 const std::string &jacobian_strategy, bool force_dense,
                                 bool force_sparse) {
    return !force_dense && !force_sparse && jacobian_strategy != "jax" &&
           lu_fill >= LU_FILL_DENSE_MIN && lu_fill_matters(sp, ns);
}

// Should this model's Newton matrix be a CSC SUNSparseMatrix factored by KLU?
//
// Use sparse KLU when: (1) KLU is available, (2) the model is large enough,
// (3) a sparsity pattern exists, and (4) the density is low enough to benefit.
// A Jacobian denser than the cutoff is effectively dense and KLU's overhead
// makes it slower than a dense LU — which also guards models where Functional
// rate laws make the sparsity pattern nearly dense. JAX Jacobians always fill a
// dense matrix, so that strategy forces dense mode.
//
// The two force flags (GH #102, GH #29) straddle the size/density test but not
// the hard requirements around it: KLU needs a real sparsity pattern to build
// its CSC matrix, and a JAX Jacobian only ever fills a dense one, so sparse_ok
// gates both overrides. force_dense wins the (rejected upstream) both-set case
// only as a belt-and-braces default.
//
// `jacobian_strategy` is SolverOptions::jacobian / SteadyStateOptions::jacobian.
// "jax" forces dense; "fd" stays sparse and takes the colored difference
// quotient below, exactly as it does on the time-course path (routing_lu_fill
// returns -1 for it, so the fill test never moves an "fd" model).
//
// `lu_fill` is routing_lu_fill(model, ns, jacobian_strategy): a model the size
// and density test sends to KLU goes dense when fill_routes_to_dense says so.
inline bool route_to_sparse_linear_solver(const JacobianSparsity &sp, int ns,
                                          const std::string &jacobian_strategy, bool force_dense,
                                          bool force_sparse, double lu_fill) {
#ifdef BNGSIM_HAS_KLU
    const bool sparse_ok = (jacobian_strategy != "jax") && !sp.empty();
    return sparse_ok && !force_dense &&
           (force_sparse ||
            ((ns >= SPARSE_THRESHOLD) && (sp.density < SPARSE_DENSITY_MAX) &&
             !fill_routes_to_dense(sp, ns, lu_fill, jacobian_strategy, force_dense, force_sparse)));
#else
    (void)sp;
    (void)ns;
    (void)jacobian_strategy;
    (void)force_dense;
    (void)force_sparse;
    (void)lu_fill;
    return false;
#endif
}

// ── CSC structure ────────────────────────────────────────────────────────────
//
// Copy the model's CSC pattern into a SUNSparseMatrix's index arrays. Every
// sparse Jacobian callback must call this before filling values: CVODE may call
// SUNMatZero() ahead of the callback, and SUNMatZero_Sparse clears BOTH the
// values and the structural indices. Templated on the index type so this header
// needs no SUNDIALS include; pass SUNSparseMatrix_IndexPointers(J) and
// SUNSparseMatrix_IndexValues(J).
template <typename IndexT>
inline void install_csc_structure(IndexT *col_ptrs, IndexT *row_indices,
                                  const JacobianSparsity &sp) {
    for (int j = 0; j <= sp.n; ++j) {
        col_ptrs[j] = static_cast<IndexT>(sp.col_ptrs[j]);
    }
    for (int k = 0; k < sp.nnz; ++k) {
        row_indices[k] = static_cast<IndexT>(sp.row_indices[k]);
    }
}

// ── Colored finite-difference Jacobian (Curtis-Powell-Reid) ──────────────────
//
// Computes J = ∂f/∂y into the CSC value array. Columns that share a color have
// non-overlapping sparsity patterns, so they can be perturbed simultaneously in
// a single RHS evaluation. This reduces the cost from O(N) RHS evals (one per
// column) to O(n_colors) ≈ 5-20, which is the key speedup for large sparse
// models.
//
// For each color c:
//   1. Perturb ALL columns j with color[j] == c simultaneously:
//      y_pert = y + Σ_j h_j * e_j
//   2. Evaluate f(y_pert) — one single RHS call.
//   3. Extract individual column contributions: for each column j in color c,
//      for each nonzero row i in column j:
//        J[i][j] = (f_pert[i] - f[i]) / h_j
//      This works because no other column j' in the same color has a nonzero
//      in row i (that's the coloring guarantee).
//
// Reference: Curtis, Powell, Reid (1974) "On the estimation of sparse
// Jacobian matrices", J. Inst. Math. Appl. 13, 117–119.
//
// Precondition: `sp` carries a coloring (NetworkModel::ensure_jacobian_coloring,
// not the bare accessor), `fy` is f at the unperturbed `y`, and the three
// scratch buffers are ns long — the caller owns them so a per-step Jacobian
// allocates nothing. `eval_rhs(t, y_pert, fy_pert)` evaluates the SAME right-
// hand side the session is integrating.
template <typename EvalRhs>
inline void colored_fd_jacobian(const JacobianSparsity &sp, int ns, double t, const double *y,
                                const double *fy, double *jac_data, double *y_pert, double *fy_pert,
                                double *h_vals, EvalRhs &&eval_rhs) {
    // Finite difference perturbation scale
    const double sqrt_uround = 1.4901161193847656e-8; // sqrt(machine epsilon)

    // Iterate over colors (one RHS eval per color)
    for (int c = 0; c < sp.n_colors; ++c) {
        const auto &group = sp.color_groups[c];

        // 1. Build perturbed state: y_pert = y + Σ_{j in group} h_j * e_j
        std::memcpy(y_pert, y, static_cast<std::size_t>(ns) * sizeof(double));

        for (int j : group) {
            double h = sqrt_uround * std::max(std::abs(y[j]), 1.0);
            h_vals[j] = h;
            y_pert[j] += h;
        }

        // 2. Single RHS evaluation for all columns in this color group
        eval_rhs(t, y_pert, fy_pert);

        // 3. Extract Jacobian entries for each column in the group
        for (int j : group) {
            double inv_h = 1.0 / h_vals[j];
            int64_t col_start = sp.col_ptrs[j];
            int64_t col_end = sp.col_ptrs[j + 1];

            for (int64_t k = col_start; k < col_end; ++k) {
                int i = static_cast<int>(sp.row_indices[k]);
                jac_data[k] = (fy_pert[i] - fy[i]) * inv_h;
            }
        }
    }
}

} // namespace bngsim
