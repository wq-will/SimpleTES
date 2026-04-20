# EVOLVE-BLOCK-START

CPP_CODE = r'''
#include <Eigen/Dense>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>

using Eigen::MatrixXd;
using Eigen::VectorXd;

static inline double soft_thresh(double z, double gamma) {
    if (z >  gamma) return z - gamma;
    if (z < -gamma) return z + gamma;
    return 0.0;
}

// ============================================================
// NAIVE METHOD  (p >= 500)
// Maintains residual r = y - X*beta.
// grad_j = X_j^T r / n  recomputed from r at each coordinate update.
// ============================================================
void solve_naive(
    const MatrixXd& X,          // (n, p) column-major
    const VectorXd& y,
    const VectorXd& lam_path,
    MatrixXd&       coef_path,  // (p, n_lam) output, pre-zeroed
    double thresh,              // convergence threshold (glmnet default 1e-7)
    int    maxit)               // max inner loop iterations
{
    const int n    = X.rows();
    const int p    = X.cols();
    const int nlam = lam_path.size();
    const double fn = static_cast<double>(n);

    const double tol     = thresh;   // absolute convergence criterion

    // Column squared norms (scaled): xv[j] = ||X_j||^2 / n
    VectorXd xv(p);
    for (int j = 0; j < p; ++j)
        xv(j) = X.col(j).squaredNorm() / fn;

    VectorXd beta = VectorXd::Zero(p);
    VectorXd r    = y;              // residual r = y - X*beta (starts at y since beta=0)

    std::vector<bool> ever_active(p, false);  // has beta[j] ever been nonzero?
    std::vector<bool> screened(p, false);     // admitted by strong rule?
    std::vector<int>  active;                 // current active set (ever nonzero)
    active.reserve(p);

    double prev_lam = 0.0;

    for (int li = 0; li < nlam; ++li) {
        const double lam  = lam_path(li);
        const double tlam = 2.0 * lam - prev_lam;  // strong rule threshold

        // ---- Step 1: Strong-rule screening ----
        // Admit feature j if |X_j^T r / n| > 2*lam - prev_lam
        for (int j = 0; j < p; ++j) {
            if (!screened[j] && std::abs(X.col(j).dot(r) / fn) > tlam)
                screened[j] = true;
        }

        // ---- Step 2: Outer loop (JSS Section 2.6) ----
        // Alternate between: full screened pass + active-set inner loop + KKT check.
        // Repeat until KKT passes for all p features.
        int nlp = 0;
        while (true) {
            // 2a. One full cycle of CD over all screened features
            double dmax = 0.0;
            for (int j = 0; j < p; ++j) {
                if (!screened[j]) continue;
                const double bj_old = beta(j);
                const double gj     = X.col(j).dot(r) / fn;
                const double bj_new = soft_thresh(gj + bj_old * xv(j), lam) / xv(j);
                if (bj_new == bj_old) continue;
                const double delta = bj_new - bj_old;
                beta(j) = bj_new;
                r.noalias() -= delta * X.col(j);
                if (!ever_active[j]) {
                    ever_active[j] = true;
                    active.push_back(j);
                }
                const double ch = xv(j) * delta * delta;
                if (ch > dmax) dmax = ch;
            }

            // 2b. Inner loop: CD over active set until convergence
            // (JSS: "iterate on only the active set till convergence")
            while (dmax >= tol && nlp < maxit) {
                ++nlp;
                dmax = 0.0;
                for (int j : active) {
                    const double bj_old = beta(j);
                    const double gj     = X.col(j).dot(r) / fn;
                    const double bj_new = soft_thresh(gj + bj_old * xv(j), lam) / xv(j);
                    if (bj_new == bj_old) continue;
                    const double delta = bj_new - bj_old;
                    beta(j) = bj_new;
                    r.noalias() -= delta * X.col(j);
                    const double ch = xv(j) * delta * delta;
                    if (ch > dmax) dmax = ch;
                }
            }

            // 2c. KKT check on ALL p features (not just unscreened).
            // A feature can be screened (admitted by strong rule) yet have beta=0
            // if soft-threshold zeroed it at a prior lambda. At a smaller lambda,
            // the same feature may satisfy |grad|>lam and need activation.
            // glmnet handles this by re-running the outer screened pass whenever
            // any feature — screened or not — has |grad|>lam and beta=0.
            r = y - X * beta;   // recompute r exactly before KKT
            bool kkt_ok = true;
            for (int j = 0; j < p; ++j) {
                if (std::abs(X.col(j).dot(r) / fn) > lam * (1.0 + 1e-9)) {
                    if (!screened[j]) {
                        screened[j] = true;
                        kkt_ok = false;
                    } else if (!ever_active[j]) {
                        // Screened but never activated: needs another outer pass
                        kkt_ok = false;
                    }
                }
            }
            if (kkt_ok) break;
            if (nlp >= maxit) break;
        }

        coef_path.col(li) = beta;
        prev_lam = lam;
    }
}

// ============================================================
// COVARIANCE METHOD  (p < 500)
// Maintains c[j] = X_j^T r  and  G[i,j] = X_{active[i]}^T X_{active[j]}.
// grad_j = c[idx] / n  (O(1) lookup).
// After beta[j] changes by delta: c -= delta * G[:, idx]  (O(|A|)).
// (JSS Section 2.2: "covariance updating")
// ============================================================
void solve_cov(
    const MatrixXd& X,
    const VectorXd& y,
    const VectorXd& lam_path,
    MatrixXd&       coef_path,
    double thresh,
    int    maxit)
{
    const int n    = X.rows();
    const int p    = X.cols();
    const int nlam = lam_path.size();
    const double fn = static_cast<double>(n);

    const double tol     = thresh;   // absolute convergence criterion

    VectorXd xv(p);
    for (int j = 0; j < p; ++j)
        xv(j) = X.col(j).squaredNorm() / fn;

    VectorXd beta = VectorXd::Zero(p);
    VectorXd r    = y;

    // Active set: indices ever admitted (screened) — covariance method
    // admits all screened features into Gram matrix eagerly.
    std::vector<bool> screened(p, false);
    std::vector<int>  active;      // maps index -> feature column
    active.reserve(p);
    std::vector<int>  feat_to_idx(p, -1);  // inverse map: feature -> index in active

    // Gram matrix G(i,j) = X_{active[i]}^T X_{active[j]}  (grows lazily)
    MatrixXd G;   // resized as active set grows
    // c[idx] = X_{active[idx]}^T r  (unscaled, i.e. NOT divided by n)
    VectorXd c;

    // Expand G and c when new features are added to active
    auto expand = [&](int new_size) {
        if (new_size <= static_cast<int>(G.cols())) return;
        const int old_size = static_cast<int>(G.cols());
        MatrixXd G2 = MatrixXd::Zero(new_size, new_size);
        if (old_size > 0) G2.topLeftCorner(old_size, old_size) = G;
        for (int col = old_size; col < new_size; ++col) {
            const int jc = active[col];
            for (int row = 0; row <= col; ++row) {
                const int jr = active[row];
                const double dot = X.col(jc).dot(X.col(jr));
                G2(row, col) = dot;
                G2(col, row) = dot;
            }
        }
        G = std::move(G2);

        VectorXd c2 = VectorXd::Zero(new_size);
        if (old_size > 0) c2.head(old_size) = c;
        for (int idx = old_size; idx < new_size; ++idx)
            c2(idx) = X.col(active[idx]).dot(r);
        c = std::move(c2);
    };

    double prev_lam = 0.0;

    for (int li = 0; li < nlam; ++li) {
        const double lam  = lam_path(li);
        const double tlam = 2.0 * lam - prev_lam;

        // ---- Step 1: Strong-rule screening, add to active set ----
        // Full gradient to screen (O(np) — same as naive)
        VectorXd grad = X.transpose() * r / fn;
        for (int j = 0; j < p; ++j) {
            if (!screened[j] && std::abs(grad(j)) > tlam) {
                screened[j]    = true;
                feat_to_idx[j] = static_cast<int>(active.size());
                active.push_back(j);
            }
        }
        expand(static_cast<int>(active.size()));

        // ---- Step 2: Outer loop ----
        int nlp = 0;
        while (true) {
            // 2a. One full cycle of CD over active set using Gram cache
            const int a   = static_cast<int>(active.size());
            double    dmax = 0.0;
            for (int idx = 0; idx < a; ++idx) {
                const int    j      = active[idx];
                const double bj_old = beta(j);
                // Gradient from cache: c[idx]/n + bj_old * xv[j]
                const double gj     = c(idx) / fn + bj_old * xv(j);
                const double bj_new = soft_thresh(gj, lam) / xv(j);
                if (bj_new == bj_old) continue;
                const double delta = bj_new - bj_old;
                beta(j) = bj_new;
                // Incremental residual and gradient cache update
                r.noalias() -= delta * X.col(j);
                c.noalias() -= delta * G.col(idx);
                const double ch = xv(j) * delta * delta;
                if (ch > dmax) dmax = ch;
            }

            // 2b. Inner loop on active set until convergence
            while (dmax >= tol && nlp < maxit) {
                ++nlp;
                dmax = 0.0;
                for (int idx = 0; idx < a; ++idx) {
                    const int    j      = active[idx];
                    const double bj_old = beta(j);
                    const double gj     = c(idx) / fn + bj_old * xv(j);
                    const double bj_new = soft_thresh(gj, lam) / xv(j);
                    if (bj_new == bj_old) continue;
                    const double delta = bj_new - bj_old;
                    beta(j) = bj_new;
                    r.noalias() -= delta * X.col(j);
                    c.noalias() -= delta * G.col(idx);
                    const double ch = xv(j) * delta * delta;
                    if (ch > dmax) dmax = ch;
                }
            }

            // 2c. KKT check on ALL p features
            r    = y - X * beta;
            grad = X.transpose() * r / fn;
            bool kkt_ok = true;
            for (int j = 0; j < p; ++j) {
                if (std::abs(grad(j)) > lam * (1.0 + 1e-9)) {
                    if (!screened[j]) {
                        screened[j]    = true;
                        feat_to_idx[j] = static_cast<int>(active.size());
                        active.push_back(j);
                        kkt_ok = false;
                    } else if (feat_to_idx[j] < 0) {
                        // Screened but not yet in active (feat_to_idx=-1): add it
                        feat_to_idx[j] = static_cast<int>(active.size());
                        active.push_back(j);
                        kkt_ok = false;
                    }
                }
            }
            if (kkt_ok) break;

            // Expand Gram / c for newly admitted features, re-sync c from fresh r
            expand(static_cast<int>(active.size()));
            const int a2 = static_cast<int>(active.size());
            for (int idx = 0; idx < a2; ++idx)
                c(idx) = X.col(active[idx]).dot(r);

            if (nlp >= maxit) break;
        }

        coef_path.col(li) = beta;
        prev_lam = lam;
    }
}

// ============================================================
// Main: binary I/O, dispatch to naive or covariance
// ============================================================
int main() {
    int32_t n, p, n_lambda;
    if (fread(&n,        sizeof(int32_t), 1, stdin) != 1) return 1;
    if (fread(&p,        sizeof(int32_t), 1, stdin) != 1) return 1;
    if (fread(&n_lambda, sizeof(int32_t), 1, stdin) != 1) return 1;

    // X arrives row-major (C order from numpy); copy to column-major Eigen matrix
    std::vector<double> buf(static_cast<size_t>(n) * p);
    if (fread(buf.data(), sizeof(double), buf.size(), stdin) != buf.size()) return 1;
    MatrixXd X(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            X(i, j) = buf[static_cast<size_t>(i) * p + j];

    VectorXd y(n);
    if (fread(y.data(), sizeof(double), n, stdin) != static_cast<size_t>(n)) return 1;

    VectorXd lam_path(n_lambda);
    if (fread(lam_path.data(), sizeof(double), n_lambda, stdin)
            != static_cast<size_t>(n_lambda)) return 1;

    MatrixXd coef_path = MatrixXd::Zero(p, n_lambda);

    // glmnet switching rule: covariance when p < 500, naive otherwise
    // (type.gaussian = ifelse(nvars < 500, "covariance", "naive"))
    const double thresh = 1e-9;  // tighter than glmnet's 1e-7 to meet 1e-6 gap on unstandardized data
    const int    maxit  = 100000;  // glmnet default

    if (p < 500)
        solve_cov  (X, y, lam_path, coef_path, thresh, maxit);
    else
        solve_naive(X, y, lam_path, coef_path, thresh, maxit);

    // Output column-major (Eigen default)
    fwrite(coef_path.data(), sizeof(double),
           static_cast<size_t>(p) * n_lambda, stdout);
    return 0;
}
'''

# EVOLVE-BLOCK-END