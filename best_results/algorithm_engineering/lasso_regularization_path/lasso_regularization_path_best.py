# EVOLVE-BLOCK-START

CPP_CODE = r'''
#include <Eigen/Dense>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>
#include <cstring>
#ifdef _OPENMP
    #include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::Index;

// =============================================================
// CD solver
// =============================================================
static void solve_cd(const MatrixXd &X,   // (n x p) column-major
                     const VectorXd &y,
                     const VectorXd &lambda_path,
                     int n, int p, int n_lambda) {
    const double inv_n = 1.0 / static_cast<double>(n);

    auto soft_thresh = [](double z, double gamma) -> double {
        if (z >  gamma) return z - gamma;
        if (z < -gamma) return z + gamma;
        return 0.0;
    };

    // column squared norms scaled by 1/n
    VectorXd xv(p);
    for (int j = 0; j < p; ++j) {
        double sq = X.col(j).squaredNorm() * inv_n;
        xv(j) = (sq == 0.0) ? 1e-14 : sq;
    }

    double lambda_max = 0.0;
    for (int i = 0; i < n_lambda; ++i)
        lambda_max = std::max(lambda_max, lambda_path(i));

    VectorXd beta = VectorXd::Zero(p);
    VectorXd r    = y;
    VectorXd grad = X.transpose() * r;
    grad.array() *= inv_n;

    std::vector<char> screened(p, 0);
    std::vector<char> ever_active(p, 0);
    std::vector<int>  active;
    active.reserve(p);

    const double EPS_KKT  = 1e-9;
    const double TOL_BASE = 1e-9;
    const int    MAX_IT   = 100000;

    double prev_lambda = lambda_max;  // first strong rule: 2*lambda_0 - lambda_max

    for (int li = 0; li < n_lambda; ++li) {
        const double lambda     = lambda_path(li);
        const double strong_thr = 2.0 * lambda - prev_lambda;
        const double tol        = std::max(TOL_BASE, TOL_BASE * lambda / lambda_max);

        // strong rule screening
        for (int j = 0; j < p; ++j)
            if (!screened[j] && std::fabs(grad(j)) > strong_thr)
                screened[j] = 1;

        while (true) {
            // outer sweep over screened features
            double dmax = 0.0;
            for (int j = 0; j < p; ++j) {
                if (!screened[j]) continue;
                const double bj_old = beta(j);
                const double gj     = X.col(j).dot(r) * inv_n;
                const double bj_new = soft_thresh(gj + bj_old * xv(j), lambda) / xv(j);
                if (bj_new == bj_old) continue;
                const double delta  = bj_new - bj_old;
                beta(j) = bj_new;
                r.noalias() -= delta * X.col(j);
                if (!ever_active[j]) { ever_active[j] = 1; active.push_back(j); }
                dmax = std::max(dmax, xv(j) * delta * delta);
            }

            // inner loop on ever-active set
            int nlp = 0;
            while (dmax >= tol && nlp < MAX_IT) {
                ++nlp; dmax = 0.0;
                for (int j : active) {
                    const double bj_old = beta(j);
                    const double gj     = X.col(j).dot(r) * inv_n;
                    const double bj_new = soft_thresh(gj + bj_old * xv(j), lambda) / xv(j);
                    if (bj_new == bj_old) continue;
                    const double delta  = bj_new - bj_old;
                    beta(j) = bj_new;
                    r.noalias() -= delta * X.col(j);
                    dmax = std::max(dmax, xv(j) * delta * delta);
                }
            }

            // KKT check
            grad.noalias() = X.transpose() * r;
            grad.array() *= inv_n;
            bool any_violation = false;
            for (int j = 0; j < p; ++j) {
                if (std::fabs(grad(j)) > lambda * (1.0 + EPS_KKT)) {
                    if (!screened[j]) { screened[j] = 1; any_violation = true; }
                    else if (!ever_active[j]) any_violation = true;
                }
            }
            if (!any_violation) break;
        }

        // write solution
        fwrite(beta.data(), sizeof(double), static_cast<size_t>(p), stdout);
        prev_lambda = lambda;
    }
}

// =============================================================
// LARS solver
// =============================================================
static void solve_lars(int n, int p, int n_lambda,
                       MatrixXd &X_T,          // (p x n) scaled by 1/sqrt(n)
                       VectorXd &y,             // scaled by 1/sqrt(n)
                       const VectorXd &lambda_path) {
    const double eps = 1e-12;
    const double INF = std::numeric_limits<double>::infinity();

    VectorXd col_norm2 = X_T.rowwise().squaredNorm();
    VectorXd c = X_T * y;
    double lambda_max = c.cwiseAbs().maxCoeff();

    size_t target_idx = 0;
    while (target_idx < static_cast<size_t>(n_lambda) &&
           lambda_path(target_idx) >= lambda_max - eps) {
        ++target_idx;
    }
    if (target_idx > 0) {
        std::vector<double> zero_buf(p, 0.0);
        for (size_t i = 0; i < target_idx; ++i)
            fwrite(zero_buf.data(), sizeof(double), static_cast<size_t>(p), stdout);
    }
    if (target_idx == static_cast<size_t>(n_lambda)) return;

    const int max_active = std::min(n, p);
    const size_t GRAM_ENTRY_LIMIT = 250000000ULL;
    const bool use_gram = static_cast<size_t>(p) * static_cast<size_t>(max_active) <= GRAM_ENTRY_LIMIT;

    MatrixXd M_active;
    if (use_gram) M_active.resize(p, max_active);

    MatrixXd G_inv(max_active, max_active);
    G_inv.setZero();

    std::vector<char> is_active(p, 0);
    std::vector<int>  active;
    std::vector<double> sign_active;
    active.reserve(max_active);
    sign_active.reserve(max_active);

    VectorXd beta    = VectorXd::Zero(p);
    VectorXd a_vec(p);
    VectorXd d(max_active);
    VectorXd s_vec(max_active);
    VectorXd tmp_u;
    if (!use_gram) tmp_u = VectorXd::Zero(n);

    std::vector<double> out_buf(p);

    auto add_variable = [&](int var_idx, double sgn) {
        int a = static_cast<int>(active.size());
        if (use_gram)
            M_active.col(a).noalias() = X_T * X_T.row(var_idx).transpose();
        Eigen::VectorXd v(a);
        for (int i = 0; i < a; ++i)
            v(i) = use_gram ? M_active(active[i], a)
                            : X_T.row(active[i]).dot(X_T.row(var_idx));
        double g_kk = col_norm2(var_idx);
        if (a == 0) {
            G_inv(0, 0) = 1.0 / g_kk;
        } else {
            Eigen::VectorXd w = G_inv.topLeftCorner(a, a) * v;
            double s = g_kk - v.dot(w);
            if (s <= eps) s = eps;
            G_inv.topLeftCorner(a, a).noalias() += (w * w.transpose()) / s;
            G_inv.block(0, a, a, 1).noalias() = -w / s;
            G_inv.block(a, 0, 1, a).noalias() = (-w / s).transpose();
            G_inv(a, a) = 1.0 / s;
        }
        active.push_back(var_idx);
        sign_active.push_back(sgn);
        is_active[var_idx] = 1;
    };

    auto drop_variable = [&](int drop_pos) {
        int a = static_cast<int>(active.size());
        int var_idx = active[drop_pos];
        beta(var_idx) = 0.0;
        is_active[var_idx] = 0;
        int last = a - 1;
        if (drop_pos != last) {
            std::swap(active[drop_pos], active[last]);
            std::swap(sign_active[drop_pos], sign_active[last]);
            G_inv.row(drop_pos).head(a).swap(G_inv.row(last).head(a));
            G_inv.col(drop_pos).head(a).swap(G_inv.col(last).head(a));
            if (use_gram) M_active.col(drop_pos).swap(M_active.col(last));
        }
        active.pop_back();
        sign_active.pop_back();
        int new_a = a - 1;
        if (new_a > 0) {
            double d_val = G_inv(new_a, new_a);
            Eigen::VectorXd b = G_inv.topRows(new_a).col(new_a);
            G_inv.topLeftCorner(new_a, new_a).noalias() -= (b * b.transpose()) / d_val;
        }
    };

    double lambda = lambda_max;

    while (target_idx < static_cast<size_t>(n_lambda)) {
        double lambda_target = lambda_path[target_idx];

        while (lambda > lambda_target + eps) {
            if (active.empty()) {
                Index j_max;
                c.cwiseAbs().maxCoeff(&j_max);
                double sgn = (c(j_max) >= 0.0) ? 1.0 : -1.0;
                add_variable(static_cast<int>(j_max), sgn);
                continue;
            }

            const int a = static_cast<int>(active.size());

            for (int i = 0; i < a; ++i) s_vec(i) = sign_active[i];
            d.head(a).noalias() = G_inv.topLeftCorner(a, a) * s_vec.head(a);

            if (use_gram) {
                a_vec.noalias() = M_active.leftCols(a) * d.head(a);
            } else {
                tmp_u.setZero();
                for (int i = 0; i < a; ++i)
                    tmp_u.noalias() += d(i) * X_T.row(active[i]).transpose();
                a_vec.noalias() = X_T * tmp_u;
            }

            // gamma_out
            double gamma_out = INF;
            int drop_pos = -1;
            for (int i = 0; i < a; ++i) {
                double di = d(i);
                if (std::abs(di) > eps) {
                    double gamma_i = -beta(active[i]) / di;
                    if (gamma_i > eps && gamma_i < gamma_out) {
                        gamma_out = gamma_i; drop_pos = i;
                    }
                }
            }

            // gamma_in (sequential — evaluator sets OMP_NUM_THREADS=1)
            double gamma_in = INF;
            int enter_var = -1;
            for (int j = 0; j < p; ++j) {
                if (is_active[j]) continue;
                double a_j = a_vec(j), c_j = c(j);
                double d1 = 1.0 - a_j, g1 = (d1 > eps) ? (lambda - c_j) / d1 : INF;
                double d2 = 1.0 + a_j, g2 = (d2 > eps) ? (lambda + c_j) / d2 : INF;
                double gj = (g1 < g2) ? g1 : g2;
                if (gj > eps && gj < gamma_in) { gamma_in = gj; enter_var = j; }
            }

            double gamma_target = lambda - lambda_target;
            double gamma = gamma_target;
            bool event_in = false, event_out = false;
            if (gamma_in  < gamma) { gamma = gamma_in;  event_in  = true; }
            if (gamma_out < gamma) { gamma = gamma_out; event_out = true; }
            if (gamma > lambda) gamma = lambda;

            double lambda_old = lambda;

            for (int i = 0; i < a; ++i) beta(active[i]) += gamma * d(i);
            c.noalias() -= gamma * a_vec;
            lambda -= gamma;
            if (lambda < 0.0) lambda = 0.0;

            // interpolate for grid points in this segment
            while (target_idx < static_cast<size_t>(n_lambda) &&
                   lambda_path[target_idx] <= lambda_old + eps &&
                   lambda_path[target_idx] >= lambda - eps) {
                double lam_t = lambda_path[target_idx];
                double factor = (gamma > 0.0) ? (lam_t - lambda) / gamma : 0.0;
                std::memcpy(out_buf.data(), beta.data(), static_cast<size_t>(p) * sizeof(double));
                for (int i = 0; i < a; ++i)
                    out_buf[active[i]] -= factor * gamma * d(i);
                fwrite(out_buf.data(), sizeof(double), static_cast<size_t>(p), stdout);
                ++target_idx;
            }

            if (lambda <= eps) {
                VectorXd beta_ols = X_T.transpose().colPivHouseholderQr().solve(y);
                std::memcpy(out_buf.data(), beta_ols.data(), static_cast<size_t>(p) * sizeof(double));
                while (target_idx < static_cast<size_t>(n_lambda)) {
                    fwrite(out_buf.data(), sizeof(double), static_cast<size_t>(p), stdout);
                    ++target_idx;
                }
                return;
            }

            if (event_in && (!event_out || gamma_in <= gamma_out + eps) && enter_var >= 0)
                add_variable(enter_var, (c(enter_var) >= 0.0) ? 1.0 : -1.0);
            else if (event_out && drop_pos >= 0)
                drop_variable(drop_pos);
        }
    }
}

// =============================================================
// Main
// =============================================================
int main() {
    int32_t n_i, p_i, nl_i;
    if (fread(&n_i, sizeof(int32_t), 1, stdin) != 1) return 1;
    if (fread(&p_i, sizeof(int32_t), 1, stdin) != 1) return 1;
    if (fread(&nl_i, sizeof(int32_t), 1, stdin) != 1) return 1;
    const int n = n_i, p = p_i, n_lambda = nl_i;

    // Read X row-major
    std::vector<double> buf_X(static_cast<size_t>(n) * static_cast<size_t>(p));
    if (fread(buf_X.data(), sizeof(double), buf_X.size(), stdin) != buf_X.size()) return 1;

    // Read y
    VectorXd y_raw(n);
    if (fread(y_raw.data(), sizeof(double), static_cast<size_t>(n), stdin)
            != static_cast<size_t>(n)) return 1;

    // Read lambda path
    VectorXd lambda_path(n_lambda);
    if (fread(lambda_path.data(), sizeof(double), static_cast<size_t>(n_lambda), stdin)
            != static_cast<size_t>(n_lambda)) return 1;

    // Switching rule:
    // LARS: p <= 2000 and n >= p/4 (not too wide, active set manageable)
    // CD:   everything else (wide, sparse-style, or very large p)
    const bool use_lars = (p <= 2000) && (n >= p / 4);

    if (use_lars) {
        // Build X_T (p x n) scaled by 1/sqrt(n)
        const double scale = 1.0 / std::sqrt(static_cast<double>(n));
        // Map buf_X directly as X^T (p x n) — reinterprets row-major (n x p)
        // as column-major (p x n), which is exactly X^T. Same trick as original.
        Map<MatrixXd> X_T_raw(buf_X.data(), p, n);
        MatrixXd X_T = X_T_raw * scale;
        VectorXd y   = y_raw * scale;
        // lambda_path is in original scale (1/2n loss).
        // After scaling X,y by 1/sqrt(n): c = X_T*y = X^T y/n, same units as lambda.
        // No rescaling needed.
        solve_lars(n, p, n_lambda, X_T, y, lambda_path);
    } else {
        // Map buf_X as column-major (n x p) for CD.
        // buf_X is row-major (n x p), so we need to transpose it properly.
        // Map as (p x n) then transpose to get true (n x p) column-major.
        Map<MatrixXd> X_T_raw(buf_X.data(), p, n);
        MatrixXd X = X_T_raw.transpose();  // (n x p) column-major
        solve_cd(X, y_raw, lambda_path, n, p, n_lambda);
    }

    return 0;
}
'''

COMPILE_FLAGS = ["-fopenmp"]
# EVOLVE-BLOCK-END