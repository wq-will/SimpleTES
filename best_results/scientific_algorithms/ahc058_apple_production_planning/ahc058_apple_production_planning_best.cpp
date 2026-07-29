#include <bits/stdc++.h>
using namespace std;

#pragma GCC optimize("Ofast")

/*--------------------------------------------------------------
  Fixed problem parameters (as given in the statement)
--------------------------------------------------------------*/
constexpr int MAXN = 10;               // number of IDs (fixed)
constexpr int MAXL = 4;                // number of levels (fixed)
constexpr int MAXT = 500;              // total turns (fixed)
constexpr int MAXSZ = MAXN * MAXL;     // total machine types (40)

/*--------------------------------------------------------------
  Search parameters – tuned to fit into the 2‑second limit
--------------------------------------------------------------*/
constexpr double TOTAL_TIME = 1.99;    // seconds allocated for the whole run
constexpr int RESTARTS = 14;           // number of independent restarts
constexpr double TEMP_SCALE = 6e8;     // annealing temperature factor
constexpr int SHIFT_MAX = 300;         // max shift distance in mutations
constexpr int HEAD_TURNS = 45;         // look‑ahead depth for the first schedule

/*--------------------------------------------------------------
  Global data
--------------------------------------------------------------*/
int N, L, T;
long long K_input;
double K0;                             // initial apples (as double)
double A[MAXN];                        // production capacities of level‑0 machines
double C_flat[MAXSZ];                  // flattened cost matrix C[i][j]
int SZ;                                // N * L (should be 40)

/*--------------------------------------------------------------
  State of the system at a certain turn
--------------------------------------------------------------*/
struct State {
    double apples;
    double B[MAXSZ];   // number of machines of each type
    double P[MAXSZ];   // power (upgrade count) of each type
};

/*--------------------------------------------------------------
  Compute per‑unit‑power contribution of each machine type.
  val[i][j] = apples per turn contributed by one additional unit of
  power of machine j^i under the current state.
--------------------------------------------------------------*/
inline void compute_val(const State& st, double val[MAXL][MAXN]) {
    for (int j = 0; j < N; ++j) {
        val[0][j] = A[j] * st.P[j];
    }
    for (int i = 1; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            val[i][j] = st.P[idx] * val[i-1][j];
        }
    }
}

/*--------------------------------------------------------------
  Greedy estimator: best affordable strengthening action for the
  current state (or -1 for doing nothing). Benefit is optimistic,
  based on remaining turns.
--------------------------------------------------------------*/
inline int compute_best_action(const State& st, int remaining) {
    double val[MAXL][MAXN];
    compute_val(st, val);
    const double EPS = 1e-12;
    double best_net = -1e300;
    int best_idx = -1;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double cost = C_flat[idx] * (st.P[idx] + 1.0);
            if (cost > st.apples + EPS) continue;      // cannot afford

            double benefit = 0.0;
            if (i == 0) {
                benefit = A[j] * st.B[idx] * remaining;
            } else {
                int eff = remaining - i;                 // delay before apples appear
                if (eff <= 0) continue;
                benefit = st.B[idx] * val[i-1][j] * eff;
            }
            double net = benefit - cost;
            if (net > best_net + EPS) {
                best_net = net;
                best_idx = idx;
            }
        }
    }
    /* If even the best net gain is negative, do nothing */
    if (best_net <= 0) return -1;
    return best_idx;
}

/*--------------------------------------------------------------
  Full simulation of a schedule; also stores the state after each
  turn (required for fast suffix recomputation). Returns final
  apple count or a negative value if the schedule is illegal.
--------------------------------------------------------------*/
inline double simulate_full(const int act[MAXT], State states[MAXT + 1]) {
    State cur;
    cur.apples = K0;
    for (int i = 0; i < SZ; ++i) {
        cur.B[i] = 1.0;
        cur.P[i] = 0.0;
    }
    states[0] = cur;
    for (int turn = 0; turn < T; ++turn) {
        int a = act[turn];
        if (a != -1) {
            double cost = C_flat[a] * (cur.P[a] + 1.0);
            if (cost > cur.apples + 1e-12) return -1.0;   // illegal
            cur.apples -= cost;
            cur.P[a] += 1.0;
        }
        double inc = 0.0;
        for (int j = 0; j < N; ++j) inc += A[j] * cur.B[j] * cur.P[j];
        cur.apples += inc;
        for (int i = 1; i < L; ++i) {
            int base = i * N;
            int lower = (i - 1) * N;
            for (int j = 0; j < N; ++j) {
                cur.B[lower + j] += cur.B[base + j] * cur.P[base + j];
            }
        }
        states[turn + 1] = cur;
    }
    return cur.apples;
}

/*--------------------------------------------------------------
  Simulate only the suffix of a schedule starting from a given
  state (used when only a part of the plan changed).
--------------------------------------------------------------*/
inline double simulate_from_index(int start, const int act[MAXT],
                                 const State& start_state) {
    State cur = start_state;
    for (int turn = start; turn < T; ++turn) {
        int a = act[turn];
        if (a != -1) {
            double cost = C_flat[a] * (cur.P[a] + 1.0);
            if (cost > cur.apples + 1e-12) return -1.0;   // illegal
            cur.apples -= cost;
            cur.P[a] += 1.0;
        }
        double inc = 0.0;
        for (int j = 0; j < N; ++j) inc += A[j] * cur.B[j] * cur.P[j];
        cur.apples += inc;
        for (int i = 1; i < L; ++i) {
            int base = i * N;
            int lower = (i - 1) * N;
            for (int j = 0; j < N; ++j) {
                cur.B[lower + j] += cur.B[base + j] * cur.P[base + j];
            }
        }
    }
    return cur.apples;
}

/*--------------------------------------------------------------
  Re‑compute the suffix of the state array after a modification.
--------------------------------------------------------------*/
inline void simulate_and_update(int start, const int act[MAXT],
                               State states[MAXT + 1]) {
    State cur = states[start];
    for (int turn = start; turn < T; ++turn) {
        int a = act[turn];
        if (a != -1) {
            cur.apples -= C_flat[a] * (cur.P[a] + 1.0);
            cur.P[a] += 1.0;
        }
        double inc = 0.0;
        for (int j = 0; j < N; ++j) inc += A[j] * cur.B[j] * cur.P[j];
        cur.apples += inc;
        for (int i = 1; i < L; ++i) {
            int base = i * N;
            int lower = (i - 1) * N;
            for (int j = 0; j < N; ++j) {
                cur.B[lower + j] += cur.B[base + j] * cur.P[base + j];
            }
        }
        states[turn + 1] = cur;
    }
}

/*--------------------------------------------------------------
  Generate an initial schedule using a short look‑ahead for the
  first few turns, then fall back to plain greedy.
--------------------------------------------------------------*/
void generate_initial_lookahead(int act[MAXT], bool random_flag, mt19937_64& rng) {
    double B[MAXSZ];
    double P[MAXSZ];
    for (int i = 0; i < SZ; ++i) {
        B[i] = 1.0;
        P[i] = 0.0;
    }
    double apples = K0;
    uniform_real_distribution<double> prob(0.0, 1.0);
    for (int turn = 0; turn < T; ++turn) {
        int remaining = T - turn;
        vector<int> affordable;
        affordable.reserve(SZ + 1);
        affordable.push_back(-1);
        for (int idx = 0; idx < SZ; ++idx) {
            double cost = C_flat[idx] * (P[idx] + 1.0);
            if (cost <= apples + 1e-12) affordable.push_back(idx);
        }

        int chosen = -1;
        if (turn < HEAD_TURNS) {
            double best_final = -1e300;
            for (int a : affordable) {
                double B2[MAXSZ];
                double P2[MAXSZ];
                memcpy(B2, B, sizeof(double) * SZ);
                memcpy(P2, P, sizeof(double) * SZ);
                double apples2 = apples;

                if (a != -1) {
                    double cost = C_flat[a] * (P2[a] + 1.0);
                    apples2 -= cost;
                    P2[a] += 1.0;
                }

                // production for this turn
                double inc = 0.0;
                for (int j = 0; j < N; ++j) inc += A[j] * B2[j] * P2[j];
                apples2 += inc;
                for (int i = 1; i < L; ++i) {
                    int base = i * N;
                    int lower = (i - 1) * N;
                    for (int j = 0; j < N; ++j) {
                        B2[lower + j] += B2[base + j] * P2[base + j];
                    }
                }

                // greedy for the rest
                int rem = remaining - 1;
                for (int step = 0; step < rem; ++step) {
                    State tmp;
                    tmp.apples = apples2;
                    for (int i = 0; i < SZ; ++i) {
                        tmp.B[i] = B2[i];
                        tmp.P[i] = P2[i];
                    }
                    int best = compute_best_action(tmp, rem - step);
                    if (best != -1) {
                        double cost = C_flat[best] * (P2[best] + 1.0);
                        apples2 -= cost;
                        P2[best] += 1.0;
                    }
                    double inc2 = 0.0;
                    for (int j = 0; j < N; ++j) inc2 += A[j] * B2[j] * P2[j];
                    apples2 += inc2;
                    for (int i = 1; i < L; ++i) {
                        int base = i * N;
                        int lower = (i - 1) * N;
                        for (int j = 0; j < N; ++j) {
                            B2[lower + j] += B2[base + j] * P2[base + j];
                        }
                    }
                }
                if (apples2 > best_final) {
                    best_final = apples2;
                    chosen = a;
                }
            }
            if (random_flag && prob(rng) < 0.07) {
                chosen = affordable[rng() % affordable.size()];
            }
        } else {
            // plain greedy
            State cur_state;
            cur_state.apples = apples;
            for (int i = 0; i < SZ; ++i) {
                cur_state.B[i] = B[i];
                cur_state.P[i] = P[i];
            }
            chosen = compute_best_action(cur_state, remaining);
            if (random_flag && prob(rng) < 0.30) {
                vector<int> cand;
                cand.reserve(SZ + 1);
                for (int idx = 0; idx < SZ; ++idx) {
                    double cost = C_flat[idx] * (P[idx] + 1.0);
                    if (cost <= apples + 1e-12) cand.push_back(idx);
                }
                cand.push_back(-1);
                chosen = cand[rng() % cand.size()];
            }
        }

        act[turn] = chosen;
        if (chosen != -1) {
            double cost = C_flat[chosen] * (P[chosen] + 1.0);
            apples -= cost;
            P[chosen] += 1.0;
        }

        // production for this turn
        double inc = 0.0;
        for (int j = 0; j < N; ++j) inc += A[j] * B[j] * P[j];
        apples += inc;
        for (int i = 1; i < L; ++i) {
            int base = i * N;
            int lower = (i - 1) * N;
            for (int j = 0; j < N; ++j) {
                B[lower + j] += B[base + j] * P[base + j];
            }
        }
    }
}

/*--------------------------------------------------------------
  Generate an initial schedule using plain greedy (with optional
  randomisation).
--------------------------------------------------------------*/
void generate_initial_basic(int act[MAXT], bool random_flag, mt19937_64& rng) {
    double B_loc[MAXSZ];
    double P_loc[MAXSZ];
    for (int i = 0; i < SZ; ++i) {
        B_loc[i] = 1.0;
        P_loc[i] = 0.0;
    }
    double apples_loc = K0;
    uniform_real_distribution<double> prob(0.0, 1.0);
    for (int turn = 0; turn < T; ++turn) {
        int remaining = T - turn;
        State cur;
        cur.apples = apples_loc;
        for (int i = 0; i < SZ; ++i) {
            cur.B[i] = B_loc[i];
            cur.P[i] = P_loc[i];
        }
        int best = compute_best_action(cur, remaining);
        int chosen = best;
        if (random_flag && prob(rng) < 0.30) {
            vector<int> cand;
            cand.reserve(SZ + 1);
            for (int idx = 0; idx < SZ; ++idx) {
                double cost = C_flat[idx] * (cur.P[idx] + 1.0);
                if (cost <= apples_loc + 1e-12) cand.push_back(idx);
            }
            cand.push_back(-1);
            chosen = cand[rng() % cand.size()];
        }
        act[turn] = chosen;
        if (chosen != -1) {
            double cost = C_flat[chosen] * (P_loc[chosen] + 1.0);
            apples_loc -= cost;
            P_loc[chosen] += 1.0;
        }
        double inc = 0.0;
        for (int j = 0; j < N; ++j) inc += A[j] * B_loc[j] * P_loc[j];
        apples_loc += inc;
        for (int i = 1; i < L; ++i) {
            int base = i * N;
            int lower = (i - 1) * N;
            for (int j = 0; j < N; ++j) {
                B_loc[lower + j] += B_loc[base + j] * P_loc[base + j];
            }
        }
    }
}

/*--------------------------------------------------------------
  Main workflow
--------------------------------------------------------------*/
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> N >> L >> T >> K_input)) return 0;
    K0 = static_cast<double>(K_input);
    for (int j = 0; j < N; ++j) {
        long long a; cin >> a;
        A[j] = static_cast<double>(a);
    }
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            long long c; cin >> c;
            C_flat[i * N + j] = static_cast<double>(c);
        }
    }
    SZ = N * L;  // should be 40

    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

    uniform_int_distribution<int> dist_turn(0, T - 1);
    uniform_int_distribution<int> dist_move(0, 5);
    uniform_int_distribution<int> dist_shift(-SHIFT_MAX, SHIFT_MAX);
    uniform_real_distribution<double> prob01(0.0, 1.0);

    static int global_best_act[MAXT];
    for (int i = 0; i < T; ++i) global_best_act[i] = -1;
    double global_best_score = -1e300;

    /* baseline: all do‑nothing */
    static State dummy_states[MAXT + 1];
    global_best_score = simulate_full(global_best_act, dummy_states);

    auto total_start = chrono::steady_clock::now();

    /*=================== First schedule with look‑ahead ===================*/
    static int cur_act[MAXT];
    generate_initial_lookahead(cur_act, false, rng);
    static State cur_states[MAXT + 1];
    double cur_score = simulate_full(cur_act, cur_states);
    if (cur_score > global_best_score) {
        global_best_score = cur_score;
        memcpy(global_best_act, cur_act, sizeof(int) * T);
    }

    /*=================== Independent restarts ===================*/
    for (int restart = 1; restart < RESTARTS; ++restart) {
        double elapsed_total = chrono::duration<double>(chrono::steady_clock::now() - total_start).count();
        if (elapsed_total >= TOTAL_TIME) break;

        static int cur_act2[MAXT];
        static int best_act[MAXT];
        static int cand_act[MAXT];
        static State cur_states2[MAXT + 1];

        bool random_flag = true;
        generate_initial_basic(cur_act2, random_flag, rng);
        memcpy(best_act, cur_act2, sizeof(int) * T);
        double cur_score2 = simulate_full(cur_act2, cur_states2);
        double best_score = cur_score2;

        double restart_budget = TOTAL_TIME / RESTARTS;
        auto restart_start = chrono::steady_clock::now();

        while (true) {
            double elapsed = chrono::duration<double>(chrono::steady_clock::now() - total_start).count();
            if (elapsed >= TOTAL_TIME) break;
            double elapsed_restart = chrono::duration<double>(chrono::steady_clock::now() - restart_start).count();
            if (elapsed_restart >= restart_budget) break;

            memcpy(cand_act, cur_act2, sizeof(int) * T);
            int start_idx = 0;
            int mv = dist_move(rng);

            if (mv == 0) { // random affordable replace one turn
                int t = dist_turn(rng);
                const State& st = cur_states2[t];
                vector<int> poss;
                poss.reserve(SZ + 1);
                for (int idx = 0; idx < SZ; ++idx) {
                    double cost = C_flat[idx] * (st.P[idx] + 1.0);
                    if (cost <= st.apples + 1e-12) poss.push_back(idx);
                }
                poss.push_back(-1);
                cand_act[t] = poss[rng() % poss.size()];
                start_idx = t;
            } else if (mv == 1) { // best affordable replace one turn
                int t = dist_turn(rng);
                const State& st = cur_states2[t];
                cand_act[t] = compute_best_action(st, T - t);
                start_idx = t;
            } else if (mv == 2) { // set a turn to do‑nothing
                int t = dist_turn(rng);
                cand_act[t] = -1;
                start_idx = t;
            } else if (mv == 3) { // swap two turns
                int t1 = dist_turn(rng);
                int t2 = dist_turn(rng);
                if (t1 == t2) continue;
                swap(cand_act[t1], cand_act[t2]);
                start_idx = min(t1, t2);
            } else if (mv == 4) { // shift one action
                int src = dist_turn(rng);
                int delta = dist_shift(rng);
                int dst = src + delta;
                if (dst < 0) dst = 0;
                if (dst >= T) dst = T - 1;
                if (src == dst) continue;
                int val = cand_act[src];
                if (src < dst) {
                    for (int k = src; k < dst; ++k) cand_act[k] = cand_act[k + 1];
                    cand_act[dst] = val;
                } else {
                    for (int k = src; k > dst; --k) cand_act[k] = cand_act[k - 1];
                    cand_act[dst] = val;
                }
                start_idx = min(src, dst);
            } else { // block move (length up to 3)
                int len = rng() % 3 + 1;
                int start = dist_turn(rng);
                if (start + len > T) len = T - start;
                int dst = dist_turn(rng);
                if (dst >= start && dst < start + len) continue; // overlap – skip
                int block[3];
                for (int i = 0; i < len; ++i) block[i] = cand_act[start + i];
                if (dst < start) {
                    for (int i = start + len - 1; i >= start; --i) {
                        cand_act[i] = cand_act[i - len];
                    }
                    for (int i = 0; i < len; ++i) cand_act[dst + i] = block[i];
                } else {
                    for (int i = start; i < dst - len + 1; ++i) {
                        cand_act[i] = cand_act[i + len];
                    }
                    int insert = dst - len + 1;
                    for (int i = 0; i < len; ++i) cand_act[insert + i] = block[i];
                }
                start_idx = min(start, dst);
            }

            // ensure mutation really changed something
            bool changed = false;
            for (int i = 0; i < T; ++i) {
                if (cand_act[i] != cur_act2[i]) { changed = true; break; }
            }
            if (!changed) continue;

            double cand_score = simulate_from_index(start_idx, cand_act, cur_states2[start_idx]);
            if (cand_score < 0) continue; // illegal candidate

            if (cand_score > cur_score2 + 1e-12) {
                memcpy(cur_act2, cand_act, sizeof(int) * T);
                cur_score2 = cand_score;
                simulate_and_update(start_idx, cur_act2, cur_states2);
                if (cur_score2 > best_score + 1e-12) {
                    best_score = cur_score2;
                    memcpy(best_act, cur_act2, sizeof(int) * T);
                }
            } else {
                double remaining_time = TOTAL_TIME - elapsed;
                double temp = TEMP_SCALE * (remaining_time / TOTAL_TIME) + 1e-12;
                double prob = exp((cand_score - cur_score2) / temp);
                if (prob > prob01(rng)) {
                    memcpy(cur_act2, cand_act, sizeof(int) * T);
                    cur_score2 = cand_score;
                    simulate_and_update(start_idx, cur_act2, cur_states2);
                }
            }
        } // end inner while

        if (best_score > global_best_score + 1e-12) {
            global_best_score = best_score;
            memcpy(global_best_act, best_act, sizeof(int) * T);
        }
    } // end restarts

    /*=================== Polishing phase ===================*/
    static State final_states[MAXT + 1];
    double final_score = simulate_full(global_best_act, final_states);
    while (true) {
        double elapsed = chrono::duration<double>(chrono::steady_clock::now() - total_start).count();
        if (elapsed >= TOTAL_TIME) break;
        bool improved = false;

        // per‑turn greedy replacement
        for (int t = 0; t < T; ++t) {
            int best = compute_best_action(final_states[t], T - t);
            if (best != -1 && best != global_best_act[t]) {
                static int cand_seq[MAXT];
                memcpy(cand_seq, global_best_act, sizeof(int) * T);
                cand_seq[t] = best;
                double val = simulate_from_index(t, cand_seq, final_states[t]);
                if (val > final_score + 1e-12) {
                    memcpy(global_best_act, cand_seq, sizeof(int) * T);
                    final_score = val;
                    simulate_and_update(t, global_best_act, final_states);
                    improved = true;
                }
            }
        }

        // adjacent swap improvement
        for (int t = 0; t < T - 1; ++t) {
            if (global_best_act[t] == global_best_act[t + 1]) continue;
            static int cand_seq2[MAXT];
            memcpy(cand_seq2, global_best_act, sizeof(int) * T);
            swap(cand_seq2[t], cand_seq2[t + 1]);
            double val = simulate_from_index(t, cand_seq2, final_states[t]);
            if (val > final_score + 1e-12) {
                memcpy(global_best_act, cand_seq2, sizeof(int) * T);
                final_score = val;
                simulate_and_update(t, global_best_act, final_states);
                improved = true;
            }
        }

        if (!improved) break;
    }

    /*=================== Final random exploration (extra time) ====================*/
    while (true) {
        double elapsed = chrono::duration<double>(chrono::steady_clock::now() - total_start).count();
        if (elapsed >= TOTAL_TIME) break;
        static int cand_seq[MAXT];
        memcpy(cand_seq, global_best_act, sizeof(int) * T);
        int mv = dist_move(rng);
        int start_idx = 0;
        if (mv == 0) {
            int t = dist_turn(rng);
            const State& st = final_states[t];
            vector<int> poss;
            poss.reserve(SZ + 1);
            for (int idx = 0; idx < SZ; ++idx) {
                double cost = C_flat[idx] * (st.P[idx] + 1.0);
                if (cost <= st.apples + 1e-12) poss.push_back(idx);
            }
            poss.push_back(-1);
            cand_seq[t] = poss[rng() % poss.size()];
            start_idx = t;
        } else if (mv == 1) {
            int t = dist_turn(rng);
            const State& st = final_states[t];
            cand_seq[t] = compute_best_action(st, T - t);
            start_idx = t;
        } else if (mv == 2) {
            int t = dist_turn(rng);
            cand_seq[t] = -1;
            start_idx = t;
        } else if (mv == 3) {
            int t1 = dist_turn(rng);
            int t2 = dist_turn(rng);
            if (t1 == t2) continue;
            swap(cand_seq[t1], cand_seq[t2]);
            start_idx = min(t1, t2);
        } else if (mv == 4) {
            int src = dist_turn(rng);
            int delta = dist_shift(rng);
            int dst = src + delta;
            if (dst < 0) dst = 0;
            if (dst >= T) dst = T - 1;
            if (src == dst) continue;
            int val = cand_seq[src];
            if (src < dst) {
                for (int k = src; k < dst; ++k) cand_seq[k] = cand_seq[k + 1];
                cand_seq[dst] = val;
            } else {
                for (int k = src; k > dst; --k) cand_seq[k] = cand_seq[k - 1];
                cand_seq[dst] = val;
            }
            start_idx = min(src, dst);
        } else {
            int len = rng() % 3 + 1;
            int start = dist_turn(rng);
            if (start + len > T) len = T - start;
            int dst = dist_turn(rng);
            if (dst >= start && dst < start + len) continue;
            int block[3];
            for (int i = 0; i < len; ++i) block[i] = cand_seq[start + i];
            if (dst < start) {
                for (int i = start + len - 1; i >= start; --i) {
                    cand_seq[i] = cand_seq[i - len];
                }
                for (int i = 0; i < len; ++i) cand_seq[dst + i] = block[i];
            } else {
                for (int i = start; i < dst - len + 1; ++i) {
                    cand_seq[i] = cand_seq[i + len];
                }
                int insert = dst - len + 1;
                for (int i = 0; i < len; ++i) cand_seq[insert + i] = block[i];
            }
            start_idx = min(start, dst);
        }

        // ensure mutation really changed something
        bool changed = false;
        for (int i = 0; i < T; ++i) {
            if (cand_seq[i] != global_best_act[i]) { changed = true; break; }
        }
        if (!changed) continue;

        double cand_score = simulate_from_index(start_idx, cand_seq, final_states[start_idx]);
        if (cand_score < 0) continue;
        if (cand_score > final_score + 1e-12) {
            memcpy(global_best_act, cand_seq, sizeof(int) * T);
            final_score = cand_score;
            simulate_and_update(start_idx, global_best_act, final_states);
        }
    }

    /*=================== Output ====================*/
    for (int turn = 0; turn < T; ++turn) {
        if (global_best_act[turn] == -1) {
            cout << "-1\n";
        } else {
            int lvl = global_best_act[turn] / N;
            int id  = global_best_act[turn] % N;
            cout << lvl << " " << id << "\n";
        }
    }
    return 0;
}
