#include <bits/stdc++.h>
using namespace std;

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------
const int MAX_COORD = 100000;
const int MAX_VERT   = 1000;
const int MAX_PERIM  = 400000;
const double SAFETY = 0.05;                 // safety margin for time limit
double TIME_LIMIT = 2.0;                    // can be overridden by a CLI argument

// ------------------------------------------------------------
// Fast Xor‑Shift RNG
// ------------------------------------------------------------
struct XorShift {
    uint64_t x;
    XorShift() {
        x = chrono::steady_clock::now().time_since_epoch().count()
            ^ ((uint64_t)random_device()() << 32) ^ random_device()();
    }
    uint64_t next() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return x;
    }
    // [0, n)
    int next_int(int n) {                 // returns 0 if n<=0
        return n <= 0 ? 0 : int(next() % (uint64_t)n);
    }
    // [a, b]
    int next_int(int a, int b) {          // a<=b expected
        if (a > b) return a;
        return a + next_int(b - a + 1);
    }
    double next_double() {
        return next() / (double)UINT64_MAX;
    }
} rng;

// ------------------------------------------------------------
// Timer
// ------------------------------------------------------------
struct Timer {
    chrono::steady_clock::time_point start;
    Timer() { reset(); }
    void reset() { start = chrono::steady_clock::now(); }
    double elapsed() const {
        return chrono::duration_cast<chrono::duration<double>>
               (chrono::steady_clock::now() - start).count();
    }
} timer;

// ------------------------------------------------------------
// Geometry utilities
// ------------------------------------------------------------
struct Point {
    int x, y;
    bool operator==(const Point& o) const { return x == o.x && y == o.y; }
    bool operator<(const Point& o) const {
        return x < o.x || (x == o.x && y < o.y);
    }
};
struct PointHash {
    size_t operator()(const Point& p) const noexcept {
        return ((size_t)p.x << 20) ^ (size_t)p.y;
    }
};

struct Fish {
    Point p;
    int type;                 // +1 = mackerel, -1 = sardine
};
vector<Fish> fish;

// ------------------------------------------------------------
// Compression + 2‑D prefix sum of net difference (mackerel – sardine)
// ------------------------------------------------------------
vector<int> xs, ys;                // sorted unique coordinates (including borders)
int NX = 0, NY = 0;                // compressed sizes
vector<short> prefDiff;            // (NY+1)*(NX+1), row‑major
vector<int> next_ge_x, prev_le_x;
vector<int> next_ge_y, prev_le_y;

inline int IDX(int cx, int cy) {   // row‑major index, 0‑based
    return cy * (NX + 1) + cx;
}

// inclusive rectangle query (original coordinates), returns net diff
inline int query_diff(int x1, int x2, int y1, int y2) {
    if (x1 > x2 || y1 > y2) return 0;
    int xi1 = next_ge_x[x1];
    int xi2 = prev_le_x[x2];
    int yi1 = next_ge_y[y1];
    int yi2 = prev_le_y[y2];
    if (xi1 > xi2 || yi1 > yi2) return 0;
    ++xi1; ++yi1; ++xi2; ++yi2;                 // to 1‑based prefix indices
    int a = prefDiff[IDX(xi2, yi2)];
    int b = prefDiff[IDX(xi1 - 1, yi2)];
    int c = prefDiff[IDX(xi2, yi1 - 1)];
    int d = prefDiff[IDX(xi1 - 1, yi1 - 1)];
    return a - b - c + d;
}

// ------------------------------------------------------------
// Polygon helpers
// ------------------------------------------------------------
inline long long perimeter(const vector<Point>& p) {
    if (p.size() < 2) return 0;
    long long per = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const Point& a = p[i];
        const Point& b = p[(i + 1) % p.size()];
        per += llabs((long long)a.x - b.x) + llabs((long long)a.y - b.y);
    }
    return per;
}

// axis‑aligned segment intersection test
inline bool seg_inter(const Point& a1, const Point& a2,
                      const Point& b1, const Point& b2) {
    Point A1 = a1, A2 = a2, B1 = b1, B2 = b2;
    if (A1.x == A2.x) { if (A1.y > A2.y) swap(A1.y, A2.y); }
    else { if (A1.x > A2.x) swap(A1.x, A2.x); }
    if (B1.x == B2.x) { if (B1.y > B2.y) swap(B1.y, B2.y); }
    else { if (B1.x > B2.x) swap(B1.x, B2.x); }

    bool ah = (A1.y == A2.y);
    bool bh = (B1.y == B2.y);
    if (ah == bh) {
        if (ah) return A1.y == B1.y && max(A1.x, B1.x) <= min(A2.x, B2.x);
        else   return A1.x == B1.x && max(A1.y, B1.y) <= min(A2.y, B2.y);
    } else {
        Point H1 = ah ? A1 : B1, H2 = ah ? A2 : B2;
        Point V1 = ah ? B1 : A1, V2 = ah ? B2 : A2;
        return V1.x >= H1.x && V1.x <= H2.x && H1.y >= V1.y && H1.y <= V2.y;
    }
}

// full self‑intersection test (only used for final safety)
bool self_inter_full(const vector<Point>& poly) {
    int m = (int)poly.size();
    if (m < 4) return false;
    for (int i = 0; i < m; ++i) {
        Point a1 = poly[i], a2 = poly[(i + 1) % m];
        for (int j = i + 2; j < m; ++j) {
            if (i == 0 && j == m - 1) continue;
            Point b1 = poly[j], b2 = poly[(j + 1) % m];
            if (seg_inter(a1, a2, b1, b2)) return true;
        }
    }
    return false;
}

// local intersection test for a set of edges (indices of edges)
bool self_inter_local(const vector<Point>& poly, const vector<int>& crit) {
    int m = (int)poly.size();
    if (m < 4) return false;
    vector<int> ids = crit;
    sort(ids.begin(), ids.end());
    ids.erase(unique(ids.begin(), ids.end()), ids.end());

    for (int e1 : ids) {
        int i1 = (e1 % m + m) % m;
        const Point& a1 = poly[i1];
        const Point& a2 = poly[(i1 + 1) % m];
        for (int e2 = 0; e2 < m; ++e2) {
            bool adj = (e2 == i1) ||
                       (e2 == (i1 + 1) % m) ||
                       ((e2 + 1) % m == i1);
            if (adj) continue;
            const Point& b1 = poly[e2];
            const Point& b2 = poly[(e2 + 1) % m];
            if (seg_inter(a1, a2, b1, b2)) return true;
        }
    }
    return false;
}

// distinct vertices test (used only for final validation)
bool distinct_vertices(const vector<Point>& poly) {
    unordered_set<Point, PointHash> seen;
    seen.reserve(poly.size() * 2);
    for (const auto& p : poly) if (!seen.insert(p).second) return false;
    return true;
}

// basic structural validation (perimeter + axis alignment)
bool poly_valid(const vector<Point>& poly) {
    int m = (int)poly.size();
    if (m != 0 && (m < 4 || m > MAX_VERT)) return false;
    if (m == 0) return true;
    if (perimeter(poly) > MAX_PERIM) return false;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Point& a = poly[i];
        const Point& b = poly[(i + 1) % poly.size()];
        if (a.x < 0 || a.x > MAX_COORD || a.y < 0 || a.y > MAX_COORD) return false;
        if (a.x != b.x && a.y != b.y) return false;      // must be orthogonal
        if (a.x == b.x && a.y == b.y) return false;      // zero‑length edge
    }
    return true;
}

// ------------------------------------------------------------
// Signed area (orientation) helper
// ------------------------------------------------------------
inline long long signed_area2(const vector<Point>& poly) {
    if (poly.size() < 3) return 0;
    long long a = 0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Point& p = poly[i];
        const Point& q = poly[(i + 1) % poly.size()];
        a += (long long)(p.x - q.x) * (p.y + q.y);
    }
    return a;   // >0 => CCW, <0 => CW
}

// ------------------------------------------------------------
// Guide lists (static from fish, dynamic from current best)
// ------------------------------------------------------------
vector<int> guide_static_x, guide_static_y;
vector<int> guide_best_x,   guide_best_y;

void update_best_guides(const vector<Point>& poly) {
    guide_best_x.clear(); guide_best_y.clear();
    unordered_set<int> sx, sy;
    for (const auto& p : poly) {
        sx.insert(p.x);
        sy.insert(p.y);
    }
    guide_best_x.assign(sx.begin(), sx.end());
    guide_best_y.assign(sy.begin(), sy.end());
}

// ------------------------------------------------------------
// Grid‑based coarse search for a good starting rectangle
// ------------------------------------------------------------
int grid_best_x1, grid_best_y1, grid_best_x2, grid_best_y2;
void compute_grid_best_rectangle() {
    const int G = 200;                     // grid dimension
    int cellSize = (MAX_COORD + G) / G;    // ceil division (~501)
    vector<vector<int>> grid(G, vector<int>(G, 0));
    for (const auto& f : fish) {
        int xi = f.p.x / cellSize;
        if (xi >= G) xi = G - 1;
        int yi = f.p.y / cellSize;
        if (yi >= G) yi = G - 1;
        grid[yi][xi] += f.type;
    }

    int bestScore = INT_MIN;
    int bestL = 0, bestR = -1, bestB = 0, bestT = -1;
    vector<int> rows(G);
    for (int left = 0; left < G; ++left) {
        fill(rows.begin(), rows.end(), 0);
        for (int right = left; right < G; ++right) {
            for (int y = 0; y < G; ++y) rows[y] += grid[y][right];
            // Kadane on rows
            int cur = 0, curStart = 0;
            for (int y = 0; y < G; ++y) {
                cur += rows[y];
                if (cur > bestScore) {
                    bestScore = cur;
                    bestL = left; bestR = right;
                    bestB = curStart; bestT = y;
                }
                if (cur < 0) {
                    cur = 0;
                    curStart = y + 1;
                }
            }
        }
    }

    // fallback rectangle if no positive gain
    if (bestScore <= 0) {
        grid_best_x1 = 0; grid_best_y1 = 0;
        grid_best_x2 = 1; grid_best_y2 = 1;
        return;
    }

    int x1 = bestL * cellSize;
    int x2 = min((bestR + 1) * cellSize - 1, MAX_COORD);
    int y1 = bestB * cellSize;
    int y2 = min((bestT + 1) * cellSize - 1, MAX_COORD);
    if (x1 == x2) {
        if (x2 < MAX_COORD) ++x2;
        else if (x1 > 0) --x1;
    }
    if (y1 == y2) {
        if (y2 < MAX_COORD) ++y2;
        else if (y1 > 0) --y1;
    }
    grid_best_x1 = x1; grid_best_y1 = y1;
    grid_best_x2 = x2; grid_best_y2 = y2;

    // Random sampling to possibly improve the rectangle
    const int RAND_SAMPLES = 20000;
    for (int i = 0; i < RAND_SAMPLES; ++i) {
        int xi1 = rng.next_int(0, (int)xs.size() - 1);
        int xi2 = rng.next_int(xi1, (int)xs.size() - 1);
        int yi1 = rng.next_int(0, (int)ys.size() - 1);
        int yi2 = rng.next_int(yi1, (int)ys.size() - 1);
        int x1s = xs[xi1], x2s = xs[xi2];
        int y1s = ys[yi1], y2s = ys[yi2];
        if (x1s == x2s || y1s == y2s) continue;
        int diff = query_diff(x1s, x2s, y1s, y2s);
        if (diff > bestScore) {
            bestScore = diff;
            grid_best_x1 = x1s; grid_best_y1 = y1s;
            grid_best_x2 = x2s; grid_best_y2 = y2s;
        }
    }
}

// ------------------------------------------------------------
// Initial rectangle (uses result of grid+random search)
// ------------------------------------------------------------
vector<Point> initial_rectangle() {
    return {{grid_best_x1, grid_best_y1},
            {grid_best_x2, grid_best_y1},
            {grid_best_x2, grid_best_y2},
            {grid_best_x1, grid_best_y2}};
}

// ------------------------------------------------------------
// Greedy local improvement (expansion and contraction)
// ------------------------------------------------------------
void greedy_expand(vector<Point>& poly, int& curDiff) {
    bool improved = true;
    int iter = 0;
    const int MAX_ITER = 3000;
    long long signedArea = signed_area2(poly);
    while (improved && iter++ < MAX_ITER) {
        improved = false;
        int m = (int)poly.size();
        for (int e = 0; e < m; ++e) {
            Point& A = poly[e];
            Point& B = poly[(e + 1) % m];
            bool vertical = (A.x == B.x);
            int oldVal = vertical ? A.x : A.y;

            int bestDelta = 0;
            int bestNewVal = oldVal;

            // try both outward and inward one‑step moves
            for (int dir = -1; dir <= 1; dir += 2) {
                int newVal = oldVal + dir;
                if (newVal < 0 || newVal > MAX_COORD) continue;

                int delta = 0;
                int sign = 0;
                if (vertical) {
                    int ql = min(oldVal, newVal) + 1;
                    int qr = max(oldVal, newVal);
                    delta = query_diff(ql, qr, min(A.y, B.y), max(A.y, B.y));
                    sign = (newVal > oldVal) ? 1 : -1;
                    if (A.y > B.y) sign = -sign;
                    if (signedArea < 0) sign = -sign;
                } else {
                    int ql = min(oldVal, newVal) + 1;
                    int qr = max(oldVal, newVal);
                    delta = query_diff(min(A.x, B.x), max(A.x, B.x), ql, qr);
                    sign = (newVal < oldVal) ? 1 : -1;
                    if (A.x > B.x) sign = -sign;
                    if (signedArea < 0) sign = -sign;
                }
                int diff = sign * delta;
                if (diff > bestDelta) {
                    bestDelta = diff;
                    bestNewVal = newVal;
                }
            }
            if (bestDelta <= 0) continue;

            // Apply the move
            Point oldA = A, oldB = B;
            if (vertical) {
                A.x = B.x = bestNewVal;
            } else {
                A.y = B.y = bestNewVal;
            }

            // Local validity
            vector<int> crit = { (e - 1 + m) % m, e, (e + 1) % m };
            if (!poly_valid(poly) || self_inter_local(poly, crit)) {
                // revert
                A = oldA; B = oldB;
                continue;
            }

            curDiff += bestDelta;
            signedArea = signed_area2(poly);
            improved = true;
        }
    }
}

// ------------------------------------------------------------
// Simulated annealing core
// ------------------------------------------------------------
void simulated_annealing() {
    // ---- initial solution ----
    vector<Point> curPoly = initial_rectangle();
    int curDiff = query_diff(curPoly[0].x, curPoly[2].x,
                             curPoly[0].y, curPoly[2].y);
    if (!poly_valid(curPoly) || !distinct_vertices(curPoly) || self_inter_full(curPoly)) {
        curPoly = {{0,0},{1,0},{1,1},{0,1}};
        curDiff = query_diff(0,1,0,1);
    }
    vector<Point> bestPoly = curPoly;
    int bestDiff = curDiff;
    long long signedArea = signed_area2(curPoly);
    if (signedArea == 0 && curPoly.size() >= 3) signedArea = 1;

    // ---- static guide set (fish coordinates ±1 + borders + random) ----
    {
        unordered_set<int> gx, gy;
        for (const auto& f : fish) {
            gx.insert(f.p.x);
            gx.insert(max(0, f.p.x - 1));
            gx.insert(min(MAX_COORD, f.p.x + 1));
            gy.insert(f.p.y);
            gy.insert(max(0, f.p.y - 1));
            gy.insert(min(MAX_COORD, f.p.y + 1));
        }
        gx.insert(0); gx.insert(MAX_COORD);
        gy.insert(0); gy.insert(MAX_COORD);
        guide_static_x.assign(gx.begin(), gx.end());
        guide_static_y.assign(gy.begin(), gy.end());
        for (int i = 0; i < 300; ++i) {
            guide_static_x.push_back(rng.next_int(0, MAX_COORD));
            guide_static_y.push_back(rng.next_int(0, MAX_COORD));
        }
    }

    // ---- annealing parameters ----
    const double START_T = 350.0;
    const double END_T   = 0.02;
    double last_restart = 0.0;
    const double RESTART_INTERVAL = 0.13;   // seconds

    // ---- main loop ----
    while (timer.elapsed() < TIME_LIMIT) {
        double progress = timer.elapsed() / TIME_LIMIT;
        double T = START_T * pow(END_T / START_T, progress);
        if (T < END_T) T = END_T;

        // Periodic restart to best solution
        if (timer.elapsed() - last_restart > RESTART_INTERVAL) {
            curPoly = bestPoly;
            curDiff = bestDiff;
            signedArea = signed_area2(curPoly);
            last_restart = timer.elapsed();
            continue;
        }

        // dynamic move probabilities
        int moveEdgeProb = 68, addBulgeProb = 12;
        int msize = (int)curPoly.size();
        if (msize + 2 > MAX_VERT) { moveEdgeProb = 82; addBulgeProb = 0; }
        else if (msize > 300) { moveEdgeProb = 58; addBulgeProb = 9; }

        int roll = rng.next_int(100);
        vector<int> critical;

        // ------------------------------------------------
        // 1) slide an edge
        // ------------------------------------------------
        if (roll < moveEdgeProb && curPoly.size() >= 4) {
            int m = (int)curPoly.size();
            int e = rng.next_int(m);
            Point& A = curPoly[e];
            Point& B = curPoly[(e + 1) % m];
            bool vertical = (A.x == B.x);
            int oldVal = vertical ? A.x : A.y;

            // propose new coordinate
            int newVal = -1;
            bool ok = false;
            const vector<int>* dyn = vertical ? &guide_best_x : &guide_best_y;
            const vector<int>* sta = vertical ? &guide_static_x : &guide_static_y;

            if (!dyn->empty() && rng.next_double() < 0.34) {
                newVal = (*dyn)[rng.next_int((int)dyn->size())];
                ok = true;
            }
            if (!ok && !sta->empty() && rng.next_double() < 0.80) {
                newVal = (*sta)[rng.next_int((int)sta->size())];
                ok = true;
            }
            if (!ok) {
                double fac = max(0.1, 1.0 - progress * 0.94);
                int step = max(1, (int)((MAX_COORD / 135.0) * fac + 1));
                int delta = rng.next_int(-step, step);
                if (progress > 0.78 && rng.next_double() < 0.65) delta = rng.next_int(-2, 2);
                if (delta == 0) delta = (rng.next_double() < 0.5) ? -1 : 1;
                newVal = (vertical ? A.x : A.y) + delta;
            }
            newVal = max(0, min(MAX_COORD, newVal));
            if (newVal == oldVal) continue;

            // compute delta net diff
            int delta = 0;
            int sign = 0;
            if (vertical) {
                int ql = min(oldVal, newVal) + 1;
                int qr = max(oldVal, newVal);
                delta = query_diff(ql, qr, min(A.y, B.y), max(A.y, B.y));
                sign = (newVal > oldVal) ? 1 : -1;
                if (A.y > B.y) sign = -sign;
                if (signedArea < 0) sign = -sign;
                A.x = B.x = newVal;
            } else {
                int ql = min(oldVal, newVal) + 1;
                int qr = max(oldVal, newVal);
                delta = query_diff(min(A.x, B.x), max(A.x, B.x), ql, qr);
                sign = (newVal < oldVal) ? 1 : -1;
                if (A.x > B.x) sign = -sign;
                if (signedArea < 0) sign = -sign;
                A.y = B.y = newVal;
            }

            int before = curDiff;
            curDiff += sign * delta;

            // local validity
            critical = { (e - 1 + m) % m, e, (e + 1) % m };
            if (!poly_valid(curPoly) || self_inter_local(curPoly, critical)) {
                // revert
                if (vertical) { A.x = B.x = oldVal; }
                else { A.y = B.y = oldVal; }
                curDiff = before;
                continue;
            }

            // accept / reject
            double diff = (double)(curDiff - before);
            if (diff >= 0.0 || (T > 1e-12 && rng.next_double() < exp(diff / T))) {
                signedArea = signed_area2(curPoly);
                if (curDiff > bestDiff) {
                    bestDiff = curDiff;
                    bestPoly = curPoly;
                    update_best_guides(bestPoly);
                }
            } else {
                // revert
                if (vertical) { A.x = B.x = oldVal; }
                else { A.y = B.y = oldVal; }
                curDiff = before;
            }
        }
        // ------------------------------------------------
        // 2) insert a rectangular bulge (adds two vertices)
        // ------------------------------------------------
        else if (roll < moveEdgeProb + addBulgeProb &&
                 (int)curPoly.size() + 2 <= MAX_VERT && curPoly.size() >= 4) {
            int m = (int)curPoly.size();
            int e = rng.next_int(m);
            Point& A = curPoly[e];
            Point& B = curPoly[(e + 1) % m];
            bool vertical = (A.x == B.x);
            int oldVal = vertical ? A.x : A.y;

            // propose new coordinate
            int newVal = -1;
            bool ok = false;
            const vector<int>* dyn = vertical ? &guide_best_x : &guide_best_y;
            const vector<int>* sta = vertical ? &guide_static_x : &guide_static_y;
            if (!dyn->empty() && rng.next_double() < 0.28) {
                newVal = (*dyn)[rng.next_int((int)dyn->size())];
                ok = true;
            }
            if (!ok && !sta->empty() && rng.next_double() < 0.82) {
                newVal = (*sta)[rng.next_int((int)sta->size())];
                ok = true;
            }
            if (!ok) {
                double fac = max(0.1, 1.0 - progress * 0.88);
                int depth = max(1, (int)((MAX_COORD / 285.0) * fac + 1));
                int d = rng.next_int(1, depth);
                if (progress > 0.78 && rng.next_double() < 0.68) d = rng.next_int(1, 2);
                int sgn = (rng.next_double() < 0.5) ? 1 : -1;
                newVal = (vertical ? A.x : A.y) + sgn * d;
            }
            newVal = max(0, min(MAX_COORD, newVal));
            if (newVal == oldVal) continue;

            // compute delta net diff
            int delta = 0;
            int sign = 0;
            Point v1, v2;
            if (vertical) {
                v1 = {newVal, A.y};
                v2 = {newVal, B.y};
                delta = query_diff(min(oldVal, newVal), max(oldVal, newVal),
                                   min(A.y, B.y), max(A.y, B.y));
                sign = (newVal > oldVal) ? 1 : -1;
                if (A.y > B.y) sign = -sign;
                if (signedArea < 0) sign = -sign;
            } else {
                v1 = {A.x, newVal};
                v2 = {B.x, newVal};
                delta = query_diff(min(A.x, B.x), max(A.x, B.x),
                                   min(oldVal, newVal), max(oldVal, newVal));
                sign = (newVal < oldVal) ? 1 : -1;
                if (A.x > B.x) sign = -sign;
                if (signedArea < 0) sign = -sign;
            }

            int before = curDiff;
            curDiff += sign * delta;

            // insert vertices
            int insertPos = e + 1;                     // after vertex A
            curPoly.insert(curPoly.begin() + insertPos, v1);
            curPoly.insert(curPoly.begin() + insertPos + 1, v2);

            // local checks
            critical = { e, e + 1, e + 2 };
            if (!poly_valid(curPoly) || self_inter_local(curPoly, critical)) {
                // revert insertion
                curPoly.erase(curPoly.begin() + insertPos,
                              curPoly.begin() + insertPos + 2);
                curDiff = before;
                continue;
            }

            // accept / reject
            double diff = (double)(curDiff - before);
            if (diff >= 0.0 || (T > 1e-12 && rng.next_double() < exp(diff / T))) {
                signedArea = signed_area2(curPoly);
                if (curDiff > bestDiff) {
                    bestDiff = curDiff;
                    bestPoly = curPoly;
                    update_best_guides(bestPoly);
                }
            } else {
                // revert insertion
                curPoly.erase(curPoly.begin() + insertPos,
                              curPoly.begin() + insertPos + 2);
                curDiff = before;
            }
        }
        // ------------------------------------------------
        // 3) delete a collinear vertex (if we have more than 4)
        // ------------------------------------------------
        else if (curPoly.size() > 4) {
            int m = (int)curPoly.size();
            int start = rng.next_int(m);
            bool done = false;
            for (int k = 0; k < m && !done; ++k) {
                int i = (start + k) % m;
                int ip = (i - 1 + m) % m;
                int in = (i + 1) % m;
                const Point& p0 = curPoly[ip];
                const Point& p1 = curPoly[i];
                const Point& p2 = curPoly[in];
                if ((p0.x == p1.x && p1.x == p2.x) ||
                    (p0.y == p1.y && p1.y == p2.y)) {
                    // erase vertex i
                    curPoly.erase(curPoly.begin() + i);
                    // local validity
                    int newM = (int)curPoly.size();
                    vector<int> crit = { (ip - 1 + newM) % newM,
                                         ip,
                                         (ip + 1) % newM };
                    if (!poly_valid(curPoly) ||
                        self_inter_local(curPoly, crit)) {
                        // revert
                        curPoly.insert(curPoly.begin() + i, p1);
                        continue;
                    }
                    done = true;
                }
            }
            // deletions do not affect curDiff
        }
        // ------------------------------------------------
        // (no move performed) continue loop
        // ------------------------------------------------
    }

    // ----- post‑annealing greedy improvement (inward/outward moves) -----
    greedy_expand(bestPoly, bestDiff);

    // ----- final safety check -----
    if (!poly_valid(bestPoly) || bestPoly.size() < 4 ||
        !distinct_vertices(bestPoly) || self_inter_full(bestPoly)) {
        // build a guaranteed empty rectangle if possible
        int minX = MAX_COORD, minY = MAX_COORD;
        int maxX = 0, maxY = 0;
        for (const auto& f : fish) {
            minX = min(minX, f.p.x);
            minY = min(minY, f.p.y);
            maxX = max(maxX, f.p.x);
            maxY = max(maxY, f.p.y);
        }
        bool set = false;
        int ex1, ey1, ex2, ey2;
        if (minX >= 2 && minY >= 2) {
            ex1 = 0; ey1 = 0; ex2 = minX - 1; ey2 = minY - 1;
            set = true;
        } else if (maxX <= MAX_COORD - 2 && maxY <= MAX_COORD - 2) {
            ex1 = maxX + 1; ey1 = maxY + 1; ex2 = MAX_COORD; ey2 = MAX_COORD;
            set = true;
        }
        if (set && ex1 < ex2 && ey1 < ey2) {
            bestPoly = {{ex1, ey1}, {ex2, ey1}, {ex2, ey2}, {ex1, ey2}};
            bestDiff = query_diff(ex1, ex2, ey1, ey2);
        } else {
            bestPoly = {{0,0},{1,0},{1,1},{0,1}};
            bestDiff = query_diff(0,1,0,1);
        }
    }
    if (bestDiff <= 0) {
        bestPoly = {{0,0},{1,0},{1,1},{0,1}};
    }

    // ----- output -----
    cout << bestPoly.size() << "\n";
    for (const auto& p : bestPoly) {
        cout << p.x << " " << p.y << "\n";
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // optional argument to override time limit (useful for local testing)
    if (argc > 1) {
        try { TIME_LIMIT = stod(argv[1]); } catch (...) {}
    }
    TIME_LIMIT -= SAFETY;
    if (TIME_LIMIT < 0.2) TIME_LIMIT = 0.2;
    timer.reset();

    int N;
    if (!(cin >> N)) return 0;
    fish.resize(2 * N);
    for (int i = 0; i < N; ++i) {
        cin >> fish[i].p.x >> fish[i].p.y;
        fish[i].type = 1;          // mackerel (+1)
    }
    for (int i = 0; i < N; ++i) {
        cin >> fish[N + i].p.x >> fish[N + i].p.y;
        fish[N + i].type = -1;     // sardine (-1)
    }

    // ---- coordinate compression (include borders) ----
    xs = {0, MAX_COORD};
    ys = {0, MAX_COORD};
    for (const auto& f : fish) {
        xs.push_back(f.p.x);
        ys.push_back(f.p.y);
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());
    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());
    NX = (int)xs.size();
    NY = (int)ys.size();

    // ---- build mapping tables for O(1) queries ----
    next_ge_x.assign(MAX_COORD + 1, NX);
    prev_le_x.assign(MAX_COORD + 1, -1);
    {
        int idx = 0;
        for (int x = 0; x <= MAX_COORD; ++x) {
            while (idx < NX && xs[idx] < x) ++idx;
            next_ge_x[x] = idx;
        }
        idx = -1;
        for (int x = 0; x <= MAX_COORD; ++x) {
            while (idx + 1 < NX && xs[idx + 1] <= x) ++idx;
            prev_le_x[x] = idx;
        }
    }
    next_ge_y.assign(MAX_COORD + 1, NY);
    prev_le_y.assign(MAX_COORD + 1, -1);
    {
        int idx = 0;
        for (int y = 0; y <= MAX_COORD; ++y) {
            while (idx < NY && ys[idx] < y) ++idx;
            next_ge_y[y] = idx;
        }
        idx = -1;
        for (int y = 0; y <= MAX_COORD; ++y) {
            while (idx + 1 < NY && ys[idx + 1] <= y) ++idx;
            prev_le_y[y] = idx;
        }
    }

    // ---- build 2‑D prefix sum of net difference ----
    prefDiff.assign((NY + 1) * (NX + 1), 0);
    for (const auto& f : fish) {
        int ix = lower_bound(xs.begin(), xs.end(), f.p.x) - xs.begin();
        int iy = lower_bound(ys.begin(), ys.end(), f.p.y) - ys.begin();
        short val = (short)f.type;   // +1 or -1
        prefDiff[IDX(ix + 1, iy + 1)] += val;
    }
    for (int y = 1; y <= NY; ++y) {
        for (int x = 1; x <= NX; ++x) {
            prefDiff[IDX(x, y)] += prefDiff[IDX(x - 1, y)] + prefDiff[IDX(x, y - 1)] - prefDiff[IDX(x - 1, y - 1)];
        }
    }

    // ---- compute a good starting rectangle using a coarse grid ----
    compute_grid_best_rectangle();

    // ---- run the meta‑heuristic ----
    simulated_annealing();

    return 0;
}