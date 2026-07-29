# EVOLVE-BLOCK-START

import math, random, time

def construct_set():
    """Construct a set of integers aiming to maximise C = log(|A+A|/|A|) / log(|A-A|/|A|)."""
    TIME_LIMIT = 165.0  # seconds for the main optimisation phase
    start_time = time.time()

    MIN_SIZE = MIN_SET_SIZE
    MAX_SIZE = MAX_SET_SIZE
    MIN_VAL = MIN_INT
    MAX_VAL = MAX_INT

    # -------------------------------------------------------------------------
    # Initialise from the best known construction if available, otherwise use a
    # small known MSTD seed.
    # -------------------------------------------------------------------------
    cur = None
    try:
        if GLOBAL_BEST_CONSTRUCTION:
            cand = sorted(set(int(x) for x in GLOBAL_BEST_CONSTRUCTION))
            if len(cand) >= MIN_SIZE:
                cur = cand
    except Exception:
        cur = None

    if cur is None:
        # Classic 8‑element MSTD set – a reliable starting point.
        cur = [0, 2, 3, 4, 7, 10, 12, 14]

    # Clip and dedupe, respect size limits
    cur = [x for x in cur if MIN_VAL <= x <= MAX_VAL]
    cur = sorted(set(cur))
    if len(cur) > MAX_SIZE:
        cur = cur[:MAX_SIZE]
    if len(cur) < MIN_SIZE:
        cur = [0, 1]

    # -------------------------------------------------------------------------
    # Frequency tables for sums and differences (exact counts, not just sets)
    # -------------------------------------------------------------------------
    sum_cnt = {}
    diff_cnt = {}
    for a in cur:
        for b in cur:
            sum_cnt[a + b] = sum_cnt.get(a + b, 0) + 1
            diff_cnt[a - b] = diff_cnt.get(a - b, 0) + 1

    n = len(cur)

    def current_c():
        """Current C‑value based on the frequency tables."""
        sum_sz = len(sum_cnt)
        diff_sz = len(diff_cnt)
        if sum_sz <= n or diff_sz <= n:
            return 0.0
        return math.log(sum_sz / n) / math.log(diff_sz / n)

    cur_c = current_c()
    best_set = cur[:]
    best_c = cur_c
    best_sum_cnt = sum_cnt.copy()
    best_diff_cnt = diff_cnt.copy()
    best_n = n

    # -------------------------------------------------------------------------
    # Simulated annealing – add / remove / replace moves.
    # -------------------------------------------------------------------------
    while time.time() - start_time < TIME_LIMIT:
        elapsed = time.time() - start_time
        # Simple linear cooling schedule
        T = max(1e-6, 1.0 - elapsed / TIME_LIMIT)

        move_type = random.random()
        # -------------------------------------------------- Add a new element
        if move_type < 0.4:
            if n < MAX_SIZE:
                pool = set()
                # jitter around existing elements
                for _ in range(1500):
                    pool.add(random.choice(cur) + random.randint(-600, 600))
                # completely random points
                for _ in range(1500):
                    pool.add(random.randint(MIN_VAL, MAX_VAL))
                cand_list = [x for x in pool if x not in cur and MIN_VAL <= x <= MAX_VAL]
                if not cand_list:
                    continue

                best_new_c = None
                best_x = None
                get_sum = sum_cnt.get
                get_diff = diff_cnt.get

                for x in cand_list:
                    # how many *new* sums would appear?
                    sum_added = 0
                    if get_sum(x + x, 0) == 0:
                        sum_added += 1
                    for a in cur:
                        if get_sum(x + a, 0) == 0:
                            sum_added += 1

                    # how many *new* differences would appear?
                    diff_added = 0
                    for a in cur:
                        if get_diff(x - a, 0) == 0:
                            diff_added += 1
                        if get_diff(a - x, 0) == 0:
                            diff_added += 1

                    new_n = n + 1
                    new_sum_sz = len(sum_cnt) + sum_added
                    new_diff_sz = len(diff_cnt) + diff_added
                    if new_sum_sz <= new_n or new_diff_sz <= new_n:
                        continue
                    new_c = math.log(new_sum_sz / new_n) / math.log(new_diff_sz / new_n)

                    if best_new_c is None or new_c > best_new_c:
                        best_new_c = new_c
                        best_x = x

                if best_x is not None:
                    delta = best_new_c - cur_c
                    accept = False
                    if delta > 0:
                        accept = True
                    else:
                        if random.random() < math.exp(delta / T):
                            accept = True
                    if accept:
                        x = best_x
                        # update tables
                        for a in cur:
                            s = x + a
                            sum_cnt[s] = sum_cnt.get(s, 0) + 2
                            d1 = x - a
                            diff_cnt[d1] = diff_cnt.get(d1, 0) + 1
                            d2 = a - x
                            diff_cnt[d2] = diff_cnt.get(d2, 0) + 1
                        sum_cnt[x + x] = sum_cnt.get(x + x, 0) + 1
                        diff_cnt[0] = diff_cnt.get(0, 0) + 1

                        cur.append(x)
                        cur.sort()
                        n = new_n
                        cur_c = best_new_c

                        if cur_c > best_c:
                            best_c = cur_c
                            best_set = cur[:]
                            best_sum_cnt = sum_cnt.copy()
                            best_diff_cnt = diff_cnt.copy()
                            best_n = n

        # -------------------------------------------------- Remove an element
        elif move_type < 0.8:
            if n > MIN_SIZE:
                sample = random.sample(cur, min(500, n))
                best_new_c = None
                best_y = None
                for y in sample:
                    # sums that would disappear
                    sum_removed = 0
                    if sum_cnt.get(y + y, 0) == 1:
                        sum_removed += 1
                    for a in cur:
                        if a == y:
                            continue
                        if sum_cnt.get(y + a, 0) == 2:
                            sum_removed += 1

                    # differences that would disappear
                    diff_removed = 0
                    for a in cur:
                        if a == y:
                            continue
                        if diff_cnt.get(y - a, 0) == 1:
                            diff_removed += 1
                        if diff_cnt.get(a - y, 0) == 1:
                            diff_removed += 1

                    new_n = n - 1
                    new_sum_sz = len(sum_cnt) - sum_removed
                    new_diff_sz = len(diff_cnt) - diff_removed
                    if new_sum_sz <= new_n or new_diff_sz <= new_n:
                        continue
                    new_c = math.log(new_sum_sz / new_n) / math.log(new_diff_sz / new_n)

                    if best_new_c is None or new_c > best_new_c:
                        best_new_c = new_c
                        best_y = y

                if best_y is not None:
                    delta = best_new_c - cur_c
                    accept = False
                    if delta > 0:
                        accept = True
                    else:
                        if random.random() < math.exp(delta / T):
                            accept = True
                    if accept:
                        y = best_y
                        # delete contributions of y
                        cnt = sum_cnt[y + y] - 1
                        if cnt:
                            sum_cnt[y + y] = cnt
                        else:
                            del sum_cnt[y + y]

                        for a in cur:
                            if a == y:
                                continue
                            s = y + a
                            cnt = sum_cnt[s] - 2
                            if cnt:
                                sum_cnt[s] = cnt
                            else:
                                del sum_cnt[s]

                        cnt0 = diff_cnt[0] - 1
                        if cnt0:
                            diff_cnt[0] = cnt0
                        else:
                            del diff_cnt[0]

                        for a in cur:
                            if a == y:
                                continue
                            d1 = y - a
                            cnt = diff_cnt[d1] - 1
                            if cnt:
                                diff_cnt[d1] = cnt
                            else:
                                del diff_cnt[d1]
                            d2 = a - y
                            cnt = diff_cnt[d2] - 1
                            if cnt:
                                diff_cnt[d2] = cnt
                            else:
                                del diff_cnt[d2]

                        cur.remove(y)
                        n = new_n
                        cur_c = best_new_c

                        if cur_c > best_c:
                            best_c = cur_c
                            best_set = cur[:]
                            best_sum_cnt = sum_cnt.copy()
                            best_diff_cnt = diff_cnt.copy()
                            best_n = n

        # -------------------------------------------------- Replace (swap) an element
        else:
            if n > 0:
                attempts = 800
                best_new_c = None
                best_y = None
                best_x = None
                get_sum = sum_cnt.get
                get_diff = diff_cnt.get

                for _ in range(attempts):
                    y = random.choice(cur)
                    # propose a new value
                    if random.random() < 0.7:
                        x = y + random.randint(-600, 600)
                    else:
                        x = random.randint(MIN_VAL, MAX_VAL)
                    if x == y or x in cur or not (MIN_VAL <= x <= MAX_VAL):
                        continue

                    # contributions lost by removing y
                    sum_removed = 0
                    if sum_cnt.get(y + y, 0) == 1:
                        sum_removed += 1
                    for a in cur:
                        if a == y:
                            continue
                        if sum_cnt.get(y + a, 0) == 2:
                            sum_removed += 1

                    diff_removed = 0
                    for a in cur:
                        if a == y:
                            continue
                        if diff_cnt.get(y - a, 0) == 1:
                            diff_removed += 1
                        if diff_cnt.get(a - y, 0) == 1:
                            diff_removed += 1

                    # contributions added by inserting x
                    sum_added = 0
                    if get_sum(x + x, 0) == 0:
                        sum_added += 1
                    for a in cur:
                        if a == y:
                            continue
                        if get_sum(x + a, 0) == 0:
                            sum_added += 1

                    diff_added = 0
                    for a in cur:
                        if a == y:
                            continue
                        if get_diff(x - a, 0) == 0:
                            diff_added += 1
                        if get_diff(a - x, 0) == 0:
                            diff_added += 1

                    new_sum_sz = len(sum_cnt) - sum_removed + sum_added
                    new_diff_sz = len(diff_cnt) - diff_removed + diff_added
                    if new_sum_sz <= n or new_diff_sz <= n:
                        continue
                    new_c = math.log(new_sum_sz / n) / math.log(new_diff_sz / n)

                    if best_new_c is None or new_c > best_new_c:
                        best_new_c = new_c
                        best_y = y
                        best_x = x

                if best_x is not None:
                    delta = best_new_c - cur_c
                    accept = False
                    if delta > 0:
                        accept = True
                    else:
                        if random.random() < math.exp(delta / T):
                            accept = True
                    if accept:
                        y = best_y
                        x = best_x
                        # ---- remove y ----
                        cnt = sum_cnt[y + y] - 1
                        if cnt:
                            sum_cnt[y + y] = cnt
                        else:
                            del sum_cnt[y + y]

                        for a in cur:
                            if a == y:
                                continue
                            s = y + a
                            cnt = sum_cnt[s] - 2
                            if cnt:
                                sum_cnt[s] = cnt
                            else:
                                del sum_cnt[s]

                        cnt0 = diff_cnt[0] - 1
                        if cnt0:
                            diff_cnt[0] = cnt0
                        else:
                            del diff_cnt[0]

                        for a in cur:
                            if a == y:
                                continue
                            d1 = y - a
                            cnt = diff_cnt[d1] - 1
                            if cnt:
                                diff_cnt[d1] = cnt
                            else:
                                del diff_cnt[d1]
                            d2 = a - y
                            cnt = diff_cnt[d2] - 1
                            if cnt:
                                diff_cnt[d2] = cnt
                            else:
                                del diff_cnt[d2]

                        # ---- add x ----
                        for a in cur:
                            if a == y:
                                continue
                            s = x + a
                            sum_cnt[s] = sum_cnt.get(s, 0) + 2
                            d1 = x - a
                            diff_cnt[d1] = diff_cnt.get(d1, 0) + 1
                            d2 = a - x
                            diff_cnt[d2] = diff_cnt.get(d2, 0) + 1
                        sum_cnt[x + x] = sum_cnt.get(x + x, 0) + 1
                        diff_cnt[0] = diff_cnt.get(0, 0) + 1

                        # update the list representation
                        idx = cur.index(y)
                        cur[idx] = x
                        cur.sort()
                        cur_c = best_new_c
                        if cur_c > best_c:
                            best_c = cur_c
                            best_set = cur[:]
                            best_sum_cnt = sum_cnt.copy()
                            best_diff_cnt = diff_cnt.copy()
                            best_n = n

        # occasional random restart from the best seen configuration
        if random.random() < 0.01:
            cur = best_set[:]
            sum_cnt = best_sum_cnt.copy()
            diff_cnt = best_diff_cnt.copy()
            n = best_n
            cur_c = best_c

    # -------------------------------------------------------------------------
    # Post‑processing: prune any removable element that improves C.
    # -------------------------------------------------------------------------
    improved = True
    while improved:
        improved = False
        for y in list(cur):
            sum_removed = 0
            if sum_cnt.get(y + y, 0) == 1:
                sum_removed += 1
            for a in cur:
                if a == y:
                    continue
                if sum_cnt.get(y + a, 0) == 2:
                    sum_removed += 1

            diff_removed = 0
            for a in cur:
                if a == y:
                    continue
                if diff_cnt.get(y - a, 0) == 1:
                    diff_removed += 1
                if diff_cnt.get(a - y, 0) == 1:
                    diff_removed += 1

            new_n = n - 1
            new_sum_sz = len(sum_cnt) - sum_removed
            new_diff_sz = len(diff_cnt) - diff_removed
            if new_sum_sz <= new_n or new_diff_sz <= new_n:
                continue
            new_c = math.log(new_sum_sz / new_n) / math.log(new_diff_sz / new_n)
            if new_c > cur_c + 1e-12:
                # perform removal
                cnt = sum_cnt[y + y] - 1
                if cnt:
                    sum_cnt[y + y] = cnt
                else:
                    del sum_cnt[y + y]

                for a in cur:
                    if a == y:
                        continue
                    s = y + a
                    cnt = sum_cnt[s] - 2
                    if cnt:
                        sum_cnt[s] = cnt
                    else:
                        del sum_cnt[s]

                cnt0 = diff_cnt[0] - 1
                if cnt0:
                    diff_cnt[0] = cnt0
                else:
                    del diff_cnt[0]

                for a in cur:
                    if a == y:
                        continue
                    d1 = y - a
                    cnt = diff_cnt[d1] - 1
                    if cnt:
                        diff_cnt[d1] = cnt
                    else:
                        del diff_cnt[d1]
                    d2 = a - y
                    cnt = diff_cnt[d2] - 1
                    if cnt:
                        diff_cnt[d2] = cnt
                    else:
                        del diff_cnt[d2]

                cur.remove(y)
                n = new_n
                cur_c = new_c
                if cur_c > best_c:
                    best_c = cur_c
                    best_set = cur[:]
                improved = True
                break

    # -------------------------------------------------------------------------
    # Greedy addition: keep adding the best element while it raises C.
    # -------------------------------------------------------------------------
    while n < MAX_SIZE:
        pool = set()
        cur_min, cur_max = cur[0], cur[-1]
        # extend both ends modestly
        for d in range(1, 501):
            pool.add(cur_min - d)
            pool.add(cur_max + d)
        # sprinkle random points
        for _ in range(3000):
            pool.add(random.randint(MIN_VAL, MAX_VAL))
        cand_list = [x for x in pool if x not in cur and MIN_VAL <= x <= MAX_VAL]
        if not cand_list:
            break

        best_new_c = None
        best_x = None
        get_sum = sum_cnt.get
        get_diff = diff_cnt.get

        for x in cand_list:
            sum_added = 0
            if get_sum(x + x, 0) == 0:
                sum_added += 1
            for a in cur:
                if get_sum(x + a, 0) == 0:
                    sum_added += 1

            diff_added = 0
            for a in cur:
                if get_diff(x - a, 0) == 0:
                    diff_added += 1
                if get_diff(a - x, 0) == 0:
                    diff_added += 1

            new_n = n + 1
            new_sum_sz = len(sum_cnt) + sum_added
            new_diff_sz = len(diff_cnt) + diff_added
            if new_sum_sz <= new_n or new_diff_sz <= new_n:
                continue
            new_c = math.log(new_sum_sz / new_n) / math.log(new_diff_sz / new_n)

            if best_new_c is None or new_c > best_new_c:
                best_new_c = new_c
                best_x = x

        if best_x is None or best_new_c <= cur_c:
            break

        # apply the selected addition
        x = best_x
        for a in cur:
            s = x + a
            sum_cnt[s] = sum_cnt.get(s, 0) + 2
            d1 = x - a
            diff_cnt[d1] = diff_cnt.get(d1, 0) + 1
            d2 = a - x
            diff_cnt[d2] = diff_cnt.get(d2, 0) + 1
        sum_cnt[x + x] = sum_cnt.get(x + x, 0) + 1
        diff_cnt[0] = diff_cnt.get(0, 0) + 1

        cur.append(x)
        cur.sort()
        n = new_n
        cur_c = best_new_c
        if cur_c > best_c:
            best_c = cur_c
            best_set = cur[:]

    # -------------------------------------------------------------------------
    # Local replacement refinement (small neighbourhood search)
    # -------------------------------------------------------------------------
    improved = True
    while improved:
        improved = False
        elems = cur[:]
        random.shuffle(elems)
        for y in elems:
            best_local_c = cur_c
            best_local_x = None
            for _ in range(30):
                if random.random() < 0.5:
                    x = y + random.randint(-600, 600)
                else:
                    x = random.randint(MIN_VAL, MAX_VAL)
                if x == y or x in cur or not (MIN_VAL <= x <= MAX_VAL):
                    continue

                # remove y
                sum_removed = 0
                if sum_cnt.get(y + y, 0) == 1:
                    sum_removed += 1
                for a in cur:
                    if a == y:
                        continue
                    if sum_cnt.get(y + a, 0) == 2:
                        sum_removed += 1

                diff_removed = 0
                for a in cur:
                    if a == y:
                        continue
                    if diff_cnt.get(y - a, 0) == 1:
                        diff_removed += 1
                    if diff_cnt.get(a - y, 0) == 1:
                        diff_removed += 1

                # add x
                sum_added = 0
                if sum_cnt.get(x + x, 0) == 0:
                    sum_added += 1
                for a in cur:
                    if a == y:
                        continue
                    if sum_cnt.get(x + a, 0) == 0:
                        sum_added += 1

                diff_added = 0
                for a in cur:
                    if a == y:
                        continue
                    if diff_cnt.get(x - a, 0) == 0:
                        diff_added += 1
                    if diff_cnt.get(a - x, 0) == 0:
                        diff_added += 1

                new_sum_sz = len(sum_cnt) - sum_removed + sum_added
                new_diff_sz = len(diff_cnt) - diff_removed + diff_added
                if new_sum_sz <= n or new_diff_sz <= n:
                    continue
                new_c = math.log(new_sum_sz / n) / math.log(new_diff_sz / n)
                if new_c > best_local_c + 1e-12:
                    best_local_c = new_c
                    best_local_x = x

            if best_local_x is not None:
                # perform the replacement
                x = best_local_x
                # ----- remove y -----
                cnt = sum_cnt[y + y] - 1
                if cnt:
                    sum_cnt[y + y] = cnt
                else:
                    del sum_cnt[y + y]

                for a in cur:
                    if a == y:
                        continue
                    s = y + a
                    cnt = sum_cnt[s] - 2
                    if cnt:
                        sum_cnt[s] = cnt
                    else:
                        del sum_cnt[s]

                cnt0 = diff_cnt[0] - 1
                if cnt0:
                    diff_cnt[0] = cnt0
                else:
                    del diff_cnt[0]

                for a in cur:
                    if a == y:
                        continue
                    d1 = y - a
                    cnt = diff_cnt[d1] - 1
                    if cnt:
                        diff_cnt[d1] = cnt
                    else:
                        del diff_cnt[d1]
                    d2 = a - y
                    cnt = diff_cnt[d2] - 1
                    if cnt:
                        diff_cnt[d2] = cnt
                    else:
                        del diff_cnt[d2]

                # ----- add x -----
                for a in cur:
                    if a == y:
                        continue
                    s = x + a
                    sum_cnt[s] = sum_cnt.get(s, 0) + 2
                    d1 = x - a
                    diff_cnt[d1] = diff_cnt.get(d1, 0) + 1
                    d2 = a - x
                    diff_cnt[d2] = diff_cnt.get(d2, 0) + 1
                sum_cnt[x + x] = sum_cnt.get(x + x, 0) + 1
                diff_cnt[0] = diff_cnt.get(0, 0) + 1

                # update list
                idx = cur.index(y)
                cur[idx] = x
                cur.sort()
                cur_c = best_local_c
                if cur_c > best_c:
                    best_c = cur_c
                    best_set = cur[:]
                    best_sum_cnt = sum_cnt.copy()
                    best_diff_cnt = diff_cnt.copy()
                    best_n = n
                improved = True
                break  # restart neighbourhood scan

    return best_set
# EVOLVE-BLOCK-END

MIN_SET_SIZE = 2
MAX_SET_SIZE = 512
MIN_INT = -1_000_000
MAX_INT = 1_000_000


def _sanitize_output(values):
    """Convert arbitrary iterable output into a valid sorted integer list."""
    try:
        raw = list(values)
    except TypeError as e:
        raise ValueError(f"Output is not iterable: {e}")

    ints = []
    for x in raw:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(xf):
            continue
        xi = int(round(xf))
        xi = max(MIN_INT, min(MAX_INT, xi))
        ints.append(xi)

    unique_vals = sorted(set(ints))
    if len(unique_vals) > MAX_SET_SIZE:
        unique_vals = unique_vals[:MAX_SET_SIZE]

    if len(unique_vals) < MIN_SET_SIZE:
        unique_vals = [0, 1]

    return unique_vals


def _compute_c(values):
    n = len(values)
    sumset = {a + b for a in values for b in values}
    diffset = {a - b for a in values for b in values}

    sum_ratio = len(sumset) / n
    diff_ratio = len(diffset) / n

    if sum_ratio <= 1.0 or diff_ratio <= 1.0:
        return 0.0

    return float(math.log(sum_ratio) / math.log(diff_ratio))


def run_code():
    """Return (A_values, claimed_c)."""
    values = construct_set()
    values = _sanitize_output(values)
    c_value = _compute_c(values)
    return values, c_value


if __name__ == "__main__":
    candidate_values, candidate_c = run_code()
    print(f"|A|={len(candidate_values)}, C(A)={candidate_c:.10f}")
