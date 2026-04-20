from __future__ import annotations
from utils_ae import *
import os
import bisect
import math
import re
import uuid
from pathlib import Path
from typing import Any, Optional
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import numpy as np

# EVOLVE-BLOCK-START

from scipy.optimize import linear_sum_assignment


class Placer(AbstractPlacer):
    """
    Router‑friendly placer.

    The algorithm builds a variety of qubit orderings, evaluates each
    using the router (counting ``move`` operations as the primary metric
    and total Euclidean travel distance as a tie‑breaker), and then
    refines the best candidate with reverse‑through‑time (RTT) passes,
    hill‑climbing swaps in the initial layout and intra‑row swap
    optimisation for storage destinations.
    """

    # ------------------------------------------------------------------
    # Tunable parameters (feel free to adjust)
    # ------------------------------------------------------------------
    COST_MULTIPLIER: float = 4.0          # column‑distance budget before fallback
    SWAP_PASSES: int = 6                  # intra‑row swap optimisation passes
    ROW_WEIGHT: float = 1.0               # weight for row‑distance in storage‑assign cost
    HILL_CLIMB_ATTEMPTS: int = 400        # random swaps on the best initial layout
    RANDOM_ORDER_SEEDS: tuple[int, ...] = tuple(range(1, 6))
    RTT_ROUNDS: int = 3                   # reverse‑through‑time refinement rounds
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _f_ent_col(col: int, d_ec: float, ryd: float) -> float:
        """Physical y‑coordinate of an entangling column (mirrors evaluator)."""
        base = (col // 2) * d_ec
        offset = -ryd / 2.0 if (col % 2 == 0) else ryd / 2.0
        return base + offset

    def _target_storage_col(
        self,
        q: int,
        loc: dict[int, tuple[int, int]],
        d_ec: float,
        d_sc: float,
        ryd: float,
        L_ec: float,
        L_sc: float,
        storage_cols: int,
    ) -> int:
        """Map an entangling column to a preferred storage‑column index."""
        ent_col = loc[q][1]
        y_ent = self._f_ent_col(ent_col, d_ec, ryd) - L_ec / 2.0
        t_c = int(round((y_ent + L_sc / 2.0) / d_sc))
        return max(0, min(storage_cols - 1, t_c))

    # ------------------------------------------------------------------
    # DP for optimal monotone column assignment (preserves order)
    # ------------------------------------------------------------------
    def _optimal_monotone_assignment(self, targets: list[int], free: list[int]) -> list[int]:
        """Minimise Σ|free[i]‑target[i]| while preserving order (both lists sorted)."""
        m, n = len(targets), len(free)
        INF = 10 ** 12
        dp = [[INF] * (n + 1) for _ in range(m + 1)]
        take = [[False] * (n + 1) for _ in range(m + 1)]

        for j in range(n + 1):
            dp[0][j] = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if dp[i][j - 1] < dp[i][j]:
                    dp[i][j] = dp[i][j - 1]
                    take[i][j] = False
                cost = dp[i - 1][j - 1] + abs(free[j - 1] - targets[i - 1])
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    take[i][j] = True

        assign = [0] * m
        i, j = m, n
        while i > 0:
            if take[i][j]:
                assign[i - 1] = free[j - 1]
                i -= 1
                j -= 1
            else:
                j -= 1
        return assign

    # ------------------------------------------------------------------
    # Assign storage cells for a group sharing the same source entangling row.
    # Returns (chosen_row, max_assigned_column).
    # ------------------------------------------------------------------
    def _assign_storage_for_group(
        self,
        group: list[int],
        loc: dict[int, tuple[int, int]],
        storage_placement: dict[int, tuple[int, int, int]],
        occupied: set[tuple[int, int, int]],
        storage_rows: int,
        storage_cols: int,
        d_ec: float,
        d_sc: float,
        ryd: float,
        L_ec: float,
        L_sc: float,
        last_assigned_row: int,
        global_last_col: int,
        cost_threshold: float,
        used_rows: set[int],
    ) -> tuple[int, int]:
        # distance‑aware target columns (storage‑index space)
        target_cols = [
            self._target_storage_col(
                q, loc, d_ec, d_sc, ryd, L_ec, L_sc, storage_cols
            )
            for q in group
        ]

        # preserve source‑order inside the group (router‑compatible)
        paired = list(zip(target_cols, group))
        paired.sort(key=lambda x: x[0])
        sorted_targets = [t for t, _ in paired]
        sorted_qubits = [q for _, q in paired]

        src_row = loc[group[0]][0]                     # all share this source row
        best_row, best_assign, best_cost = None, None, None
        best_col_cost = None

        # ----- try a global monotone placement (columns > global_last_col) -----
        for r in range(max(last_assigned_row, 0), storage_rows):
            if r in used_rows:
                continue
            free = [
                c for c in range(storage_cols)
                if (0, r, c) not in occupied and c > global_last_col
            ]
            if len(free) < len(group):
                continue
            assign = self._optimal_monotone_assignment(sorted_targets, free)
            col_cost = sum(abs(a - t) for a, t in zip(assign, sorted_targets))
            row_cost = self.ROW_WEIGHT * abs(r - src_row) * len(group)
            total = col_cost + row_cost
            if best_cost is None or total < best_cost:
                best_cost, best_row, best_assign, best_col_cost = total, r, assign, col_cost
            if col_cost <= cost_threshold:
                break

        if (
            best_row is not None
            and best_assign is not None
            and best_col_cost is not None
            and best_col_cost <= cost_threshold
        ):
            for q, col in zip(sorted_qubits, best_assign):
                storage_placement[q] = (0, best_row, col)
                occupied.add((0, best_row, col))
            return best_row, max(best_assign)

        # ----- fallback: row‑local monotone placement (ignore global column order) -----
        fallback_row, fallback_assign, fallback_cost = None, None, None
        for r in range(max(last_assigned_row, 0), storage_rows):
            if r in used_rows:
                continue
            free = [
                c for c in range(storage_cols)
                if (0, r, c) not in occupied
            ]
            if len(free) < len(group):
                continue
            assign = self._optimal_monotone_assignment(sorted_targets, free)
            col_cost = sum(abs(a - t) for a, t in zip(assign, sorted_targets))
            row_cost = self.ROW_WEIGHT * abs(r - src_row) * len(group)
            total = col_cost + row_cost
            if fallback_cost is None or total < fallback_cost:
                fallback_cost, fallback_row, fallback_assign = total, r, assign
                if col_cost == 0:
                    break

        if fallback_row is None or fallback_assign is None:
            raise RuntimeError(
                "Ran out of storage rows/columns while placing moving qubits"
            )

        for q, col in zip(sorted_qubits, fallback_assign):
            storage_placement[q] = (0, fallback_row, col)
            occupied.add((0, fallback_row, col))
        return fallback_row, max(fallback_assign)

    # ------------------------------------------------------------------
    # Multi‑pass intra‑row swap optimiser (router‑compatible)
    # ------------------------------------------------------------------
    def _optimize_storage_swaps(
        self,
        moving_qubits: set[int],
        loc: dict[int, tuple[int, int]],
        storage_placement: dict[int, tuple[int, int, int]],
    ) -> None:
        if self.SWAP_PASSES == 0 or len(moving_qubits) < 2:
            return

        src_phys = {
            q: self.calc_physical_coordinate(1, *loc[q]) for q in moving_qubits
        }

        for _ in range(self.SWAP_PASSES):
            improved = True
            while improved:
                improved = False
                qubits = list(moving_qubits)
                for i in range(len(qubits)):
                    qi = qubits[i]
                    for j in range(i + 1, len(qubits)):
                        qj = qubits[j]

                        # current Euclidean cost
                        cur = (
                            math.hypot(
                                src_phys[qi][0] -
                                self.calc_physical_coordinate(*storage_placement[qi])[0],
                                src_phys[qi][1] -
                                self.calc_physical_coordinate(*storage_placement[qi])[1],
                            )
                            + math.hypot(
                                src_phys[qj][0] -
                                self.calc_physical_coordinate(*storage_placement[qj])[0],
                                src_phys[qj][1] -
                                self.calc_physical_coordinate(*storage_placement[qj])[1],
                            )
                        )
                        # cost after swapping destinations
                        swap = (
                            math.hypot(
                                src_phys[qi][0] -
                                self.calc_physical_coordinate(*storage_placement[qj])[0],
                                src_phys[qi][1] -
                                self.calc_physical_coordinate(*storage_placement[qj])[1],
                            )
                            + math.hypot(
                                src_phys[qj][0] -
                                self.calc_physical_coordinate(*storage_placement[qi])[0],
                                src_phys[qj][1] -
                                self.calc_physical_coordinate(*storage_placement[qi])[1],
                            )
                        )
                        if swap + 1e-9 >= cur:
                            continue

                        # test router compatibility after the swap
                        dst_i = storage_placement[qj]
                        dst_j = storage_placement[qi]
                        ok = True
                        for qk in moving_qubits:
                            if qk in (qi, qj):
                                continue
                            a_vec = (loc[qi][0], dst_i[1], loc[qi][1], dst_i[2])
                            b_vec = (loc[qk][0], storage_placement[qk][1],
                                     loc[qk][1], storage_placement[qk][2])
                            if not Router._compatible_2d(a_vec, b_vec):
                                ok = False
                                break
                            a_vec2 = (loc[qj][0], dst_j[1], loc[qj][1], dst_j[2])
                            if not Router._compatible_2d(a_vec2, b_vec):
                                ok = False
                                break
                        if not ok:
                            continue
                        if not Router._compatible_2d(
                            (loc[qi][0], dst_i[1], loc[qi][1], dst_i[2]),
                            (loc[qj][0], dst_j[1], loc[qj][1], dst_j[2]),
                        ):
                            continue

                        # accept swap
                        storage_placement[qi], storage_placement[qj] = storage_placement[qj], storage_placement[qi]
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    for q in moving_qubits:
                        src_phys[q] = self.calc_physical_coordinate(1, *loc[q])

    # ------------------------------------------------------------------
    # Core placement routine for a single forward (or reverse) pass.
    # Returns (placements_by_2q_stage, final_storage_snapshot,
    #          total_euclidean_distance)
    # ------------------------------------------------------------------
    def _single_pass(
        self,
        stages: list[ZNAAStage],
        reuse_info: list[list[int]],
        init: list[tuple[int, int, int]],
    ) -> tuple[
        list[list[list[tuple[int, int, int]]]],
        list[tuple[int, int, int]],
        float,
    ]:
        n_q = len(init)

        storage_rows, storage_cols = self.config.storage_shape
        ent_rows, ent_cols = self.config.entangling_shape
        if ent_cols % 2 != 0:
            raise RuntimeError("Entangling zone must have an even number of columns")
        pair_cols = ent_cols // 2

        d_ec = self.config.distance_entangle[1]
        d_sc = self.config.distance_storage[1]
        ryd = self.config.rydberg_radius

        L_ec = self._f_ent_col(ent_cols - 1, d_ec, ryd)
        L_sc = (storage_cols - 1) * d_sc

        two_q_stage_indices = [
            i for i, st in enumerate(stages) if st.stage_type == "2q" and st.gates
        ]

        # current placement of qubits that are in storage
        storage_placement: dict[int, tuple[int, int, int]] = {q: init[q] for q in range(n_q)}
        occupied: set[tuple[int, int, int]] = set(init)

        # map of qubits that stay in the entangling zone across stages
        reused_loc: dict[int, tuple[int, int]] = {}   # q → (ent_row, ent_col)

        placements_by_2q_stage: list[list[list[tuple[int, int, int]]]] = []

        for gi, stage_idx in enumerate(two_q_stage_indices):
            stage = stages[stage_idx]

            # ------------------------------------------------------------------
            # 1️⃣ Determine active qubits and reuse set for this stage
            # ------------------------------------------------------------------
            pairs: list[tuple[int, int]] = [(g.qubits[0], g.qubits[1]) for g in stage.gates]
            active_this_stage: set[int] = {q for a, b in pairs for q in (a, b)}

            # explicit reuse hints (restricted to active qubits)
            reuse_out: set[int] = set(reuse_info[stage_idx]) & active_this_stage

            # persistent reuse: keep any qubit that stays paired with the same partner in the next stage
            if gi + 1 < len(two_q_stage_indices):
                next_stage_idx = two_q_stage_indices[gi + 1]
                next_pairs = {
                    frozenset((g.qubits[0], g.qubits[1])) for g in stages[next_stage_idx].gates
                }
                for q1, q2 in pairs:
                    if frozenset((q1, q2)) in next_pairs:
                        reuse_out.update([q1, q2])

            # final 2Q stage – force all qubits back to storage
            if gi == len(two_q_stage_indices) - 1:
                reuse_out.clear()

            # drop stale reused entries (qubits that are no longer active)
            for q in list(reused_loc):
                if q not in active_this_stage:
                    reused_loc.pop(q)

            # remember where active qubits currently sit in storage (if not reused)
            prev_storage: dict[int, tuple[int, int, int]] = {}
            for q in active_this_stage:
                if q not in reused_loc:
                    prev_storage[q] = storage_placement[q]

            # remove those moving qubits from the storage bookkeeping.
            for q, cell in prev_storage.items():
                occupied.discard(cell)
                del storage_placement[q]

            # ------------------------------------------------------------------
            # 2️⃣ Place active qubits into the entangling zone
            # ------------------------------------------------------------------
            loc: dict[int, tuple[int, int]] = {}                # qubit → (ent_row, ent_col)
            slot_parities: dict[tuple[int, int], set[int]] = {} # (row, pair_idx) → {0,1}
            free_pairs: list[tuple[int, int]] = []              # pairs still needing placement

            # keep already‑reused qubits that stay active
            for q, (r, c) in reused_loc.items():
                if q in active_this_stage:
                    loc[q] = (r, c)
                    slot = (r, c // 2)
                    parity = c % 2
                    slot_parities.setdefault(slot, set()).add(parity)

            # handle each logical pair
            for q1, q2 in pairs:
                in1 = q1 in loc
                in2 = q2 in loc

                if in1 and in2:
                    r1, c1 = loc[q1]
                    r2, c2 = loc[q2]
                    # already adjacent?
                    if r1 == r2 and c1 // 2 == c2 // 2 and abs(c1 - c2) == 1:
                        slot_parities[(r1, c1 // 2)] = {0, 1}
                        continue
                    free_pairs.append((q1, q2))
                    continue

                if in1 ^ in2:
                    # one endpoint already fixed – place the other on the free parity
                    fixed_q, other_q = (q1, q2) if in1 else (q2, q1)
                    r_fixed, c_fixed = loc[fixed_q]
                    pair_idx = c_fixed // 2
                    parity_fixed = c_fixed % 2
                    parity_other = 1 - parity_fixed
                    slot = (r_fixed, pair_idx)
                    used = slot_parities.setdefault(slot, set())
                    if parity_other in used:
                        free_pairs.append((q1, q2))
                        continue
                    c_other = 2 * pair_idx + parity_other
                    loc[other_q] = (r_fixed, c_other)
                    used.update({parity_fixed, parity_other})
                    continue

                free_pairs.append((q1, q2))

            # ----- bulk placement for remaining pairs (Hungarian per row) -----
            if free_pairs:
                def src_row(q: int) -> int:
                    if q in prev_storage:
                        return prev_storage[q][1]          # storage row
                    return reused_loc.get(q, (0, 0))[0]   # entangling row for reused

                def src_col(q: int) -> int:
                    if q in prev_storage:
                        return prev_storage[q][2]          # storage column
                    return reused_loc.get(q, (0, 0))[1]   # entangling column for reused

                # order pairs: higher source rows first, then smaller source column
                def pair_key(p: tuple[int, int]) -> tuple[float, int]:
                    r1, r2 = src_row(p[0]), src_row(p[1])
                    avg_r = -(r1 + r2) / 2.0               # higher rows first
                    c1, c2 = src_col(p[0]), src_col(p[1])
                    min_c = min(c1, c2)
                    return (avg_r, min_c)

                free_pairs.sort(key=pair_key)

                # free slots per entangling row (pair‑columns)
                free_slots_per_row: dict[int, list[int]] = {
                    r: [
                        p for p in range(pair_cols)
                        if (r, p) not in slot_parities or len(slot_parities[(r, p)]) == 0
                    ]
                    for r in range(ent_rows)
                }

                pair_ptr = 0
                for row in range(ent_rows - 1, -1, -1):
                    slots = free_slots_per_row[row]
                    if not slots:
                        continue
                    to_place = min(len(slots), len(free_pairs) - pair_ptr)
                    if to_place <= 0:
                        break
                    row_pairs = free_pairs[pair_ptr : pair_ptr + to_place]

                    # build cost matrix (pairs × slots)
                    cost_mat = np.zeros((len(row_pairs), len(slots)), dtype=float)
                    for pi, (q1, q2) in enumerate(row_pairs):
                        src1 = self.calc_physical_coordinate(*prev_storage.get(q1, (0, 0, 0)))
                        src2 = self.calc_physical_coordinate(*prev_storage.get(q2, (0, 0, 0)))
                        for si, pair_idx in enumerate(slots):
                            even_col = 2 * pair_idx
                            odd_col = even_col + 1
                            dst_even = self.calc_physical_coordinate(1, row, even_col)
                            dst_odd = self.calc_physical_coordinate(1, row, odd_col)
                            cost1 = math.hypot(src1[0] - dst_even[0], src1[1] - dst_even[1]) + \
                                    math.hypot(src2[0] - dst_odd[0], src2[1] - dst_odd[1])
                            cost2 = math.hypot(src1[0] - dst_odd[0], src1[1] - dst_odd[1]) + \
                                    math.hypot(src2[0] - dst_even[0], src2[1] - dst_even[1])
                            cost_mat[pi, si] = min(cost1, cost2)

                    r_idx, c_idx = linear_sum_assignment(cost_mat)
                    assigned_slots = [slots[i] for i in c_idx]

                    for pi, (q1, q2) in enumerate(row_pairs):
                        pair_idx = assigned_slots[pi]
                        even_col = 2 * pair_idx
                        odd_col = even_col + 1
                        src1 = self.calc_physical_coordinate(*prev_storage.get(q1, (0, 0, 0)))
                        src2 = self.calc_physical_coordinate(*prev_storage.get(q2, (0, 0, 0)))
                        dst_even = self.calc_physical_coordinate(1, row, even_col)
                        dst_odd = self.calc_physical_coordinate(1, row, odd_col)

                        cost_even_q1 = math.hypot(src1[0] - dst_even[0], src1[1] - dst_even[1]) + \
                                      math.hypot(src2[0] - dst_odd[0], src2[1] - dst_odd[1])
                        cost_even_q2 = math.hypot(src2[0] - dst_even[0], src2[1] - dst_even[1]) + \
                                      math.hypot(src1[0] - dst_odd[0], src1[1] - dst_odd[1])

                        if cost_even_q1 <= cost_even_q2:
                            loc[q1] = (row, even_col)
                            loc[q2] = (row, odd_col)
                        else:
                            loc[q1] = (row, odd_col)
                            loc[q2] = (row, even_col)

                        slot_parities[(row, pair_idx)] = {0, 1}
                    pair_ptr += to_place
                    if pair_ptr >= len(free_pairs):
                        break

                if pair_ptr < len(free_pairs):
                    raise RuntimeError(
                        f"Ran out of entangling slots while placing stage {stage_idx}"
                    )

            # ------------------------------------------------------------------
            # 3️⃣ Build pre‑pulse snapshot
            # ------------------------------------------------------------------
            pre_snapshot: list[tuple[int, int, int]] = [None] * n_q  # type: ignore
            for q in range(n_q):
                if q in loc:
                    pre_snapshot[q] = (1, loc[q][0], loc[q][1])
                else:
                    pre_snapshot[q] = storage_placement[q]

            # ------------------------------------------------------------------
            # 4️⃣ Build post‑pulse snapshot
            # ------------------------------------------------------------------
            post_snapshot: list[tuple[int, int, int]] = [None] * n_q  # type: ignore

            # 4a – reused qubits stay in entangling zone.
            for q in reuse_out:
                post_snapshot[q] = pre_snapshot[q]

            # 4b – idle qubits keep their storage cells.
            for q in range(n_q):
                if q not in active_this_stage:
                    post_snapshot[q] = storage_placement[q]

            # 4c – active non‑reused qubits move back to storage.
            non_reused_active = active_this_stage - reuse_out
            if non_reused_active:
                # group moving qubits by source entangling row (router‑friendly)
                groups_by_src_row: dict[int, list[int]] = {}
                for q in non_reused_active:
                    src_row = loc[q][0]
                    groups_by_src_row.setdefault(src_row, []).append(q)

                last_assigned_row = -1
                global_last_col = -1
                used_dest_rows: set[int] = set()

                for src_row in sorted(groups_by_src_row):
                    group = groups_by_src_row[src_row]
                    group.sort(key=lambda q: loc[q][1])  # sort by source column
                    cost_thresh = len(group) * self.COST_MULTIPLIER
                    chosen_row, new_global = self._assign_storage_for_group(
                        group=group,
                        loc=loc,
                        storage_placement=storage_placement,
                        occupied=occupied,
                        storage_rows=storage_rows,
                        storage_cols=storage_cols,
                        d_ec=d_ec,
                        d_sc=d_sc,
                        ryd=ryd,
                        L_ec=L_ec,
                        L_sc=L_sc,
                        last_assigned_row=last_assigned_row,
                        global_last_col=global_last_col,
                        cost_threshold=cost_thresh,
                        used_rows=used_dest_rows,
                    )
                    used_dest_rows.add(chosen_row)
                    last_assigned_row = chosen_row
                    if new_global != -1:
                        global_last_col = new_global

                # intra‑row swap optimisation (helps router compatibility & distance)
                self._optimize_storage_swaps(
                    moving_qubits=non_reused_active,
                    loc=loc,
                    storage_placement=storage_placement,
                )

                for q in non_reused_active:
                    post_snapshot[q] = storage_placement[q]

            placements_by_2q_stage.append([pre_snapshot, post_snapshot])

            # ------------------------------------------------------------------
            # 5️⃣ Update reused‑location map for the next stage
            # ------------------------------------------------------------------
            new_reused: dict[int, tuple[int, int]] = {}
            for q in reuse_out:
                cell = pre_snapshot[q]
                if cell[0] == 1:
                    new_reused[q] = (cell[1], cell[2])
            reused_loc = new_reused

        # ------------------------------------------------------------------
        # Ensure final layout ends in storage (required by evaluator)
        # ------------------------------------------------------------------
        if placements_by_2q_stage:
            last_snapshot = placements_by_2q_stage[-1][-1]
            if any(cell[0] != 0 for cell in last_snapshot):
                final_snapshot = last_snapshot.copy()
                free_cells = [
                    (0, r, c)
                    for r in range(storage_rows)
                    for c in range(storage_cols)
                    if (0, r, c) not in occupied
                ]
                free_cells.sort(key=lambda cell: (cell[1], cell[2]))
                free_iter = iter(free_cells)
                for q, cell in enumerate(last_snapshot):
                    if cell[0] != 0:
                        try:
                            new_cell = next(free_iter)
                        except StopIteration:
                            raise RuntimeError("Ran out of free storage cells for final placement")
                        final_snapshot[q] = new_cell
                        occupied.add(new_cell)
                placements_by_2q_stage[-1].append(final_snapshot)
                last_snapshot = final_snapshot
        else:
            last_snapshot = init.copy()

        # ------------------------------------------------------------------
        # Total Euclidean travel distance (used for tie‑breaker)
        # ------------------------------------------------------------------
        flat: list[list[tuple[int, int, int]]] = [init]
        for grp in placements_by_2q_stage:
            flat.extend(grp)

        total_dist = 0.0
        for i in range(len(flat) - 1):
            src = flat[i]
            dst = flat[i + 1]
            for q in range(n_q):
                if src[q] != dst[q]:
                    sx, sy = self.calc_physical_coordinate(*src[q])
                    dx, dy = self.calc_physical_coordinate(*dst[q])
                    total_dist += math.hypot(sx - dx, sy - dy)

        return placements_by_2q_stage, last_snapshot, total_dist

    # ------------------------------------------------------------------
    # Helper: build a storage‑only placement from a given ordering.
    # ------------------------------------------------------------------
    def _build_storage_placement(
        self,
        order: list[int],
        storage_rows: int,
        storage_cols: int,
    ) -> list[tuple[int, int, int]]:
        """Row‑major storage placement for the supplied qubit order."""
        if len(order) > storage_rows * storage_cols:
            raise RuntimeError("Not enough storage cells for all qubits")
        placement: list[tuple[int, int, int]] = [None] * len(order)  # type: ignore
        for pos, q in enumerate(order):
            r = pos // storage_cols
            c = pos % storage_cols
            placement[q] = (0, r, c)
        return placement

    # ------------------------------------------------------------------
    # Ordering heuristics
    # ------------------------------------------------------------------
    def _spectral_order(self, n_q: int, stages: list[ZNAAStage]) -> list[int]:
        """Return a qubit ordering given by the Fiedler vector of the interaction graph."""
        adj = np.zeros((n_q, n_q), dtype=float)
        for st in stages:
            if st.stage_type != "2q":
                continue
            for g in st.gates:
                a, b = g.qubits
                adj[a, b] += 1.0
                adj[b, a] += 1.0
        deg = np.sum(adj, axis=1)
        L = np.diag(deg) - adj
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return sorted(range(n_q), key=lambda q: -deg[q])
        idx = np.argsort(eigvals)
        if len(idx) < 2:
            return list(range(n_q))
        fiedler = eigvecs[:, idx[1]]
        return list(np.argsort(fiedler.real))

    def _centrality_order(self, n_q: int, stages: list[ZNAAStage]) -> list[int]:
        """Order qubits by decreasing weighted degree."""
        degree = [0] * n_q
        for st in stages:
            if st.stage_type != "2q":
                continue
            for g in st.gates:
                a, b = g.qubits
                degree[a] += 1
                degree[b] += 1
        return sorted(range(n_q), key=lambda q: (-degree[q], q))

    def _rcm_order(self, n_q: int, stages: list[ZNAAStage]) -> list[int]:
        """Return a bandwidth‑reduced ordering using reverse Cuthill‑McKee."""
        rows, cols, data = [], [], []
        for st in stages:
            if st.stage_type != "2q":
                continue
            for g in st.gates:
                a, b = g.qubits
                rows.append(a); cols.append(b); data.append(1)
                rows.append(b); cols.append(a); data.append(1)
        if not rows:
            return list(range(n_q))
        adj = coo_matrix((data, (rows, cols)), shape=(n_q, n_q))
        try:
            perm = reverse_cuthill_mckee(adj)
            return list(map(int, perm))
        except Exception:
            return list(range(n_q))

    def _greedy_cluster_order(self, n_q: int, stages: list[ZNAAStage]) -> list[int]:
        """Build an ordering by repeatedly adding the most strongly connected unused qubit."""
        adj: dict[int, dict[int, int]] = {q: {} for q in range(n_q)}
        for st in stages:
            if st.stage_type != "2q":
                continue
            for g in st.gates:
                a, b = g.qubits
                adj[a][b] = adj[a].get(b, 0) + 1
                adj[b][a] = adj[b].get(a, 0) + 1

        placed: list[int] = []
        unused = set(range(n_q))

        start = max(unused, key=lambda q: sum(adj[q].values()))
        placed.append(start)
        unused.remove(start)

        while unused:
            best_q = None
            best_w = -1
            for q in unused:
                w = sum(adj[q].get(p, 0) for p in placed)
                if w > best_w:
                    best_w = w
                    best_q = q
            if best_q is None:
                best_q = unused.pop()
            else:
                unused.remove(best_q)
            placed.append(best_q)

        return placed

    def _init_order(self, stages: list[ZNAAStage], n_q: int) -> list[int]:
        """Adjacency‑aware ordering based on the first non‑empty 2Q stage."""
        placed_set: set[int] = set()
        order: list[int] = []
        two_q_idxs = [
            i for i, st in enumerate(stages) if st.stage_type == "2q" and st.gates
        ]
        if two_q_idxs:
            first_stage = stages[two_q_idxs[0]]
            first_pairs = [(g.qubits[0], g.qubits[1]) for g in first_stage.gates]
            for a, b in first_pairs:
                if a not in placed_set:
                    order.append(a)
                    placed_set.add(a)
                if b not in placed_set:
                    order.append(b)
                    placed_set.add(b)
        remaining = [q for q in range(n_q) if q not in placed_set]
        order.extend(remaining)
        return order

    def _degree_order(self, n_q: int, degree: list[int], earliest: list[int]) -> list[int]:
        """Sort by decreasing degree, tie‑break by earliest appearance."""
        return sorted(range(n_q), key=lambda q: (-degree[q], earliest[q], q))

    # ------------------------------------------------------------------
    # Main entry point required by the pipeline.
    # ------------------------------------------------------------------
    def place(
        self,
        stages: list[ZNAAStage],
        reuse_info: list[list[int]],
    ) -> list[list[list[tuple[int, int, int]]]]:
        # ------------------------------------------------------------------
        # sanity checks
        # ------------------------------------------------------------------
        if len(stages) != len(reuse_info):
            raise RuntimeError(
                f"Length mismatch: {len(stages)} stages vs {len(reuse_info)} reuse rows"
            )

        # ------------------------------------------------------------------
        # determine number of logical qubits
        # ------------------------------------------------------------------
        max_q = -1
        for st in stages:
            for g in st.gates:
                for q in g.qubits:
                    max_q = max(max_q, q)
        n_q = max_q + 1 if max_q >= 0 else 0

        if n_q == 0:
            self.initial_placement = []
            return []

        storage_rows, storage_cols = self.config.storage_shape

        # ------------------------------------------------------------------
        # compute simple graph metrics (used by several orderings)
        # ------------------------------------------------------------------
        earliest_stage = [math.inf] * n_q
        last_stage = [-1] * n_q
        degree = [0] * n_q
        adjacency: dict[int, set[int]] = {q: set() for q in range(n_q)}
        reuse_counts = [0] * n_q

        for idx, st in enumerate(stages):
            for g in st.gates:
                for q in g.qubits:
                    if earliest_stage[q] == math.inf:
                        earliest_stage[q] = idx
                    last_stage[q] = max(last_stage[q], idx)
                    if st.stage_type == "2q":
                        degree[q] += 1
                if st.stage_type == "2q":
                    a, b = g.qubits
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        for row in reuse_info:
            for q in row:
                reuse_counts[q] += 1

        # ------------------------------------------------------------------
        # generate candidate orderings (moderate set)
        # ------------------------------------------------------------------
        init_order = self._init_order(stages, n_q) #qubit used in 1 stage is ealier
        degree_order = self._degree_order(n_q, degree, earliest_stage) # sort by (-degree, earlierest stage)
        centrality_order = self._centrality_order(n_q, stages) # sort by (-degree, qubit_id)
        greedy_cluster_order = self._greedy_cluster_order(n_q, stages) # sort by clustering
        rcm_order = self._rcm_order(n_q, stages) # there is  a try & except, seems the package is not imported correctly

        spectral_order: list[int] = []
        if n_q <= 80:
            try:
                spectral_order = self._spectral_order(n_q, stages) # fideler vector of the interaction graph
            except Exception:
                spectral_order = []

        base_candidates: list[list[int]] = [
            init_order,
            degree_order,
            centrality_order,
            greedy_cluster_order,
            rcm_order,
        ]
        if spectral_order:
            base_candidates.append(spectral_order)

        # a few deterministic random permutations
        rng = np.random.RandomState(1)
        random_orders: list[list[int]] = []
        for seed in self.RANDOM_ORDER_SEEDS:
            rng.seed(seed)
            ro = list(range(n_q))
            rng.shuffle(ro)
            random_orders.append(ro)

        base_candidates.extend(random_orders)

        # ------------------------------------------------------------------
        # expand with reversed versions (helps symmetry)
        # ------------------------------------------------------------------
        candidate_orders: list[list[int]] = []
        for cand in base_candidates:
            if cand:
                candidate_orders.append(cand)
                candidate_orders.append(list(reversed(cand)))

        # ------------------------------------------------------------------
        # evaluate candidates (using router to count move operations)
        # ------------------------------------------------------------------
        best_placements: list[list[list[tuple[int, int, int]]]] = []
        best_init: list[tuple[int, int, int]] = []
        best_move_ops = float("inf")
        best_dist = float("inf")
        best_final_snapshot: list[tuple[int, int, int]] = []

        router = Router(name="tmp", version="0.0", config_factory=lambda: self.config)

        for order in candidate_orders:
            init_pl = self._build_storage_placement(order, storage_rows, storage_cols)

            placements_fwd, final_fwd, dist_fwd = self._single_pass(
                stages, reuse_info, init_pl
            )

            # flatten for router evaluation
            seq: list[list[tuple[int, int, int]]] = [init_pl]
            for grp in placements_fwd:
                seq.extend(grp)

            try:
                segments = router.route(seq)
                move_ops = sum(
                    1 for seg in segments for op in seg
                    if getattr(op, "type_name", None) == "move"
                )
            except Exception:
                move_ops = float("inf")

            if move_ops < best_move_ops or (move_ops == best_move_ops and dist_fwd < best_dist):
                best_move_ops = move_ops
                best_dist = dist_fwd
                best_placements = placements_fwd
                best_init = init_pl
                best_final_snapshot = final_fwd

            if best_move_ops == 0:
                break  # perfect placement found

        # ------------------------------------------------------------------
        # Reverse‑through‑time (RTT) refinement – limited rounds
        # ------------------------------------------------------------------
        if best_placements:
            for _ in range(self.RTT_ROUNDS):
                rev_stages = list(reversed(stages))
                rev_reuse = [[] for _ in rev_stages]

                _, rev_final, _ = self._single_pass(rev_stages, rev_reuse, best_final_snapshot)

                placements_fwd2, final_fwd2, dist_fwd2 = self._single_pass(
                    stages, reuse_info, rev_final
                )
                seq2: list[list[tuple[int, int, int]]] = [rev_final]
                for grp in placements_fwd2:
                    seq2.extend(grp)

                try:
                    segments2 = router.route(seq2)
                    move_ops2 = sum(
                        1 for seg in segments2 for op in seg
                        if getattr(op, "type_name", None) == "move"
                    )
                except Exception:
                    move_ops2 = float("inf")

                if move_ops2 < best_move_ops or (move_ops2 == best_move_ops and dist_fwd2 < best_dist):
                    best_move_ops = move_ops2
                    best_dist = dist_fwd2
                    best_placements = placements_fwd2
                    best_init = rev_final
                    best_final_snapshot = final_fwd2
                else:
                    break  # no further improvement

        # ------------------------------------------------------------------
        # Hill‑climbing post‑processing (swap two qubits in the initial layout)
        # ------------------------------------------------------------------
        if best_placements:
            hill_rng = np.random.RandomState(42)
            for _ in range(self.HILL_CLIMB_ATTEMPTS):
                i, j = hill_rng.choice(n_q, size=2, replace=False)
                new_init = best_init.copy()
                new_init[i], new_init[j] = new_init[j], new_init[i]

                placements_fwd, final_fwd, dist_fwd = self._single_pass(
                    stages, reuse_info, new_init
                )
                seq: list[list[tuple[int, int, int]]] = [new_init]
                for grp in placements_fwd:
                    seq.extend(grp)

                try:
                    segments = router.route(seq)
                    move_ops = sum(
                        1 for seg in segments for op in seg
                        if getattr(op, "type_name", None) == "move"
                    )
                except Exception:
                    move_ops = float("inf")

                if move_ops < best_move_ops or (move_ops == best_move_ops and dist_fwd < best_dist):
                    best_move_ops = move_ops
                    best_dist = dist_fwd
                    best_placements = placements_fwd
                    best_init = new_init
                    best_final_snapshot = final_fwd

        # ------------------------------------------------------------------
        # If there are no 2‑Q stages we simply return an empty placement list.
        # ------------------------------------------------------------------
        if not best_placements:
            self.initial_placement = self._build_storage_placement(
                list(range(n_q)), storage_rows, storage_cols
            )
            return []

        self.initial_placement = best_init
        return best_placements
# EVOLVE-BLOCK-END

def run_code() -> dict[str, Any]:
    return {
        "scheduler_class": Scheduler,
        "reuse_analyzer_class": ReuseAnalyzer,
        "placer_class": Placer,
        "router_class": Router,
    }