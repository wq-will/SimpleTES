use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use ndarray::prelude::*;

struct SubsetResult {
    count: usize,
    error: f64,
    map: Vec<usize>,
    subgraph: Vec<[usize; 2]>,
    index: usize,
}

fn bfs_sort(adj_matrix: ArrayView2<f64>, start: usize, num_qubits: usize) -> Vec<usize> {
    let n = adj_matrix.shape()[0];
    let mut next_level: IndexSet<usize, ahash::RandomState> =
        IndexSet::with_hasher(ahash::RandomState::default());
    let mut bfs_order = Vec::with_capacity(num_qubits);
    let mut seen: HashSet<usize> = HashSet::with_capacity(n);
    next_level.insert(start);
    while !next_level.is_empty() {
        let this_level = next_level;
        next_level = IndexSet::with_hasher(ahash::RandomState::default());
        let mut found: Vec<usize> = Vec::new();
        for v in this_level {
            if !seen.contains(&v) {
                seen.insert(v);
                found.push(v);
                bfs_order.push(v);
                if bfs_order.len() == num_qubits {
                    return bfs_order;
                }
            }
        }
        if seen.len() == n {
            return bfs_order;
        }
        for node in found {
            for (idx, v) in adj_matrix.index_axis(Axis(0), node).iter().enumerate() {
                if *v != 0.0 {
                    next_level.insert(idx);
                }
            }
            for (idx, v) in adj_matrix.index_axis(Axis(1), node).iter().enumerate() {
                if *v != 0.0 {
                    next_level.insert(idx);
                }
            }
        }
    }
    bfs_order
}

pub fn best_subset(
    num_qubits: usize,
    coupling_adj_mat: ArrayView2<f64>,
    num_meas: usize,
    num_cx: usize,
    use_error: bool,
    symmetric_coupling_map: bool,
    err: ArrayView2<f64>,
) -> [Vec<usize>; 3] {
    let coupling_shape = coupling_adj_mat.shape();
    let avg_meas_err = err.diag().mean().unwrap_or(0.0);

    let map_fn = |k| -> SubsetResult {
        let mut subgraph: Vec<[usize; 2]> = Vec::with_capacity(num_qubits);
        let bfs = bfs_sort(coupling_adj_mat, k, num_qubits);
        let bfs_set: HashSet<usize> = bfs.iter().copied().collect();
        let mut connection_count = 0;
        for node_idx in &bfs {
            coupling_adj_mat
                .index_axis(Axis(0), *node_idx)
                .into_iter()
                .enumerate()
                .filter_map(|(node, j)| {
                    if *j != 0.0 && bfs_set.contains(&node) {
                        Some(node)
                    } else {
                        None
                    }
                })
                .for_each(|node| {
                    connection_count += 1;
                    subgraph.push([*node_idx, node]);
                });
        }
        let error = if use_error {
            let mut ret_error = 0.0;
            let meas_avg = bfs.iter().map(|i| err[[*i, *i]]).sum::<f64>() / num_qubits as f64;
            let meas_diff = meas_avg - avg_meas_err;
            if meas_diff > 0.0 {
                ret_error += num_meas as f64 * meas_diff;
            }
            let cx_sum: f64 = subgraph.iter().map(|edge| err[[edge[0], edge[1]]]).sum();
            let mut cx_err = if subgraph.is_empty() {
                0.0
            } else {
                cx_sum / subgraph.len() as f64
            };
            if symmetric_coupling_map {
                cx_err /= 2.0;
            }
            ret_error += num_cx as f64 * cx_err;
            ret_error
        } else {
            0.0
        };
        SubsetResult {
            count: connection_count,
            error,
            map: bfs,
            subgraph,
            index: k,
        }
    };

    let reduce_fn = |best: SubsetResult, curr: SubsetResult| -> SubsetResult {
        if use_error {
            if (curr.count >= best.count && curr.error < best.error)
                || (curr.count == best.count && curr.error == best.error && curr.index < best.index)
            {
                curr
            } else {
                best
            }
        } else if curr.count > best.count || (curr.count == best.count && curr.index < best.index) {
            curr
        } else {
            best
        }
    };

    let best_result = (0..coupling_shape[0])
        .map(map_fn)
        .reduce(reduce_fn)
        .unwrap_or(SubsetResult {
            count: 0,
            map: Vec::new(),
            error: f64::INFINITY,
            subgraph: Vec::new(),
            index: usize::MAX,
        });
    let best_map: Vec<usize> = best_result.map;
    let mapping: HashMap<usize, usize> = best_map
        .iter()
        .enumerate()
        .map(|(best_edge, edge)| (*edge, best_edge))
        .collect();
    let new_cmap: Vec<[usize; 2]> = best_result
        .subgraph
        .iter()
        .map(|c| [mapping[&c[0]], mapping[&c[1]]])
        .collect();
    let rows: Vec<usize> = new_cmap.iter().map(|edge| edge[0]).collect();
    let cols: Vec<usize> = new_cmap.iter().map(|edge| edge[1]).collect();

    [rows, cols, best_map]
}
