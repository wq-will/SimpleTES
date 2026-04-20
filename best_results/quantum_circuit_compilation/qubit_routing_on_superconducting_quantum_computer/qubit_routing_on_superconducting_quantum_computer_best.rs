// EVOLVE-BLOCK-START

use std::cmp::{max, min};

/// Scaling mode for heuristic weights (kept for compatibility with the engine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetScaling {
    Constant,
    Size,
}

/// Representation of the current front layer.
#[derive(Debug, Clone, Default)]
struct FrontLayerScores {
    nodes: Vec<[usize; 2]>,
    qubits: Vec<Option<(usize, usize)>>,
}

impl FrontLayerScores {
    fn from_ctx(ctx: &SwapSelectionContext<'_>) -> Self {
        let mut out = Self {
            nodes: Vec::new(),
            qubits: vec![None; ctx.topology().num_qubits()],
        };
        for pair in ctx.front_layer().physical_pairs() {
            let [a, b] = *pair;
            let idx = out.nodes.len();
            out.nodes.push([a, b]);
            out.qubits[a] = Some((idx, b));
            out.qubits[b] = Some((idx, a));
        }
        out
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn is_active(&self, q: usize) -> bool {
        self.qubits[q].is_some()
    }

    fn iter_active(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes.iter().flatten().cloned()
    }

    fn total_score(&self, topo: TopologyView<'_>) -> f64 {
        self.nodes
            .iter()
            .map(|pair| topo.distance(pair[0], pair[1]) as f64)
            .sum()
    }

    fn max_distance(&self, topo: TopologyView<'_>) -> u32 {
        self.nodes
            .iter()
            .map(|pair| topo.distance(pair[0], pair[1]))
            .max()
            .unwrap_or(0)
    }

    /// Change in the (un‑weighted) sum of distances caused by applying `swap`.
    fn score_delta(&self, swap: (usize, usize), topo: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut delta = 0.0;
        if let Some((_, c)) = self.qubits[a] {
            delta += (topo.distance(b, c) as f64) - (topo.distance(a, c) as f64);
        }
        if let Some((_, c)) = self.qubits[b] {
            delta += (topo.distance(a, c) as f64) - (topo.distance(b, c) as f64);
        }
        delta
    }
}

/// Scores for the look‑ahead set.
#[derive(Debug, Clone)]
struct ExtendedSetScores {
    pairs: Vec<(usize, usize, f64)>, // (phys_a, phys_b, weight)
    len: usize,
}

impl ExtendedSetScores {
    fn new() -> Self {
        Self {
            pairs: Vec::new(),
            len: 0,
        }
    }

    fn push(&mut self, a: usize, b: usize) {
        // Exponential decay weight: 0.5ⁱ
        let weight = 0.5_f64.powi(self.len as i32);
        self.pairs.push((a, b, weight));
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn total_score(&self, topo: TopologyView<'_>) -> f64 {
        self.pairs
            .iter()
            .map(|&(a, b, w)| w * topo.distance(a, b) as f64)
            .sum()
    }

    /// Weighted change in the look‑ahead distance sum caused by `swap`.
    fn score_delta(&self, swap: (usize, usize), topo: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for &(p, q, w) in &self.pairs {
            // the swapped edge itself does not change its own distance
            if (p == a && q == b) || (p == b && q == a) {
                continue;
            }
            if p == a {
                total += w * ((topo.distance(b, q) as f64) - (topo.distance(p, q) as f64));
            } else if q == a {
                total += w * ((topo.distance(b, p) as f64) - (topo.distance(p, q) as f64));
            } else if p == b {
                total += w * ((topo.distance(a, q) as f64) - (topo.distance(p, q) as f64));
            } else if q == b {
                total += w * ((topo.distance(a, p) as f64) - (topo.distance(p, q) as f64));
            }
        }
        total
    }
}

/// Build the look‑ahead set (up to `max_size` gates).  The set is built by
/// a BFS from the current front layer following DAG dependencies.
fn build_extended_set(ctx: &SwapSelectionContext<'_>, max_size: usize) -> ExtendedSetScores {
    let mut out = ExtendedSetScores::new();
    if max_size == 0 {
        return out;
    }

    // Prefer a pre‑computed set supplied by the engine.
    let pre = ctx.precomputed_extended_set_logical_pairs();
    if !pre.is_empty() {
        for pair in pre.iter().take(max_size) {
            out.push(
                ctx.layout().physical_of_logical(pair[0]),
                ctx.layout().physical_of_logical(pair[1]),
            );
        }
        return out;
    }

    // BFS from the front layer.
    let mut remaining_pred = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut to_visit = ctx.front_layer().node_ids().to_vec();
    let mut i = 0usize;
    while i < to_visit.len() && out.len() < max_size {
        let nid = to_visit[i];
        for &succ in ctx.circuit().node(nid).successors() {
            remaining_pred[succ] -= 1;
            if remaining_pred[succ] == 0 {
                if let Some((a, b)) = ctx.circuit().node(succ).two_qubit_pair() {
                    out.push(
                        ctx.layout().physical_of_logical(a),
                        ctx.layout().physical_of_logical(b),
                    );
                }
                to_visit.push(succ);
            }
        }
        i += 1;
    }
    out
}

/// Enumerate candidate SWAPs:
///   – any edge incident to a qubit that appears in the front layer **or**
///     in the look‑ahead set,
///   – plus all edges that belong to the shortest paths of **all** front‑layer
///     pairs,
///   – plus all edges that belong to the shortest paths of **all** look‑ahead
///     pairs.
fn enumerate_candidate_swaps_extended(
    topo: TopologyView<'_>,
    front: &FrontLayerScores,
    ext: &ExtendedSetScores,
) -> Vec<(usize, usize)> {
    let mut relevant = HashSet::<usize>::new();

    // qubits that appear in the front layer
    for q in front.iter_active() {
        relevant.insert(q);
    }
    // qubits that appear in the look‑ahead set
    for &(a, b, _) in &ext.pairs {
        relevant.insert(a);
        relevant.insert(b);
    }

    // incident edges
    let mut edges = HashSet::<(usize, usize)>::new();
    for &u in &relevant {
        for &v in topo.neighbors(u) {
            let (x, y) = if u < v { (u, v) } else { (v, u) };
            edges.insert((x, y));
        }
    }

    // edges on shortest paths of front‑layer pairs
    for pair in front.nodes.iter() {
        let (a, b) = (pair[0], pair[1]);
        if a == b {
            continue;
        }
        let path = topo.shortest_path(a, b);
        if path.len() >= 2 {
            for win in path.windows(2) {
                let u = win[0];
                let v = win[1];
                let (x, y) = if u < v { (u, v) } else { (v, u) };
                edges.insert((x, y));
            }
        }
    }

    // edges on shortest paths of look‑ahead pairs
    for &(a, b, _) in &ext.pairs {
        if a == b {
            continue;
        }
        let path = topo.shortest_path(a, b);
        if path.len() >= 2 {
            for win in path.windows(2) {
                let u = win[0];
                let v = win[1];
                let (x, y) = if u < v { (u, v) } else { (v, u) };
                edges.insert((x, y));
            }
        }
    }

    edges.into_iter().collect()
}

/// Choose a dense, connected subset of a target component for a logical component.
fn choose_dense_layout_subset(
    topo: TopologyView<'_>,
    logical_size: usize,
    target: &[usize],
) -> Result<Vec<usize>, RouterError> {
    if logical_size > target.len() {
        return Err(RouterError::Routing(format!(
            "logical component size {logical_size} exceeds target component size {}",
            target.len()
        )));
    }
    if logical_size == target.len() {
        return Ok(target.to_vec());
    }

    // adjacency matrix of the target component
    let local_index = target
        .iter()
        .enumerate()
        .map(|(i, &q)| (q, i))
        .collect::<HashMap<usize, usize>>();
    let mut adj = Array2::<f64>::zeros((target.len(), target.len()));
    for &g_a in target {
        let a = local_index[&g_a];
        for &g_b in topo.neighbors(g_a) {
            if let Some(&b) = local_index.get(&g_b) {
                adj[[a, b]] = 1.0;
            }
        }
    }

    let err = Array2::<f64>::zeros((target.len(), target.len()));
    let [_, _, best_map] = dense_layout::best_subset(
        logical_size,
        adj.view(),
        0,
        0,
        false,
        true,
        err.view(),
    );
    let chosen = best_map
        .into_iter()
        .take(logical_size)
        .map(|local| target[local])
        .collect::<Vec<_>>();
    ensure_connected_subset(topo, &chosen)?;
    Ok(chosen)
}

/// Verify that `subset` induces a connected subgraph of the topology.
fn ensure_connected_subset(topo: TopologyView<'_>, subset: &[usize]) -> Result<(), RouterError> {
    if subset.is_empty() {
        return Ok(());
    }
    let set = subset.iter().copied().collect::<HashSet<_>>();
    let mut seen = HashSet::<usize>::new();
    let mut q = VecDeque::new();
    q.push_back(subset[0]);
    seen.insert(subset[0]);

    while let Some(v) = q.pop_front() {
        for &nbr in topo.neighbors(v) {
            if set.contains(&nbr) && seen.insert(nbr) {
                q.push_back(nbr);
            }
        }
    }

    if seen.len() != set.len() {
        Err(RouterError::Routing(
            "selected layout subset is not connected".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Greedy assignment of logical interaction components to physical components.
fn assign_components_to_target(
    logical: &[Vec<usize>],
    physical: &[Vec<usize>],
) -> Result<Vec<(Vec<usize>, usize)>, RouterError> {
    if logical.is_empty() {
        return Ok(Vec::new());
    }

    // sort logical components by decreasing size
    let mut logical_sorted = logical.to_vec();
    logical_sorted.sort_by_key(|c| std::cmp::Reverse(c.len()));

    // sort physical components by decreasing size
    let mut phys_sorted = physical
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.len()))
        .collect::<Vec<_>>();
    phys_sorted.sort_by_key(|(_, sz)| std::cmp::Reverse(*sz));

    // keep track of remaining capacity of each physical component
    let mut capacity = phys_sorted
        .iter()
        .map(|(i, sz)| (*i, *sz))
        .collect::<HashMap<usize, usize>>();
    let mut assignments = Vec::<(Vec<usize>, usize)>::new();

    for comp in logical_sorted {
        let sz = comp.len();
        let mut chosen = None;
        for (pidx, _) in &phys_sorted {
            if capacity.get(pidx).copied().unwrap_or(0) >= sz {
                chosen = Some(*pidx);
                break;
            }
        }
        let pidx = chosen.ok_or_else(|| {
            RouterError::Routing(format!(
                "logical component of size {sz} cannot fit any target component"
            ))
        })?;
        *capacity.get_mut(&pidx).unwrap() -= sz;
        assignments.push((comp, pidx));
    }
    Ok(assignments)
}

/// Logical adjacency and per‑qubit interaction degrees.
fn compute_logical_adjacency(
    ctx: &InitialLayoutContext<'_>,
) -> (Vec<HashMap<usize, usize>>, Vec<usize>) {
    let circuit = ctx.circuit();
    let n = circuit.num_logical_qubits();
    let mut adj = vec![HashMap::<usize, usize>::new(); n];
    for nid in circuit.node_ids() {
        if let Some((a, b)) = circuit.node(nid).two_qubit_pair() {
            *adj[a].entry(b).or_insert(0) += 1;
            *adj[b].entry(a).or_insert(0) += 1;
        }
    }
    let deg = adj.iter().map(|m| m.values().sum()).collect();
    (adj, deg)
}

/// Greedy embedding of a logical component onto a physical component.
/// The seed logical qubit is the highest‑degree logical; the seed physical qubit
/// is the most central (minimum sum of distances) among the available qubits,
/// breaking ties toward higher physical degree.
fn embed_component(
    logical: &[usize],
    phys: &[usize],
    adj: &[HashMap<usize, usize>],
    deg: &[usize],
    distances: &[Vec<u32>],
    topo: TopologyView<'_>,
) -> Result<Vec<(usize, usize)>, RouterError> {
    let mut available: HashSet<usize> = phys.iter().copied().collect();
    let mut placed: HashMap<usize, usize> = HashMap::new();

    // seed logical – highest degree
    let seed_logical = *logical
        .iter()
        .max_by_key(|&&l| deg[l])
        .ok_or_else(|| RouterError::Routing("empty logical component".to_string()))?;

    // seed physical – most central, tie‑break on degree
    let seed_physical = *available
        .iter()
        .min_by_key(|&&p| {
            let sum: u64 = distances[p].iter().map(|&d| d as u64).sum();
            let deg = topo.degree(p) as u64;
            (sum, std::u64::MAX - deg) // smaller sum, larger degree preferred
        })
        .ok_or_else(|| RouterError::Routing("empty physical component".to_string()))?;

    placed.insert(seed_logical, seed_physical);
    available.remove(&seed_physical);

    // place remaining logical qubits
    while placed.len() < logical.len() {
        // pick logical qubit with strongest total weight to already placed neighbours
        let mut best_logical = None;
        let mut best_weight = 0usize;
        for &l in logical {
            if placed.contains_key(&l) {
                continue;
            }
            let w: usize = adj[l]
                .iter()
                .filter_map(|(&nbr, &wt)| if placed.contains_key(&nbr) { Some(wt) } else { None })
                .sum();
            if w > best_weight {
                best_weight = w;
                best_logical = Some(l);
            }
        }
        let l = best_logical.unwrap_or_else(|| {
            *logical.iter().find(|&&x| !placed.contains_key(&x)).unwrap()
        });

        // choose physical qubit minimizing weighted distance to already placed neighbours
        let mut best_physical = None;
        let mut best_cost = u64::MAX;
        for &p in &available {
            let mut cost = 0u64;
            for (&nbr, &wt) in adj[l].iter() {
                if let Some(&p_nbr) = placed.get(&nbr) {
                    let d = distances[p][p_nbr] as u64;
                    cost += d * wt as u64;
                }
            }
            if cost < best_cost {
                best_cost = cost;
                best_physical = Some(p);
            }
        }
        let p = best_physical.ok_or_else(|| {
            RouterError::Routing("failed to find a physical qubit for embedding".to_string())
        })?;
        placed.insert(l, p);
        available.remove(&p);
    }

    Ok(placed.into_iter().collect())
}

/// Build an initial layout that respects connected components and places high‑degree
/// logical qubits onto central, high‑degree physical qubits.
fn choose_disjoint_aware_layout(ctx: &InitialLayoutContext<'_>) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topo = ctx.topology();
    let n_logical = circuit.num_logical_qubits();

    // logical adjacency & degrees
    let (logical_adj, logical_deg) = compute_logical_adjacency(ctx);

    // logical interaction components
    let logical_comps = circuit.logical_interaction_components();

    // physical connected components
    let phys_comps = topo.connected_components();
    if phys_comps.is_empty() {
        return Err(RouterError::Routing(
            "topology has no connected components".to_string(),
        ));
    }

    // assign each logical component to a fitting physical component
    let assignments = assign_components_to_target(&logical_comps, &phys_comps)?;

    // final mapping vector (logical → physical)
    let mut mapping = vec![usize::MAX; n_logical];
    let distances = topo.distances();

    // embed each component
    for (log_comp, phys_idx) in assignments {
        let phys_comp = &phys_comps[phys_idx];
        let embed = embed_component(
            &log_comp,
            phys_comp,
            &logical_adj,
            &logical_deg,
            distances,
            topo,
        )?;
        for (l, p) in embed {
            mapping[l] = p;
        }
    }

    // fill any unmapped logical qubits with any free physical qubits
    let mut used = vec![false; topo.num_qubits()];
    for &p in &mapping {
        if p != usize::MAX {
            used[p] = true;
        }
    }
    let mut free = (0..topo.num_qubits()).filter(|q| !used[*q]);
    for slot in &mut mapping {
        if *slot == usize::MAX {
            *slot = free
                .next()
                .ok_or_else(|| RouterError::Routing("not enough physical qubits".to_string()))?;
        }
    }
    Ok(mapping)
}

/// Generate a completely random injective layout.
fn random_initial_layout(
    ctx: &InitialLayoutContext<'_>,
    rng: &mut RngState,
) -> Result<Vec<usize>, RouterError> {
    let n_logical = ctx.circuit().num_logical_qubits();
    let n_physical = ctx.topology().num_qubits();
    if n_physical < n_logical {
        return Err(RouterError::Routing(
            "not enough physical qubits for random layout".to_string(),
        ));
    }
    let mut phys: Vec<usize> = (0..n_physical).collect();
    rng.shuffle_slice(&mut phys);
    Ok(phys.into_iter().take(n_logical).collect())
}

/// Cheap quality metric for a layout: sum of distances for all two‑qubit gates.
fn layout_score(layout: &[usize], circuit: CircuitDagView<'_>, topo: TopologyView<'_>) -> u64 {
    let mut sum = 0u64;
    for nid in circuit.node_ids() {
        if let Some((a, b)) = circuit.node(nid).two_qubit_pair() {
            let pa = layout[a];
            let pb = layout[b];
            sum += topo.distance(pa, pb) as u64;
        }
    }
    sum
}

/// Compute the set of free physical qubits for a given layout.
fn compute_free_set(layout: &[usize], total_physical: usize) -> Vec<usize> {
    let mut used = vec![false; total_physical];
    for &p in layout {
        used[p] = true;
    }
    (0..total_physical).filter(|p| !used[*p]).collect()
}

/// Policy implementation.
#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    // heuristic weights
    pub basic_weight: f64,
    pub lookahead_weight: f64, // overridden dynamically
    pub lookahead_size: usize,
    pub set_scaling: SetScaling,
    // decay parameters
    pub use_decay: bool,
    pub decay_increment: f64,
    pub decay_reset: usize,
    pub best_epsilon: f64,
    decay_state: Vec<f64>,
    // cached logical degrees (filled once per routing run)
    logical_degrees: Vec<usize>,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            basic_weight: 1.0,
            // we want full look‑ahead contribution
            lookahead_weight: 1.0,
            lookahead_size: 70, // placeholder – real value comes from `compute_dynamic_params`
            set_scaling: SetScaling::Constant,
            use_decay: true,
            // stronger penalty for repeatedly swapping the same qubits
            decay_increment: 0.03,
            decay_reset: 5,
            best_epsilon: 1e-10,
            decay_state: Vec::new(),
            logical_degrees: Vec::new(),
        }
    }
}

impl CandidatePolicy {
    /// Initialise / refresh the decay table.
    fn refresh_decay_state(&mut self, ctx: &SwapSelectionContext<'_>) {
        if !self.use_decay {
            return;
        }
        let n = ctx.topology().num_qubits();
        if self.decay_state.len() != n {
            self.decay_state = vec![1.0; n];
        }
        if ctx.swaps_since_progress() == 0 {
            self.decay_state.fill(1.0);
            return;
        }
        if let Some((a, b)) = ctx.last_applied_swap() {
            if ctx.swaps_since_progress() % self.decay_reset == 0 {
                self.decay_state.fill(1.0);
            } else {
                self.decay_state[a] += self.decay_increment;
                self.decay_state[b] += self.decay_increment;
            }
        }
    }

    /// Dynamically adapt heuristic parameters based on topology density and the
    /// number of remaining two‑qubit gates.
    fn compute_dynamic_params(
        &self,
        ctx: &SwapSelectionContext<'_>,
    ) -> (f64, f64, usize, SetScaling) {
        let topo = ctx.topology();
        let n = topo.num_qubits();
        let e = topo.num_edges();
        let avg_deg = if n > 0 { (2 * e) as f64 / n as f64 } else { 0.0 };

        // base horizon depending on density
        let base_horizon = if avg_deg >= 3.5 {
            250
        } else if avg_deg >= 2.5 {
            300
        } else {
            400
        };

        // scale with remaining two‑qubit gates
        let remaining = ctx.remaining().remaining_two_qubit_node_ids().len();
        let horizon = min(base_horizon + remaining / 20, 1000);

        // we keep the look‑ahead weight at full strength
        let lookahead_w = 1.0;

        (self.basic_weight, lookahead_w, horizon, SetScaling::Constant)
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        rng: &mut RngState,
    ) -> Result<Vec<usize>, RouterError> {
        // deterministic greedy layout
        let mut best_layout = choose_disjoint_aware_layout(ctx)?;
        let mut best_score = layout_score(&best_layout, ctx.circuit(), ctx.topology());

        // cache logical degrees for later use (optional, currently unused)
        let (_, deg) = compute_logical_adjacency(ctx);
        self.logical_degrees = deg;

        let trials = 8_000; // hill‑climbing steps per layout
        let n_logical = ctx.circuit().num_logical_qubits();
        let n_physical = ctx.topology().num_qubits();

        // free‑slot set for the current layout
        let mut free_set = compute_free_set(&best_layout, n_physical);

        // hill‑climbing on the deterministic layout
        if n_logical > 1 {
            for _ in 0..trials {
                if !free_set.is_empty() && rng.gen_f64() < 0.5 {
                    // move a logical qubit into a free slot
                    let i = rng.gen_index(n_logical);
                    let f_idx = rng.gen_index(free_set.len());
                    let f = free_set[f_idx];
                    let mut cand = best_layout.clone();
                    let old = cand[i];
                    cand[i] = f;
                    let sc = layout_score(&cand, ctx.circuit(), ctx.topology());
                    if sc < best_score {
                        best_score = sc;
                        best_layout = cand;
                        free_set[f_idx] = old;
                    }
                } else {
                    // swap two logical qubits
                    let i = rng.gen_index(n_logical);
                    let mut j = rng.gen_index(n_logical);
                    while j == i {
                        j = rng.gen_index(n_logical);
                    }
                    let mut cand = best_layout.clone();
                    cand.swap(i, j);
                    let sc = layout_score(&cand, ctx.circuit(), ctx.topology());
                    if sc < best_score {
                        best_score = sc;
                        best_layout = cand;
                    }
                }
            }
        }

        // helper: random layout + hill‑climbing
        let mut random_hc = |rng: &mut RngState| -> Result<(Vec<usize>, u64), RouterError> {
            let mut layout = random_initial_layout(ctx, rng)?;
            let mut score = layout_score(&layout, ctx.circuit(), ctx.topology());
            let mut free = compute_free_set(&layout, n_physical);
            if n_logical > 1 {
                for _ in 0..trials {
                    if !free.is_empty() && rng.gen_f64() < 0.5 {
                        let i = rng.gen_index(n_logical);
                        let f_idx = rng.gen_index(free.len());
                        let f = free[f_idx];
                        let mut cand = layout.clone();
                        let old = cand[i];
                        cand[i] = f;
                        let sc = layout_score(&cand, ctx.circuit(), ctx.topology());
                        if sc < score {
                            score = sc;
                            layout = cand;
                            free[f_idx] = old;
                        }
                    } else {
                        let i = rng.gen_index(n_logical);
                        let mut j = rng.gen_index(n_logical);
                        while j == i {
                            j = rng.gen_index(n_logical);
                        }
                        let mut cand = layout.clone();
                        cand.swap(i, j);
                        let sc = layout_score(&cand, ctx.circuit(), ctx.topology());
                        if sc < score {
                            score = sc;
                            layout = cand;
                        }
                    }
                }
            }
            Ok((layout, score))
        };

        // collect candidates: deterministic layout + many random restarts
        let mut candidates = Vec::with_capacity(32);
        candidates.push((best_layout.clone(), best_score));

        for _ in 0..30 {
            let (layout, score) = random_hc(rng)?;
            candidates.push((layout, score));
        }

        // keep the best candidate
        let (mut chosen_layout, mut chosen_score) = candidates
            .into_iter()
            .min_by_key(|(_, s)| *s)
            .unwrap();

        // final refinement pass (more aggressive than baseline)
        let final_trials = 8_000;
        if n_logical > 1 {
            let mut free = compute_free_set(&chosen_layout, n_physical);
            for _ in 0..final_trials {
                if !free.is_empty() && rng.gen_f64() < 0.5 {
                    let i = rng.gen_index(n_logical);
                    let f_idx = rng.gen_index(free.len());
                    let f = free[f_idx];
                    let mut cand = chosen_layout.clone();
                    let old = cand[i];
                    cand[i] = f;
                    let sc = layout_score(&cand, ctx.circuit(), ctx.topology());
                    if sc < chosen_score {
                        chosen_score = sc;
                        chosen_layout = cand;
                        free[f_idx] = old;
                    }
                } else {
                    let i = rng.gen_index(n_logical);
                    let mut j = rng.gen_index(n_logical);
                    while j == i {
                        j = rng.gen_index(n_logical);
                    }
                    let mut cand = chosen_layout.clone();
                    cand.swap(i, j);
                    let sc = layout_score(&cand, ctx.circuit(), ctx.topology());
                    if sc < chosen_score {
                        chosen_score = sc;
                        chosen_layout = cand;
                    }
                }
            }
        }

        // deterministic (small circuits) / random (large circuits) pair‑wise local search
        if n_logical <= 200 {
            // up to 45 full passes
            let max_passes = 45;
            for _ in 0..max_passes {
                let mut improved = false;
                for i in 0..n_logical {
                    for j in (i + 1)..n_logical {
                        let mut cand = chosen_layout.clone();
                        cand.swap(i, j);
                        let sc = layout_score(&cand, ctx.circuit(), ctx.topology());
                        if sc < chosen_score {
                            chosen_score = sc;
                            chosen_layout = cand;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        } else {
            // five short random passes
            for _ in 0..5 {
                let mut improved = false;
                for _ in 0..n_logical {
                    let i = rng.gen_index(n_logical);
                    let mut j = rng.gen_index(n_logical);
                    while j == i {
                        j = rng.gen_index(n_logical);
                    }
                    let mut cand = chosen_layout.clone();
                    cand.swap(i, j);
                    let sc = layout_score(&cand, ctx.circuit(), ctx.topology());
                    if sc < chosen_score {
                        chosen_score = sc;
                        chosen_layout = cand;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        Ok(chosen_layout)
    }

    fn choose_best_swap(
        &mut self,
        ctx: &SwapSelectionContext<'_>,
        rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        // update decay penalties
        self.refresh_decay_state(ctx);

        // dynamic parameters
        let (basic_w, lookahead_w, lookahead_sz_base, _) = self.compute_dynamic_params(ctx);
        let lookahead_sz = lookahead_sz_base; // already limited in `build_extended_set`

        // front layer and look‑ahead structures
        let front = FrontLayerScores::from_ctx(ctx);
        let ext = build_extended_set(ctx, lookahead_sz);
        let candidates = enumerate_candidate_swaps_extended(ctx.topology(), &front, &ext);
        if candidates.is_empty() {
            return None;
        }

        // quick lookup for look‑ahead‑only qubits
        let mut ext_qubits = HashSet::<usize>::new();
        for &(a, b, _) in &ext.pairs {
            ext_qubits.insert(a);
            ext_qubits.insert(b);
        }

        // base (no‑swap) cost
        let mut base_score = basic_w * front.total_score(ctx.topology());
        if !ext.is_empty() && lookahead_w != 0.0 {
            base_score += lookahead_w * ext.total_score(ctx.topology());
        }

        // pre‑compute maximal front‑layer distance
        let max_before = front.max_distance(ctx.topology());

        // heuristic constants (tuned for higher score)
        let immediate_gain_factor = 2.0;          // stronger saved‑distance bonus
        let max_dist_reduction_factor = 3.0;     // larger reward for shrinking max distance
        let max_dist_increase_factor = 0.1;      // mild penalty for increasing max distance
        let active_swap_penalty = 0.03;          // unchanged
        let ext_swap_penalty = 0.015;            // unchanged
        let idle_swap_penalty = 0.005;           // unchanged
        let undo_swap_penalty = 2.5;             // unchanged

        // evaluate candidates
        let mut best_score = f64::INFINITY;
        let mut best_swaps = Vec::<(usize, usize)>::new();
        let mut best_front_delta = f64::INFINITY;
        let mut best_new_max = u32::MAX;
        let mut best_new_total = f64::INFINITY;

        for &swap in &candidates {
            // front‑layer contribution
            let delta_front = basic_w * front.score_delta(swap, ctx.topology());
            if delta_front < best_front_delta {
                best_front_delta = delta_front;
            }

            // look‑ahead contribution
            let delta_ext = if !ext.is_empty() && lookahead_w != 0.0 {
                lookahead_w * ext.score_delta(swap, ctx.topology())
            } else {
                0.0
            };

            // immediate‑gain bonus and new maximal distance
            let mut saved_distance = 0.0;
            let mut new_max = max_before;
            for pair in front.nodes.iter() {
                let mut p1 = pair[0];
                let mut p2 = pair[1];
                if p1 == swap.0 {
                    p1 = swap.1;
                } else if p1 == swap.1 {
                    p1 = swap.0;
                }
                if p2 == swap.0 {
                    p2 = swap.1;
                } else if p2 == swap.1 {
                    p2 = swap.0;
                }
                let old_dist = ctx.topology().distance(pair[0], pair[1]);
                let new_dist = ctx.topology().distance(p1, p2);
                if new_dist > new_max {
                    new_max = new_dist;
                }
                if old_dist > 1 && new_dist == 1 {
                    // gate becomes executable – we save (old_dist‑1) distance units
                    saved_distance += (old_dist - 1) as f64;
                }
            }
            let immediate_gain_bonus = basic_w * immediate_gain_factor * saved_distance;

            // max‑distance term
            let max_dist_term = if new_max < max_before {
                -basic_w * max_dist_reduction_factor * ((max_before - new_max) as f64)
            } else if new_max > max_before {
                basic_w * max_dist_increase_factor * ((new_max - max_before) as f64)
            } else {
                0.0
            };

            // raw score before penalties
            let mut score = base_score + delta_front + delta_ext + max_dist_term - immediate_gain_bonus;

            // penalties for involving active / look‑ahead / idle qubits
            let a_active = front.is_active(swap.0);
            let b_active = front.is_active(swap.1);
            let a_ext = ext_qubits.contains(&swap.0);
            let b_ext = ext_qubits.contains(&swap.1);

            if a_active && b_active {
                score += basic_w * active_swap_penalty;
            } else if a_ext || b_ext {
                score += basic_w * ext_swap_penalty;
            } else {
                // both qubits are idle
                score += basic_w * idle_swap_penalty;
            }

            // penalty for undoing the previous swap
            if let Some(prev) = ctx.last_applied_swap() {
                if swap == prev {
                    score += basic_w * undo_swap_penalty;
                }
            }

            // apply decay scaling
            if self.use_decay {
                score *= self.decay_state[swap.0].max(self.decay_state[swap.1]);
            }

            // tie‑breaking logic
            if score + self.best_epsilon < best_score {
                best_score = score;
                best_swaps.clear();
                best_swaps.push(swap);
                best_front_delta = delta_front;
                best_new_max = new_max;
                best_new_total = base_score + delta_front + delta_ext - immediate_gain_bonus;
            } else if (score - best_score).abs() <= self.best_epsilon {
                // first tie‑break on maximal distance after the swap
                if new_max < best_new_max {
                    best_swaps.clear();
                    best_swaps.push(swap);
                    best_new_max = new_max;
                    best_new_total = base_score + delta_front + delta_ext - immediate_gain_bonus;
                } else if new_max == best_new_max {
                    // second tie‑break on total (un‑penalised) distance
                    let new_total = base_score + delta_front + delta_ext - immediate_gain_bonus;
                    if new_total < best_new_total {
                        best_swaps.clear();
                        best_swaps.push(swap);
                        best_new_total = new_total;
                    }
                }
            }
        }

        // -----------------------------------------------------------------
        // Release‑valve fallback: if no candidate improves the front layer,
        // advance the most distant front‑layer pair by one hop.
        // -----------------------------------------------------------------
        if best_front_delta >= -self.best_epsilon && !front.is_empty() {
            let mut max_dist = 0u32;
            let mut max_pair = (0usize, 0usize);
            for &nid in ctx.front_layer().node_ids() {
                if let Some((l_a, l_b)) = ctx.circuit().node(nid).two_qubit_pair() {
                    let p_a = ctx.layout().logical_to_physical()[l_a];
                    let p_b = ctx.layout().logical_to_physical()[l_b];
                    let d = ctx.topology().distance(p_a, p_b);
                    if d > max_dist {
                        max_dist = d;
                        max_pair = (p_a, p_b);
                    }
                }
            }
            if max_dist > 1 {
                let path = ctx.topology().shortest_path(max_pair.0, max_pair.1);
                if path.len() >= 2 {
                    let mut a = path[0];
                    let mut b = path[1];
                    if a > b {
                        std::mem::swap(&mut a, &mut b);
                    }
                    return Some((a, b));
                }
            }
        }

        // return a random swap among the best candidates
        Some(best_swaps[rng.gen_index(best_swaps.len())])
    }
}
// EVOLVE-BLOCK-END
