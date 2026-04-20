// EVOLVE-BLOCK-START
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetScaling {
    Constant,
    Size,
}

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
            let index = out.nodes.len();
            out.nodes.push([a, b]);
            out.qubits[a] = Some((index, b));
            out.qubits[b] = Some((index, a));
        }
        out
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn is_active(&self, qubit: usize) -> bool {
        self.qubits[qubit].is_some()
    }

    fn iter_active(&self) -> impl Iterator<Item = &usize> {
        self.nodes.iter().flatten()
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        self.nodes
            .iter()
            .map(|pair| topology.distance(pair[0], pair[1]) as f64)
            .sum()
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut delta = 0.0;
        if let Some((_, c)) = self.qubits[a] {
            delta += (topology.distance(b, c) as f64) - (topology.distance(a, c) as f64);
        }
        if let Some((_, c)) = self.qubits[b] {
            delta += (topology.distance(a, c) as f64) - (topology.distance(b, c) as f64);
        }
        delta
    }
}

#[derive(Debug, Clone)]
struct ExtendedSetScores {
    qubits: Vec<Vec<usize>>,
    len: usize,
}

impl ExtendedSetScores {
    fn new(num_qubits: usize) -> Self {
        Self {
            qubits: vec![Vec::new(); num_qubits],
            len: 0,
        }
    }

    fn push(&mut self, a: usize, b: usize) {
        self.qubits[a].push(b);
        self.qubits[b].push(a);
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        self.qubits
            .iter()
            .enumerate()
            .flat_map(|(a, others)| {
                others
                    .iter()
                    .map(move |b| topology.distance(a, *b) as f64)
            })
            .sum::<f64>()
            * 0.5
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for other in &self.qubits[a] {
            if *other == b {
                continue;
            }
            total += (topology.distance(b, *other) as f64) - (topology.distance(a, *other) as f64);
        }
        for other in &self.qubits[b] {
            if *other == a {
                continue;
            }
            total += (topology.distance(a, *other) as f64) - (topology.distance(b, *other) as f64);
        }
        total
    }
}

fn build_extended_set(ctx: &SwapSelectionContext<'_>, max_size: usize) -> ExtendedSetScores {
    let mut out = ExtendedSetScores::new(ctx.topology().num_qubits());
    if max_size == 0 {
        return out;
    }

    let precomputed = ctx.precomputed_extended_set_logical_pairs();
    if !precomputed.is_empty() {
        for pair in precomputed.iter().take(max_size) {
            out.push(
                ctx.layout().physical_of_logical(pair[0]),
                ctx.layout().physical_of_logical(pair[1]),
            );
        }
        return out;
    }

    let mut required_predecessors = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut to_visit = ctx.front_layer().node_ids().to_vec();
    let mut decremented = Vec::<(usize, usize)>::new();
    let mut i = 0usize;
    while i < to_visit.len() && out.len() < max_size {
        let node_id = to_visit[i];
        for &successor in ctx.circuit().node(node_id).successors() {
            if let Some((_, amount)) = decremented.iter_mut().find(|(idx, _)| *idx == successor) {
                *amount += 1;
            } else {
                decremented.push((successor, 1));
            }
            required_predecessors[successor] -= 1;
            if required_predecessors[successor] == 0 {
                if let Some((a, b)) = ctx.circuit().node(successor).two_qubit_pair() {
                    out.push(ctx.layout().physical_of_logical(a), ctx.layout().physical_of_logical(b));
                }
                to_visit.push(successor);
            }
        }
        i += 1;
    }

    out
}

fn enumerate_candidate_swaps(
    topology: TopologyView<'_>,
    front_layer: &FrontLayerScores,
) -> Vec<(usize, usize)> {
    let mut out = Vec::<(usize, usize)>::new();

    for &phys in front_layer.iter_active() {
        for &neighbor in topology.neighbors(phys) {
            if neighbor > phys || !front_layer.is_active(neighbor) {
                out.push((phys, neighbor));
            }
        }
    }

    out
}

fn choose_dense_layout_subset(
    topology: TopologyView<'_>,
    logical_component_size: usize,
    target_component: &[usize],
) -> Result<Vec<usize>, RouterError> {
    if logical_component_size > target_component.len() {
        return Err(RouterError::Routing(format!(
            "logical component size {logical_component_size} exceeds target component size {}",
            target_component.len()
        )));
    }
    if logical_component_size == target_component.len() {
        return Ok(target_component.to_vec());
    }

    let local_index = target_component
        .iter()
        .enumerate()
        .map(|(local, global)| (*global, local))
        .collect::<HashMap<usize, usize>>();
    let mut local_adj = Array2::<f64>::zeros((target_component.len(), target_component.len()));
    for &global_a in target_component {
        let a = local_index[&global_a];
        for &global_b in topology.neighbors(global_a) {
            if let Some(&b) = local_index.get(&global_b) {
                local_adj[[a, b]] = 1.0;
            }
        }
    }
    let error_matrix = Array2::<f64>::zeros((target_component.len(), target_component.len()));
    let [_, _, best_map] = dense_layout::best_subset(
        logical_component_size,
        local_adj.view(),
        0,
        0,
        false,
        true,
        error_matrix.view(),
    );
    let chosen = best_map
        .into_iter()
        .take(logical_component_size)
        .map(|local| target_component[local])
        .collect::<Vec<_>>();
    ensure_connected_subset(topology, &chosen)?;
    Ok(chosen)
}

fn ensure_connected_subset(topology: TopologyView<'_>, subset: &[usize]) -> Result<(), RouterError> {
    if subset.is_empty() {
        return Ok(());
    }
    let set = subset.iter().copied().collect::<HashSet<_>>();
    let mut seen = HashSet::<usize>::new();
    let mut queue = VecDeque::<usize>::new();
    queue.push_back(subset[0]);
    seen.insert(subset[0]);

    while let Some(node) = queue.pop_front() {
        for &next in topology.neighbors(node) {
            if set.contains(&next) && seen.insert(next) {
                queue.push_back(next);
            }
        }
    }

    if seen.len() != set.len() {
        return Err(RouterError::Routing(
            "selected layout subset is not connected".to_string(),
        ));
    }
    Ok(())
}

fn assign_components_to_target(
    logical_components: &[Vec<usize>],
    target_components: &[Vec<usize>],
) -> Result<Vec<(Vec<usize>, usize)>, RouterError> {
    if logical_components.is_empty() {
        return Ok(Vec::new());
    }

    let mut logical_sorted = logical_components.to_vec();
    logical_sorted.sort_by_key(|component| std::cmp::Reverse(component.len()));

    let mut target_sorted = target_components
        .iter()
        .enumerate()
        .map(|(idx, component)| (idx, component.len()))
        .collect::<Vec<_>>();
    target_sorted.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    let mut free_capacity = target_sorted
        .iter()
        .map(|(idx, size)| (*idx, *size))
        .collect::<HashMap<usize, usize>>();
    let mut assignments = Vec::<(Vec<usize>, usize)>::new();

    for logical in logical_sorted {
        let size = logical.len();
        let mut chosen = None;
        for (target_idx, _) in &target_sorted {
            let cap = free_capacity.get(target_idx).copied().unwrap_or(0);
            if cap >= size {
                chosen = Some(*target_idx);
                break;
            }
        }
        let Some(target_idx) = chosen else {
            return Err(RouterError::Routing(format!(
                "logical component of size {size} cannot fit any target component"
            )));
        };
        *free_capacity
            .get_mut(&target_idx)
            .expect("selected target component must exist") -= size;
        assignments.push((logical, target_idx));
    }
    Ok(assignments)
}

fn choose_disjoint_aware_layout(ctx: &InitialLayoutContext<'_>) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    let used = circuit.used_logical_qubits();
    if used.is_empty() {
        return Ok((0..num_logical).collect());
    }

    let logical_components = circuit.logical_interaction_components();
    let target_components = topology.connected_components();
    if target_components.is_empty() {
        return Err(RouterError::Routing(
            "topology has no connected components".to_string(),
        ));
    }

    let assignments = assign_components_to_target(logical_components, target_components)?;
    let mut mapping = vec![usize::MAX; num_logical];
    let mut used_physical = vec![false; topology.num_qubits()];

    for (logical_component, target_component_idx) in assignments {
        if logical_component.is_empty() {
            continue;
        }
        let target_component = &target_components[target_component_idx];
        let local = choose_dense_layout_subset(topology, logical_component.len(), target_component)?;
        for (logical, physical) in logical_component.iter().zip(local) {
            mapping[*logical] = physical;
            used_physical[physical] = true;
        }
    }

    let mut free_physical = (0..topology.num_qubits()).filter(|q| !used_physical[*q]);
    for slot in &mut mapping {
        if *slot == usize::MAX {
            *slot = free_physical.next().ok_or_else(|| {
                RouterError::Routing("not enough physical qubits to complete layout".to_string())
            })?;
        }
    }
    Ok(mapping)
}

#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    pub basic_weight: f64,
    pub lookahead_weight: f64,
    pub lookahead_size: usize,
    pub set_scaling: SetScaling,
    pub use_decay: bool,
    pub decay_increment: f64,
    pub decay_reset: usize,
    pub best_epsilon: f64,
    decay_state: Vec<f64>,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            basic_weight: 1.0,
            lookahead_weight: 0.5,
            lookahead_size: 20,
            set_scaling: SetScaling::Size,
            use_decay: true,
            decay_increment: 0.001,
            decay_reset: 5,
            best_epsilon: 1e-10,
            decay_state: Vec::new(),
        }
    }
}

impl CandidatePolicy {
    fn refresh_decay_state(&mut self, ctx: &SwapSelectionContext<'_>) {
        if !self.use_decay {
            return;
        }

        let num_qubits = ctx.topology().num_qubits();
        if self.decay_state.len() != num_qubits {
            self.decay_state = vec![1.0; num_qubits];
        }

        if ctx.swaps_since_progress() == 0 {
            self.decay_state.fill(1.0);
            return;
        }

        let Some((a, b)) = ctx.last_applied_swap() else {
            return;
        };
        let reset = self.decay_reset.max(1);
        if ctx.swaps_since_progress() % reset == 0 {
            self.decay_state.fill(1.0);
        } else {
            self.decay_state[a] += self.decay_increment;
            self.decay_state[b] += self.decay_increment;
        }
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        _rng: &mut RngState,
    ) -> Result<Vec<usize>, RouterError> {
        self.decay_state.clear();
        choose_disjoint_aware_layout(ctx)
    }

    fn choose_best_swap(
        &mut self,
        ctx: &SwapSelectionContext<'_>,
        rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        self.refresh_decay_state(ctx);

        let front_layer = FrontLayerScores::from_ctx(ctx);
        let candidates = enumerate_candidate_swaps(ctx.topology(), &front_layer);
        if candidates.is_empty() {
            return None;
        }

        let extended_set = build_extended_set(ctx, self.lookahead_size);

        let scale = |weight: f64, size: usize, scaling: SetScaling| -> f64 {
            match scaling {
                SetScaling::Constant => weight,
                SetScaling::Size => {
                    if size == 0 {
                        0.0
                    } else {
                        weight / (size as f64)
                    }
                }
            }
        };

        let basic_weight = scale(self.basic_weight, front_layer.len(), self.set_scaling);
        let lookahead_weight = scale(
            self.lookahead_weight,
            extended_set.len(),
            self.set_scaling,
        );

        let mut swap_scores = candidates
            .iter()
            .copied()
            .map(|swap| (swap, 0.0))
            .collect::<Vec<_>>();

        let mut absolute_score = 0.0;
        absolute_score += basic_weight * front_layer.total_score(ctx.topology());
        for (swap, score) in &mut swap_scores {
            *score += basic_weight * front_layer.score_delta(*swap, ctx.topology());
        }

        if !extended_set.is_empty() && self.lookahead_weight != 0.0 {
            absolute_score += lookahead_weight * extended_set.total_score(ctx.topology());
            for (swap, score) in &mut swap_scores {
                *score += lookahead_weight * extended_set.score_delta(*swap, ctx.topology());
            }
        }

        if self.use_decay {
            for (swap, score) in &mut swap_scores {
                *score =
                    (absolute_score + *score) * self.decay_state[swap.0].max(self.decay_state[swap.1]);
            }
        } else {
            for (_, score) in &mut swap_scores {
                *score = absolute_score + *score;
            }
        }

        let mut min_score = f64::INFINITY;
        let mut best_swaps = Vec::<(usize, usize)>::new();
        for (swap, score) in &swap_scores {
            if *score + self.best_epsilon < min_score {
                min_score = *score;
                best_swaps.clear();
                best_swaps.push(*swap);
                continue;
            }
            if (*score - min_score).abs() <= self.best_epsilon {
                best_swaps.push(*swap);
            }
        }

        Some(best_swaps[rng.gen_index(best_swaps.len())])
    }
}
// EVOLVE-BLOCK-END
